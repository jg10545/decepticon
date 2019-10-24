from tqdm import tqdm
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import os
import yaml


import decepticon
from decepticon._losses import least_squares_gan_loss, build_style_model, compute_style_loss
from decepticon._losses import pixelwise_variance, total_variation_loss
from decepticon.loaders import image_loader_dataset, classifier_training_dataset
from decepticon.loaders import inpainter_training_dataset, circle_mask_dataset
from decepticon._descriptions import loss_descriptions
from decepticon import _descriptions

maskgen_lossnames = ["mask_generator_classifier_loss",
                     "mask_generator_exponential_loss",
                     "mask_generator_prior_loss",
                     "mask_generator_tv_loss",
                     "mask_generator_total_loss"]
        
inpaint_lossnames = ["inpainter_reconstruction_L1_loss",
                     "inpainter_discriminator_loss",
                     "inpainter_style_loss",
                     "inpainter_tv_loss",
                     "inpainter_total_loss",
                     "discriminator_GAN_loss"]



@tf.function
def maskgen_training_step(opt, inpt_img, maskgen, classifier, 
                          inpainter, maskdisc=None, cls_weight=1, exp_weight=0.1,
                          prior_weight=0.25, tv_weight=0, inpaint=True):
    """
    TensorFlow function to perform one training step on the mask generator.
    
    NOT currently set up for multi-class training.
    
    :opt: keras optimizer
    :input_img: batch of input images
    :maskgen: mask generator model
    :classifier: classifier model
    :inpainter: inpainting model
    :maskdisc: mask discriminator model, if using one
    :cls_weight: weight for classification loss (in paper: 12)
    :exp_weight: weight for exponential loss (in paper: 18)
    :prior_weight: weight for mask discriminator loss (in paper: 3)
    :tv_weight: weight total variation loss (not in paper)
    :inpaint: if True, fill in the mask using the inpainter before computing
        classification loss (the way they did it in the paper)
    
    Returns
    :cls_loss: classification loss for the batch
    :exp_loss: exponential loss for the batch
    :prior_loss: loss from the mask discriminator for the batch
    :tv_loss: total variation loss from the batch
    :loss: total weighted loss for the batch
    :mask: batch masks (for use in mask buffer)
    """
    input_shape = inpt_img.get_shape()
    tv_norm = input_shape[1]*input_shape[2]
    
    with tf.GradientTape() as tape:
        # predict a mask from the original image and mask it
        mask = maskgen(inpt_img)
        inverse_mask = 1-mask
        masked_inpt = inpt_img*inverse_mask 
        
        if inpaint:
            # fill in with inpainter
            inpainted = inpainter(tf.concat([masked_inpt, mask], 3))
            y = masked_inpt + mask*inpainted
        else:
            y = masked_inpt
        # run masked image through classifier
        softmax_out = classifier(y)
            
    
        # compute losses
        cls_loss = tf.reduce_mean(-1*tf.math.log(softmax_out[:,0] + K.epsilon()))
        exp_loss = tf.reduce_mean(
                            tf.exp(tf.reduce_mean(mask, axis=[1,2,3])))
        if (prior_weight > 0) & (maskdisc is not None):
            prior_loss = -1*tf.reduce_sum(maskdisc(mask))
        else:
            prior_loss = 0
            
        if tv_weight > 0:
            tv_loss = total_variation_loss(mask)/tv_norm
        else:
            tv_loss = 0
        loss = cls_weight*cls_loss + exp_weight*exp_loss + \
                    prior_weight*prior_loss + tv_weight*tv_loss
        
    # compute gradients and update
    variables = maskgen.trainable_variables
    gradients = tape.gradient(loss, variables)
    opt.apply_gradients(zip(gradients, variables))
    
    return cls_loss, exp_loss, prior_loss, tv_loss, loss, mask


@tf.function
def mask_discriminator_training_step(maskdisc, mask, prior_sample, opt):
    """
    TensorFlow function to perform one training step for the mask discriminator.
    
    :maskdisc: mask discriminator model
    :mask: batch of masks generated by mask generator
    :prior_sample: sample from prior distribution over masks
    :opt: keras optimizer
    """
    with tf.GradientTape() as tape:
        # compute losses with respect to actual masks and samples
        # from the mask prior
        gen_loss = tf.reduce_mean(maskdisc(mask))
        prior_loss = tf.reduce_mean(maskdisc(prior_sample))
        wgan_loss = gen_loss - prior_loss
                
    wgan_grads = tape.gradient(wgan_loss, maskdisc.trainable_variables)
    opt.apply_gradients(zip(wgan_grads, maskdisc.trainable_variables))
    return wgan_loss
    


@tf.function
def inpainter_training_step(inpaint_opt, disc_opt, inpt_img, mask, inpainter,
                            disc, recon_weight=100,
                            disc_weight=2, style_weight=0, 
                            tv_weight=0, style_model=None):
    """
    TensorFlow function to perform one training step on the inpainter as well
    as the inpainter's discriminator.
    
    NOT currently set up for multi-class training.
    
    :inpaint_opt: keras optimizer for inpainter
    :disc_opt: keras optimizer for discriminator
    :input_img: batch of input images
    :mask: batch of masks from mask buffer
    :inpainter: inpainting model
    :disc: discriminator model
    :recon_weight: reconstruction loss weight (equations 6, 9)
    :disc_weight: discriminator loss weight (equations 7, 9)
    :style_weight: weight for style loss (equations 6, 9)
    :tv_weight: weight for total variation loss (equation 9)
    :style_model: model to use for computing style representation
    
    Returns
    :recon_loss: reconstruction loss for the batch
    :disc_loss: inpainter loss from the discriminator 
    :style_loss: neural style loss for the batch
    :tv_loss: total variation loss for the batch
    :loss: total weighted loss for the batch
    :d_loss: discriminator loss
    """
    input_shape = inpt_img.get_shape()
    tv_norm = input_shape[1]*input_shape[2]*input_shape[3]
    
    with tf.GradientTape() as inp_tape, tf.GradientTape() as disc_tape:
        # predict a mask from the original image and mask it
        inverse_mask = 1 - mask
        masked_inpt = inpt_img*inverse_mask 
        
        # fill in with inpainter
        inpainted = inpainter(tf.concat([masked_inpt, mask], 3))
        y = masked_inpt + mask*inpainted
        
        # run masked image through discriminator
        # note that outputs will be (batch_size, X, Y, 1)
        sigmoid_out = disc(y)
    
        # ----- INPAINTER LOSSES --------
        # compute losses. style loss only computed if weight is nonzero
        recon_loss = tf.reduce_mean(tf.abs(y - inpt_img))
        # this is the discriminator's loss ON THE INPAINTER
        disc_loss = tf.reduce_mean(-1*tf.math.log(1 - sigmoid_out + K.epsilon()))
        
        if (style_weight > 0)&(style_model is not None):
            style_loss = compute_style_loss(inpt_img, y, style_model)
        else:
            style_loss = 0
        if tv_weight > 0:
            tv_loss = total_variation_loss(y)/tv_norm
        else:
            tv_loss = 0
        
        loss = recon_weight*recon_loss + disc_weight*disc_loss + \
                style_weight*style_loss + tv_weight*tv_loss
                
        # ----- DISCRIMINATOR LOSSES --------
        d_loss = tf.reduce_mean(least_squares_gan_loss(mask, sigmoid_out))
        
    # compute gradients and update
    inp_variables = inpainter.trainable_variables
    inp_gradients = inp_tape.gradient(loss, inp_variables)
    inpaint_opt.apply_gradients(zip(inp_gradients, inp_variables))
    
    disc_variables = disc.trainable_variables
    disc_gradients = disc_tape.gradient(d_loss, disc_variables)
    disc_opt.apply_gradients(zip(disc_gradients, disc_variables))
    
    return recon_loss, disc_loss, style_loss, tv_loss, loss, d_loss











class Trainer(object):
    """
    Prototype wrapper class for training inpainting network
    """
    
    def __init__(self, posfiles, negfiles, mask_generator=None, 
                 classifier=None, inpainter=None, discriminator=None,
                 maskdisc=None,
                 lr=1e-4,
                 lr_decay=0,
                 batch_size=64,
                 class_loss_weight=1, exponential_loss_weight=0.1,
                 reconstruction_weight=100, disc_weight=2, style_weight=0,
                 prior_weight=0, inpainter_tv_weight=0, maskgen_tv_weight=0,
                 eval_pos=None, logdir=None, clip=10,
                 train_maskgen_on_all=False,
                 num_parallel_calls=4, imshape=(80,80),
                 downsample=2, step=0, inpaint=True, random_buffer=False):
        """
        :posfiles: list of paths to positive image patches
        :negfiles: list of paths to negative image patches
        :mask_generator: keras mask generator model
        :classifier: pretrained convnet for classifying images
        :inpainter: Keras inpainting model
        :discriminator: Keras model for pixelwise real/fake discrimination
        :maskdisc: Keras model for mask discriminator
        :lr: learning rate
        :lr_decay: learning rate decay. set to 0 for no decay; otherwise specify
            the half-life of the learning rate.
        :class_loss_weight: for mask generator, weight on classification loss
        :exponential_loss_weight: for mask generator, weight on exponential loss.
        :reconstruction_weight: weight for L1 reconstruction loss
        :disc_weight: weight for GAN loss on inpainter
        :style_weight: weight for neural style loss on inpainter
        :prior_weight: weight for mask discriminator loss
        :inpainter_tv_weight: weight for total variation loss for inpainter
        :maskgen_tv_weight: weight for total variation loss for mask generator
        :eval_pos: batch of positive images for evaluation
        :logdir: where to save tensorboard logs
        :clip: gradient clipping
        :train_maskgen_on_all: whether to train the mask generator using negative
                image patches as well as positive
        :num_parallel_calls: number of threads to use for data loaders
        :imshape: image dimensions- only matter for pretraining steps
        :downsample: factor to downsample new models by if not passed to
                constructor
        :step: initial training step value
        :inpaint: if True, fill in the mask using the inpainter during the mask
                generator training step (like they did in the paper)
        :random_buffer: use mask prior instead of actual masks for training inpainter
        """
        assert tf.executing_eagerly(), "eager execution must be enabled first"
        self.step = step
        self._batch_size = batch_size
        self._clip = clip
        self._inpaint = inpaint
        self._random_buffer = random_buffer

        # ------ RECORD LOSS FUNCTION WEIGHTS ------
        self.weights = {"class":class_loss_weight,
                        "exp":exponential_loss_weight,
                        "recon":reconstruction_weight,
                        "disc":disc_weight,
                        "style":style_weight,
                        "prior":prior_weight,
                        "inpaint_tv":inpainter_tv_weight,
                        "maskgen_tv":maskgen_tv_weight}
        
        
        # ------ BUILD SUMMARY WRITER ------
        if logdir is not None:
            self._summary_writer = tf.compat.v2.summary.create_file_writer(logdir,
                                            flush_millis=10000)
            self._summary_writer.set_as_default()
            
        self.logdir = logdir
        

        # ------ BUILD ANY MODELS NOT GIVEN ------
        if mask_generator is None: 
            mask_generator = decepticon.build_mask_generator(downsample=downsample)
        if classifier is None:
            classifier = decepticon.build_classifier(downsample=downsample)
        if inpainter is None:
            inpainter = decepticon.build_inpainter(input_shape=(None, None, 4),
                                                   downsample=downsample)
        if discriminator is None:
            discriminator = decepticon.build_discriminator(downsample=downsample)
        if (maskdisc is None) & (prior_weight > 0):
            maskdisc = decepticon.build_mask_discriminator(downsample=downsample)
            
        if style_weight > 0:
            self._style_model = build_style_model()
        else:
            self._style_model = None
            
        self.maskgen = mask_generator
        self.classifier = classifier
        self.inpainter = inpainter
        self.discriminator = discriminator
        self.maskdisc = maskdisc

        self._posfiles = posfiles
        self._negfiles = negfiles
        self._train_maskgen_on_all = train_maskgen_on_all
        self._num_parallel_calls = num_parallel_calls
        
        # ------ SET UP OPTIMIZERS ------
        self._optimizers = {}
        for x in ["mask", "inpainter", "discriminator", "maskdisc"]:
            if lr_decay == 0:
                learnrate = lr
            else:
                learnrate = tf.keras.optimizers.schedules.ExponentialDecay(
                        lr, decay_steps=lr_decay,
                        decay_rate=0.5,
                        staircase=False)
            self._optimizers[x] = tf.keras.optimizers.Adam(learnrate, clipnorm=clip) 
            
        self._lr = lr
        self._lr_decay = lr_decay
        self._assemble_full_model()
        self._imshape = imshape
        if eval_pos is None:
            eval_pos = self._build_eval_image_batch()
        self.eval_pos = eval_pos  
        self._save_config()
        # save an initial set of images and histograms for comparison
        self.evaluate()
        
    def _build_eval_image_batch(self):
        """
        Fetch a batch of images to use for tensorboard visualization
        """
        # how many positive images do we have?
        N = len(self._posfiles)
        bs = self._batch_size
        delta = min(int(np.floor(N/bs)),1)
        files = self._posfiles[::delta][:bs]
        ds = image_loader_dataset(files, 
                                  batch_size=bs,
                                  repeat=False,
                                  shuffle=1,
                                  augment=False) 
        for x in ds:
            x = x.numpy()
            break
        return x
        
    def _maskgen_dataset(self, train_on_all=True, maxnum=None):
        """
        Build a tensorflow dataset for training the mask generator
        """
        if train_on_all:
            files = self._posfiles + self._negfiles
        else:
            files = self._posfiles
        # shuffle so pos and neg patches won't be separated (if the
        # dataset is larger than the shuffle queue)
        np.random.shuffle(files)
        if maxnum is not None:
            files = files[:maxnum]
        ds = image_loader_dataset(files, batch_size=self._batch_size,
                                      num_parallel_calls=self._num_parallel_calls)        
        return ds
    
    def _inpainter_dataset(self):
        """
        Build a tensorflow dataset for training the inpainter
        """
        ds = image_loader_dataset(self._negfiles, batch_size=self._batch_size,
                                      num_parallel_calls=self._num_parallel_calls)        
        return ds
    
    def _classifier_dataset(self):
        """
        Build a tensorflow dataset for pretraining the classifier
        """
        steps_per_epoch = int(np.floor((len(self._posfiles)+len(self._negfiles))/self._batch_size))
        ds = classifier_training_dataset(self._posfiles, self._negfiles,
                                         batch_size=self._batch_size,
                                         imshape=self._imshape)
        return ds, steps_per_epoch
    
    
    def pretrain_classifier(self, epochs=1):
        """
        Pretrain the classifier model
        """
        ds, steps_per_epoch = self._classifier_dataset()
        self.classifier.compile(tf.keras.optimizers.Adam(1e-3),
                                loss=tf.keras.losses.sparse_categorical_crossentropy,
                                metrics=["accuracy"])
        self.classifier.fit(ds, steps_per_epoch=steps_per_epoch, epochs=epochs)
        
    def pretrain_inpainter(self, epochs=1):
        """
        Pretrain the inpainter
        """
        steps_per_epoch = int(np.floor(len(self._negfiles)/self._batch_size))
        ds = inpainter_training_dataset(self._negfiles, batch_size=self._batch_size,
                                        imshape=self._imshape)
        self.inpainter.compile(tf.keras.optimizers.Adam(1e-3),
                               loss=tf.keras.losses.mae)
        self.inpainter.fit(ds, steps_per_epoch=steps_per_epoch, epochs=epochs)
                
    def _assemble_full_model(self):
        inpt = tf.keras.layers.Input((None, None, 3))
        mask = self.maskgen(inpt)
        inverse_mask = tf.keras.layers.Lambda(lambda x: 1-x)(mask)
        masked_im = tf.keras.layers.Multiply()([inpt, inverse_mask])
        # inpainter_mask_val changes here
        #scaled_mask = tf.keras.layers.Lambda(lambda x: self._inpainter_mask_val*x)(mask)
        #masked_im = tf.keras.layers.Add()([masked_im, scaled_mask])
        concatenated_masked_im = tf.keras.layers.Concatenate(axis=-1)([masked_im, mask])
        
        inpainted = self.inpainter(concatenated_masked_im)
        masked_inpainted = tf.keras.layers.Multiply()([inpainted, mask])
        reconstructed = tf.keras.layers.Add()([masked_im, masked_inpainted])
        self.full_model = tf.keras.Model(inpt, reconstructed)
        
        
    def _run_maskgen_training_step(self, x):
        # wrapper for maskgen_training_step
        *maskgen_losses, mask = maskgen_training_step(
                self._optimizers["mask"], x, self.maskgen, 
                self.classifier, self.inpainter,
                maskdisc=self.maskdisc,
                cls_weight=self.weights["class"], 
                exp_weight=self.weights["exp"],
                prior_weight=self.weights["prior"],
                tv_weight=self.weights["maskgen_tv"],
                inpaint=self._inpaint)
        maskgen_losses = dict(zip(maskgen_lossnames, maskgen_losses))
        return maskgen_losses, mask

    def _run_mask_discriminator_training_step(self, mask, prior_sample):
        # wrapper for mask_discriminator_training_step
        if (self.weights["prior"] > 0)&(self.maskdisc is not None):
            loss = mask_discriminator_training_step(
                    self.maskdisc, mask, 
                    prior_sample, 
                    self._optimizers["maskdisc"])
        else:
            loss = 0
        return loss
    
    def _run_inpainter_training_step(self, x, mask):
        # wrapper for inpainter_training_step
        inpainter_losses = inpainter_training_step(
                self._optimizers["inpainter"], 
                self._optimizers["discriminator"],
                x, mask, 
                self.inpainter, self.discriminator, 
                recon_weight=self.weights["recon"], 
                disc_weight=self.weights["disc"],
                style_weight=self.weights["style"], 
                tv_weight=self.weights["inpaint_tv"],
                style_model=self._style_model)
        inpainter_losses = dict(zip(inpaint_lossnames, inpainter_losses))
        return inpainter_losses
    

    
    def _build_mask_buffer(self, maxsize=10000):
        """
        
        """
        mask_buffer = []
        if self._random_buffer:
            ds = circle_mask_dataset(self._imshape, 
                                        batch_size=self._batch_size, 
                                        prefetch=1)
            for e, x in enumerate(ds):
                mask_buffer.append(x.numpy())
                if (e+1)*self._batch_size >= maxsize:
                    break
        else:
            for x in self._maskgen_dataset(False, maxsize):
                mask_buffer.append(self.maskgen(x).numpy())
        return np.concatenate(mask_buffer, axis=0)
    
    def fit_mask_generator(self, epochs=1, progressbar=True, evaluate=True):
        """
        Train the mask generator only.
        
        :epochs: number of times to pass through the dataset
        :progressbar: whether to display a tqdm progress bar during training
        :evaluate: whether to record images and histograms after each epoch
        """
        # build mask generator dataset
        ds_maskgen = self._maskgen_dataset(self._train_maskgen_on_all)
        # incorporate mask prior for mask generator training
        ds_circle = circle_mask_dataset(self._imshape, 
                                        batch_size=self._batch_size, 
                                        prefetch=1)
        ds_maskgen = tf.data.Dataset.zip((ds_maskgen, ds_circle))

        if progressbar:
            iterator = tqdm(range(epochs))
        else:
            iterator = range(epochs)
        # for each epoch
        for e in iterator:
            # one training epoch on mask generator
            for x, prior_sample in ds_maskgen:
                # run mask generator training step
                maskgen_losses, mask = self._run_maskgen_training_step(x)
                # record the mean pixelwise mask variance
                self._record_losses(mask_variance=pixelwise_variance(mask))
                # run the mask discriminator step (if there is one)
                mdl = self._run_mask_discriminator_training_step(mask, prior_sample)
                maskgen_losses["mask_discriminator_loss"] = mdl
                    
                # record losses to tensorboard
                self._record_losses(**maskgen_losses)
                self.step += 1
            if evaluate:
                self.evaluate()

    def fit_inpainter(self, epochs=1, progressbar=True, 
                      mask_buffer_max_size=10000, evaluate=True):
        """
        Train the mask generator only.
        
        :epochs: number of times to pass through the dataset
        :progressbar: whether to display a tqdm progress bar during training
        :mask_buffer_max_size: maximum number of examples to compute for the
            mask buffer
        :evaluate: whether to record images and histograms after each epoch
        """
        ds_inpainter = self._inpainter_dataset()
        mask_buffer = self._build_mask_buffer(mask_buffer_max_size)
        
        if progressbar:
            iterator = tqdm(range(epochs))
        else:
            iterator = range(epochs)
        # for each epoch
        for e in iterator:
            # one training epoch on the inpainter and discriminator:
            for i,x in enumerate(ds_inpainter):
                # randomly select a mask batch
                sample_indices = np.random.choice(np.arange(mask_buffer.shape[0]), 
                                          replace=False,
                                          size=x.get_shape().as_list()[0])
                mask = mask_buffer[sample_indices]
                
                inpainter_losses = self._run_inpainter_training_step(x, mask)
                self._record_losses(**inpainter_losses)
                self.step += 1
                
            if evaluate:
                self.evaluate()
                


    def fit(self, epochs=1, mask_buffer_max_size=10000, maskgen_epochs=1,
            inpainter_epochs=1, evaluate=True, save=True):
        """
        Train the models for some number of epochs. Every epoch:
            
            1) run through the positive dataset to train the mask generator
            2) run through the negative dataset, with alternating batches:
                -even batches, update inpainter
                -odd batches, update discriminator
            3) record tensorboard summaries
        
        :epochs: number of training rounds (1 pass through the dataset on
                mask generator, one pass on inpainter per epoch)
        :mask_buffer_max_size: maximum number of examples to compute for the
            mask buffer
        :maskgen_epochs: number of times to pass through dataset on mask generator,
            per epoch
        :inpainter_epochs: number of times to pass through dataset on inpainter,
            per epoch
        :evaluate: whether to save images and histograms after each half-epoch
        :save: whether to save models after each epoch
        """
        for e in tqdm(range(epochs)):
            self.fit_mask_generator(epochs=maskgen_epochs, 
                                    progressbar=False, 
                                    evaluate=evaluate)
            self.fit_inpainter(epochs=inpainter_epochs,
                               progressbar=False,
                               mask_buffer_max_size=mask_buffer_max_size,
                               evaluate=evaluate)
            
            # save all the component models
            if (save) and (self.logdir is not None):
                self.maskgen.save(os.path.join(self.logdir, "mask_generator.h5"))
                self.inpainter.save(os.path.join(self.logdir, "inpainter.h5"))
                self.discriminator.save(os.path.join(self.logdir, "discriminator.h5"))
                self.classifier.save(os.path.join(self.logdir, "classifier.h5"))
                if self.maskdisc is not None:
                    self.maskdisc.save(os.path.join(self.logdir, "mask_discriminator.h5"))

                
    def _record_losses(self, **lossvals):
        # expect a dictionary of loss names to values
        for lossname in lossvals:
            tf.compat.v2.summary.scalar(lossname, lossvals[lossname],
                                      step=self.step, 
                                      description=loss_descriptions[lossname])
          
    
    def evaluate(self):
        """
        Run a set of evaluation metrics. Requires validation data and a log
        directory to be set.
        """
        x = self.eval_pos
                
        # visualize predictions for the positive cases
        mask = self.maskgen.predict(x)
        inpainted = self.inpainter.predict(tf.concat([x*(1-mask), mask], -1) )
        reconstructed = x*(1-mask) + mask*inpainted
    
        rgb_masks = np.concatenate([mask]*3, -1)
        concatenated = np.concatenate([x, rgb_masks, 
                                       inpainted,
                                       reconstructed],
                                axis=2)
        # also, run the classifier on the reconstructed images and
        # see whether it thinks an object is present
        cls_outs = self.classifier.predict(reconstructed)
        object_score = 1 - cls_outs[:,0]
        # while we're at it let's see about testing the discriminator too
        dsc_outs = self.discriminator.predict(reconstructed)
        dsc_mask = dsc_outs[mask.astype(bool)]
        dsc_unmask = dsc_outs[(1-mask).astype(bool)]
        
        
        # record everything
        tf.compat.v2.summary.image("input__mask__inpainted__reconstructed", 
                             concatenated, max_outputs=5,
                             step=self.step,
                             description=_descriptions.image_display)
        tf.compat.v2.summary.histogram("reconstructed_classifier_score", 
                                         object_score,
                                         step=self.step,
                                         description=_descriptions.reconstructed_classifier_score)
        tf.compat.v2.summary.histogram("discriminator_score_real", 
                                         dsc_unmask,
                                         step=self.step,
                                         description=_descriptions.disc_score_real)
        tf.compat.v2.summary.histogram("discriminator_score_fake", 
                                         dsc_mask,
                                         step=self.step,
                                         description=_descriptions.disc_score_fake )
            
    def _save_config(self):
        """
        Macro to write training config to document the experiment
        """
        config = {
                "batch_size":self._batch_size,
                "class_loss_weight":self.weights["class"],
                "exponential_loss_weight":self.weights["exp"],
                "reconstruction_weight":self.weights["recon"],
                "disc_weight":self.weights["disc"],
                "style_weight":self.weights["style"],
                "prior_weight":self.weights["prior"],
                "inpainter_tv_weight":self.weights["inpaint_tv"],
                "maskgen_tv_weight":self.weights["maskgen_tv"],
                "clip":self._clip,
                "imshape":self._imshape,
                "train_maskgen_on_all":self._train_maskgen_on_all,
                "lr":self._lr,
                "lr_decay":self._lr_decay,
                "num_parallel_calls":self._num_parallel_calls,
                "inpaint":self._inpaint
                }
        config_path = os.path.join(self.logdir, "config.yml")
        yaml.dump(config, open(config_path, "w"), default_flow_style=False)
   
   


