from tqdm import tqdm
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import os
from decepticon._losses import least_squares_gan_loss, build_style_model, compute_style_loss
from decepticon.loaders import image_loader_dataset



@tf.function
def maskgen_training_step(opt, inpt_img, maskgen, classifier, 
                          inpainter, cls_weight=1, exp_weight=0.1,
                          clip=10):
    """
    TensorFlow function to perform one training step on the mask generator.
    
    NOT currently set up for multi-class training.
    
    :opt: keras optimizer
    :input_img: batch of input images
    :maskgen: mask generator model
    :classifier: classifier model
    :inpainter: inpainting model
    :cls_weight: weight for classification loss (in paper: 12)
    :exp_weight: weight for exponential loss (in paper: 18)
    
    Returns
    :cls_loss: classification loss for the batch
    :exp_loss: exponential loss for the batch
    :loss: total weighted loss for the batch
    :mask: batch masks (for use in mask buffer)
    """
    with tf.GradientTape() as tape:
        # predict a mask from the original image and mask it
        mask = maskgen(inpt_img)
        inverse_mask = 1-mask
        masked_inpt = inpt_img*inverse_mask
        
        # fill in with inpainter
        inpainted = inpainter(masked_inpt)
        y = masked_inpt + mask*inpainted
        
        
        # run masked image through classifier
        softmax_out = classifier(y)
    
        # compute losses
        cls_loss = tf.reduce_mean(-1*tf.math.log(softmax_out[:,0] + K.epsilon()))
        exp_loss = tf.reduce_mean(
                            tf.exp(tf.reduce_mean(mask, axis=[1,2,3])))
        loss = cls_weight*cls_loss + exp_weight*exp_loss
        
    # compute gradients and update
    variables = maskgen.trainable_variables
    gradients = tape.gradient(loss, variables)
    if clip > 0:
        #gradients = [tf.clip_by_value(g, -1*clip_by_value, clip_by_value) 
        #                for g in gradients]
        gradients = [tf.clip_by_norm(g, clip) for g in gradients]
    opt.apply_gradients(zip(gradients, variables))
    
    return cls_loss, exp_loss, loss, mask




@tf.function
def inpainter_training_step(opt, inpt_img, mask, inpainter,
                            disc, recon_weight=100,
                            disc_weight=2, style_weight=0, style_model=None,
                            clip=10):
    """
    TensorFlow function to perform one training step on the inpainter.
    
    NOT currently set up for multi-class training.
    
    :opt: keras optimizer
    :input_img: batch of input images
    :mask: batch of masks from mask buffer
    :inpainter: inpainting model
    :disc: discriminator model
    :recon_weight: reconstruction loss weight
    :disc_weight: discriminator loss weight
    :style_weight: weight for style loss
    :style_model: model to use for computing style representation
    
    Returns
    :recon_loss: reconstruction loss for the batch
    :disc_loss: discriminator loss for the batch
    :loss: total weighted loss for the batch
    """
    with tf.GradientTape() as tape:
        # predict a mask from the original image and mask it
        #mask = maskgen(inpt_img)
        inverse_mask = 1 - mask
        masked_inpt = inpt_img*inverse_mask
        
        # fill in with inpainter
        inpainted = inpainter(masked_inpt)
        y = masked_inpt + mask*inpainted
        
        
        # run masked image through discriminator
        # note that outputs will be (batch_size, X, Y, 1)
        sigmoid_out = disc(y)
    
        # compute losses. style loss only computed if weight is nonzero
        recon_loss = tf.reduce_mean(tf.abs(y - inpt_img))
        disc_loss = tf.reduce_mean(-1*tf.math.log(1 - sigmoid_out + K.epsilon()))
        
        if (style_weight > 0)&(style_model is not None):
            style_loss = compute_style_loss(inpt_img, y, style_model)
        else:
            style_loss = 0
        
        loss = recon_weight*recon_loss + disc_weight*disc_loss + \
                style_weight*style_loss
        
    # compute gradients and update
    variables = inpainter.trainable_variables
    gradients = tape.gradient(loss, variables)
    if clip > 0:
        gradients = [tf.clip_by_norm(g, clip) for g in gradients]
    opt.apply_gradients(zip(gradients, variables))
    
    return recon_loss, disc_loss, style_loss, loss





@tf.function
def discriminator_training_step(opt, inpt_img, mask, inpainter,
                            disc, clip=10):
    """
    TensorFlow function to perform one training step on the discriminator.
    
    :opt: keras optimizer
    :input_img: batch of input images
    :mask: batch of masks from mask buffer
    :inpainter: inpainting model
    :disc: discriminator model
    :recon_weight: reconstruction loss weight
    :disc_weight: discriminator loss weight
    
    Returns
    :recon_loss: reconstruction loss for the batch
    :disc_loss: discriminator loss for the batch
    :loss: total weighted loss for the batch
    """
    with tf.GradientTape() as tape:
        # predict a mask from the original image and mask it
        #mask = maskgen(inpt_img)
        inverse_mask = 1 - mask
        masked_inpt = inpt_img*inverse_mask
        
        # fill in with inpainter
        inpainted = inpainter(masked_inpt)
        y = masked_inpt + mask*inpainted
        
        
        # run masked image through discriminator
        # note that outputs will be (batch_size, X, Y, 1)
        sigmoid_out = disc(y)
        
        # compute loss
        loss = least_squares_gan_loss(mask, sigmoid_out)
        
    # compute gradients and update
    variables = disc.trainable_variables
    gradients = tape.gradient(loss, variables)
    if clip > 0:
        gradients = [tf.clip_by_norm(g, clip) for g in gradients]
    opt.apply_gradients(zip(gradients, variables))
    
    return loss






class Trainer(object):
    """
    Prototype wrapper class for training inpainting network
    """
    
    def __init__(self, mask_generator, classifier, inpainter, discriminator,
                 mask_trainer_dataset, inpainter_dataset, lr=1e-4,
                 steps_per_epoch=100, batch_size=64,
                 class_loss_weight=1, exponential_loss_weight=0.1,
                 reconstruction_weight=100, disc_weight=2, style_weight=0,
                 eval_pos=None, logdir=None, save_models=True, clip=0):
        """
        :mask_generator: keras mask generator model
        :classifier: pretrained convnet for classifying images
        :inpainter: keras inpainting model
        :discriminator: Keras model for pixelwise real/fake discrimination
        :mask_trainer_dataset: tf.data.Dataset object generating positive example batches
        :inpainter_dataset: tf.data.Dataset object generating negative example batches
        :class_loss_weight: for mask generator, weight on classification loss
        :exponential_loss_weight: for mask generator, weight on exponential loss.
        :reconstruction_weight: weight for L1 reconstruction loss
        :disc_weight: weight for GAN loss on inpainter
        :style_weight: weight for neural style loss on inpainter
        :eval_pos: batch of positive images for evaluation
        :logdir: where to save tensorboard logs
        :save_models: whether to save each component model at the end of
                every epoch,
        :clip: gradient clipping (0 to diable)
        """
        self.global_step = tf.compat.v1.train.get_or_create_global_step()
        assert tf.executing_eagerly(), "eager execution must be enabled first"
        self.epoch = 0
        self.eval_pos = eval_pos
        self._save_models = save_models
        self.weights = {"class":class_loss_weight,
                        "exp":exponential_loss_weight,
                        "recon":reconstruction_weight,
                        "disc":disc_weight,
                        "style":style_weight}
        self._clip = clip
        if style_weight > 0:
            self._style_model = build_style_model()
        else:
            self._style_model = None
        
        if logdir is not None:
            self._summary_writer = tf.contrib.summary.create_file_writer(logdir,
                                            flush_millis=10000)
            self._summary_writer.set_as_default()
            
        self.logdir = logdir
            
        self._steps_per_epoch = steps_per_epoch
        self._batch_size = batch_size

        self.maskgen = mask_generator
        self.classifier = classifier
        self.inpainter = inpainter
        self.discriminator = discriminator

        self._ds_mask = mask_trainer_dataset
        self._ds_inpainter = inpainter_dataset
        self._optimizers = {
                x:tf.keras.optimizers.Adam(lr) for x in ["mask", "inpainter", 
                                          "discriminator"]}
        self._assemble_full_model()
                
    def _assemble_full_model(self):
        inpt = tf.keras.layers.Input((None, None, 3))
        mask = self.maskgen(inpt)
        inverse_mask = tf.keras.layers.Lambda(lambda x: 1-x)(mask)
        masked_im = tf.keras.layers.Multiply()([inpt, inverse_mask])
        inpainted = self.inpainter(masked_im)
        masked_inpainted = tf.keras.layers.Multiply()([inpainted, mask])
        reconstructed = tf.keras.layers.Add()([masked_im, masked_inpainted])
        self.full_model = tf.keras.Model(inpt, reconstructed)
        
        
    def fit(self, epochs=1):
        """
        Train the models for some number of epochs. Every epoch:
            
            1) run through the positive dataset to train the mask generator
            2) run through the negative dataset, with alternating batches:
                -even batches, update inpainter
                -odd batches, update discriminator
            3) record tensorboard summaries
        """
        # for each epoch
        for e in tqdm(range(epochs)):
            # one training epoch on mask generator, recording masks to buffer
            mask_buffer = []
            for e, x in enumerate(self._ds_mask):
                # run training step
                cls_loss, exp_loss, mask_loss, mask = maskgen_training_step(
                        self._optimizers["mask"], x, self.maskgen, 
                        self.classifier, self.inpainter,
                        self.weights["class"], self.weights["exp"],
                        clip=self._clip)
                # record batch of masks to buffer
                mask_buffer.append(mask)
                
                if e >= self._steps_per_epoch:
                    mask_buffer = np.concatenate(mask_buffer, axis=0)
                    break
                
            # one training epoch on the inpainter and discriminator:
            for e, x in enumerate(self._ds_inpainter):
                # randomly select a mask batch
                sample_indices = np.random.choice(np.arange(mask_buffer.shape[0]), 
                                          replace=False,
                                          size=x.get_shape().as_list()[0])
                mask = mask_buffer[sample_indices]
                
                # every other step: train inpainter
                if e % 2 == 0:
                    # run training step
                    recon_loss, disc_loss, style_loss, inpaint_loss = inpainter_training_step(
                            self._optimizers["inpainter"], x, mask, 
                            self.inpainter, self.discriminator, 
                            self.weights["recon"], self.weights["disc"],
                            self.weights["style"], self._style_model,
                            clip=self._clip)
                # alternating step train discriminator
                else:
                    # run training step
                    disc_loss = discriminator_training_step(
                            self._optimizers["discriminator"],
                            x, mask, self.inpainter, self.discriminator,
                            clip=self._clip)
                if e >= self._steps_per_epoch:
                    mask_buffer = np.concatenate(mask_buffer, axis=0)
                    break
                
            # end of epoch-  record summary from last training batch
            with tf.contrib.summary.always_record_summaries():
                tf.contrib.summary.scalar("mask_generator_total_loss", mask_loss,
                                      step=self.global_step)
                tf.contrib.summary.scalar("mask_generator_classifier_loss", cls_loss,
                                      step=self.global_step)
                tf.contrib.summary.scalar("mask_generator_exponential_loss", exp_loss,
                                      step=self.global_step)
                tf.contrib.summary.scalar("inpainter_style_loss", style_loss, 
                                      step=self.global_step)
                tf.contrib.summary.scalar("inpainter_total_loss", inpaint_loss, 
                                      step=self.global_step)
                tf.contrib.summary.scalar("inpainter_reconstruction_L1_loss", recon_loss,
                                      step=self.global_step)
                tf.contrib.summary.scalar("discriminator_GAN_loss", disc_loss,
                                      step=self.global_step)
            # also record summary images
            if self.eval_pos is not None:
                self.evaluate()
                
            # save all the component models
            if (self.logdir is not None) & self._save_models:
                self.maskgen.save(os.path.join(self.logdir, "mask_generator.h5"))
                self.inpainter.save(os.path.join(self.logdir, "inpainter.h5"))
                self.discriminator.save(os.path.join(self.logdir, "discriminator.h5"))
                self.classifier.save(os.path.join(self.logdir, "classifier.h5"))
            
            self.global_step.assign_add(1)
                
                
    
    def evaluate(self):
        """
        Run a set of evaluation metrics. Requires validation data and a log
        directory to be set.
        """
        x_pos = self.eval_pos

                
        # also visualize predictions for the positive cases
        predicted_masks = self.maskgen.predict(x_pos)
        predicted_inpaints = self.inpainter.predict(x_pos*(1-predicted_masks))
        reconstructed = x_pos*(1-predicted_masks) + \
                        predicted_masks*predicted_inpaints
    
        rgb_masks = np.concatenate([predicted_masks]*3, -1)
        concatenated = np.concatenate([x_pos, rgb_masks, 
                                       predicted_inpaints, reconstructed],
                                axis=2)
        # also, run the classifier on the reconstructed images and
        # see whether it thinks an object is present
        cls_outs = self.classifier.predict(reconstructed)
        object_score = 1 - cls_outs[:,0]
        # while we're at it let's see about testing the discriminator too
        dsc_outs = self.discriminator.predict(reconstructed)
        dsc_mask = dsc_outs[predicted_masks.astype(bool)]
        dsc_unmask = dsc_outs[(1-predicted_masks).astype(bool)]
        
        
        # record everything
        with tf.contrib.summary.always_record_summaries():
            tf.contrib.summary.image("input__mask__inpainted__reconstructed", 
                             concatenated, max_images=5,
                             step=self.global_step)
            tf.contrib.summary.histogram("reconstructed_classifier_score", 
                                         object_score,
                                         step=self.global_step)
            tf.contrib.summary.histogram("discriminator_score_real", 
                                         dsc_unmask,
                                         step=self.global_step)
            tf.contrib.summary.histogram("discriminator_score_fake", 
                                         dsc_mask,
                                         step=self.global_step)
   
   
   





def build_image_file_trainer(positive_filepaths, negative_filepaths,
                             classifier,
                             mask_generator=None, 
                             inpainter=None, 
                             discriminator=None,
                             lr=1e-4, steps_per_epoch=None, batch_size=64,
                             class_loss_weight=1, exponential_loss_weight=0.1,
                             reconstruction_weight=100, disc_weight=2,
                             logdir=None, save_models=True,
                             num_parallel_calls=2,
                             train_maskgen_on_all=False):
        """
        Macro to set up a trainer for image files
        
        :positive_filepaths: list of paths to class-1 files
        :negative_filepaths: list of paths to class-0 files
        :classifier: pretrained convnet for classifying images
        :mask_generator: keras mask generator model
        :inpainter: keras inpainting model
        :discriminator: Keras model for pixelwise real/fake discrimination
        :mask_trainer_dataset: tf.data.Dataset object generating positive example batches
        :inpainter_dataset: tf.data.Dataset object generating negative example batches
        :class_loss_weight: for mask generator, weight on classification loss
        :exponential_loss_weight: for mask generator, weight on exponential loss.
        :reconstruction_weight: weight for L1 reconstruction loss
        :disc_weight: weight for GAN loss on inpainter
        :logdir: where to save tensorboard logs
        :save_models: whether to save each component model at the end of
                every epoch
        :train_maskgen_on_all: whether to train the mask generator on all
                images instead of just positive ones
        """
        if steps_per_epoch is None:
            steps_per_epoch = np.floor(
                (len(positive_filepaths)+len(negative_filepaths))/(2*batch_size))
        # build dataset loaders
        if train_maskgen_on_all:
            ds_pos = image_loader_dataset(positive_filepaths+negative_filepaths,
                                          batch_size=batch_size,
                                      num_parallel_calls=num_parallel_calls)
        else:
            ds_pos = image_loader_dataset(positive_filepaths, batch_size=batch_size,
                                      num_parallel_calls=num_parallel_calls)
        ds_neg = image_loader_dataset(positive_filepaths, batch_size=batch_size,
                                      num_parallel_calls=num_parallel_calls)
        # pull out one batch for tensorboard visualizations
        for x in ds_pos:
            eval_pos = x.numpy()
            break
        
        # build any models that were missing
        if mask_generator is None:
            from decepticon._models import build_mask_generator
            mask_generator = build_mask_generator()
        if inpainter is None:
            from decepticon._models import build_inpainter
            inpainter = build_inpainter()
        if discriminator is None:
            from decepticon._models import build_discriminator
            discriminator = build_discriminator()
            
        return Trainer(mask_generator, classifier, inpainter, discriminator,
                 ds_pos, ds_neg, lr=lr, steps_per_epoch=steps_per_epoch, 
                 batch_size=batch_size, class_loss_weight=class_loss_weight,
                 exponential_loss_weight=exponential_loss_weight,
                 reconstruction_weight=reconstruction_weight, 
                 disc_weight=disc_weight, eval_pos=eval_pos, 
                 logdir=logdir, save_models=save_models)
        
