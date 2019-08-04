from tqdm import tqdm
import numpy as np
import tensorflow as tf
from decepticon._models import _compose
from decepticon._losses import exponential_loss, least_squares_gan_loss




def build_mask_generator_trainer(mask_generator, classifier, inpainter,
                                 lr=1e-3, class_loss_weight=12,
                                 exponential_loss_weight=18):
    """
    Build a compiled Keras model for training the mask generator. This
    implementation does not include the paper's Wasserstein GAN component.
    
    :mask_generator: keras mask generator model
    :classifier: pretrained convnet for classifying images
    :inpainter: keras inpainting model
    :lr: learning rate for ADAM optimizer
    :class_loss_weight: weight on classification loss
    :exponential_loss_weight: weight on exponential loss.
    """
    opt = tf.keras.optimizers.Adam(lr)
    inpt = tf.keras.layers.Input((None, None, 3), name="maskgen_input")

    # compute mask, inverse mask, and masked input.
    # mask is 1 in the removed region, 0 outside.
    # inverse mask is 0 in removed region, 1 outside.
    mask = _compose(inpt, mask_generator)
    inverse_mask = tf.keras.layers.Lambda(lambda x: 1.-x)(mask)
    masked_input = tf.keras.layers.Multiply()([inpt, inverse_mask])

    # push through inpainter. we train that in a separate step
    # so set trainable=False for each layer.
    inpainted = _compose(masked_input, inpainter, False)
    
    # combine inpainter outputs with mask, and add to
    # original masked image
    masked_inpainted = tf.keras.layers.Multiply()([inpainted, mask])
    assembled_inpainted = tf.keras.layers.Add()([masked_inpainted, masked_input])

    # now classify as 0 or 1. classifier is pretrained
    # so freeze this as well.
    softmax_out = _compose(assembled_inpainted, classifier, False)
    
    mask_generator_trainer = tf.keras.Model(inpt, [softmax_out, mask])
    mask_generator_trainer.compile(
            opt,
            loss=[tf.keras.losses.sparse_categorical_crossentropy,
                  exponential_loss],
                  loss_weights=[class_loss_weight, 
                                exponential_loss_weight],
                  metrics=["accuracy"]
                  )
    return mask_generator_trainer



def build_inpainter_trainer(inpainter, discriminator, lr=1e-3,
                            reconstruction_loss=100, disc_loss=2):
    """
    Build a compiled Keras model for training the inpainter.
    
    :inpainter: Keras model for inpainting
    :discriminator: Keras model for pixelwise real/fake discrimination
    :lr: learning rate for ADAM optimizer
    :reconstruction_loss: weight for L1 reconstruction loss
    :disc_loss: weight for GAN loss on inpainter
    """
    opt = tf.keras.optimizers.Adam(lr)
    inpt = tf.keras.layers.Input((None, None, 3))
    mask = tf.keras.layers.Input((None, None, 1))

    #
    inverse_mask = tf.keras.layers.Lambda(lambda x: 1.-x)(mask)
    masked_input = tf.keras.layers.Multiply()([inpt, inverse_mask])

    # push through inpainter.
    inpainted = _compose(masked_input, inpainter, True)
    
    # combine inpainter outputs with mask, and add to
    # original masked image
    masked_inpainted = tf.keras.layers.Multiply()([inpainted, mask])
    assembled_inpainted = tf.keras.layers.Add()([masked_inpainted, masked_input])

    # now do pixelwise classification as 0 or 1 (real/fake).
    # discriminator is trained in a separate step
    softmax_out = _compose(assembled_inpainted, discriminator, False)
    
    inpainter_trainer = tf.keras.Model([inpt,mask], [assembled_inpainted, softmax_out])
    inpainter_trainer.compile(
            opt,
            loss=[tf.keras.losses.mae, 
                  least_squares_gan_loss],
            loss_weights = [reconstruction_loss, disc_loss],
            metrics=["mae"]
            )
    return inpainter_trainer




def build_discriminator_trainer(inpainter, discriminator, lr=1e-3):
    """
    
    :inpainter: Keras model for inpainting
    :discriminator: Keras model for pixelwise real/fake discrimination
    :lr: learning rate for ADAM optimizer
    """
    opt = tf.keras.optimizers.Adam(lr)
    inpt = tf.keras.layers.Input((None, None, 3))
    mask = tf.keras.layers.Input((None, None, 1))

    #
    inverse_mask = tf.keras.layers.Lambda(lambda x: 1.-x)(mask)
    masked_input = tf.keras.layers.Multiply()([inpt, inverse_mask])

    # push through inpainter. for this step we hold the inpainter fixed.
    inpainted = _compose(masked_input, inpainter, False)
    
    # combine inpainter outputs with mask, and add to
    # original masked image
    masked_inpainted = tf.keras.layers.Multiply()([inpainted, mask])
    assembled_inpainted = tf.keras.layers.Add()([masked_inpainted, masked_input])

    # now do pixelwise classification as 0 or 1 (real/fake).
    softmax_out = _compose(assembled_inpainted, discriminator, True)
    
    discriminator_trainer = tf.keras.Model([inpt,mask], softmax_out)
    discriminator_trainer.compile(
            opt,
            loss=least_squares_gan_loss,
            metrics=["accuracy"]
            )
    return discriminator_trainer




class Trainer(object):
    """
    Prototype wrapper class for training inpainting network
    """
    
    def __init__(self, mask_generator, classifier, inpainter, discriminator,
                 mask_trainer_dataset, image_dataset, lr=1e-3,
                 steps_per_epoch=100, batch_size=64,
                 class_loss_weight=12, exponential_loss_weight=18,
                 reconstruction_loss=100, disc_loss=2,
                 eval_pos=None, eval_neg=None, logdir=None):
        """
        :mask_generator: keras mask generator model
        :classifier: pretrained convnet for classifying images
        :inpainter: keras inpainting model
        :discriminator: Keras model for pixelwise real/fake discrimination
        :mask_trainer_dataset: tf.data.Dataset object with ___
        :image_dataset: tf.data.Dataset object with ___
        :class_loss_weight: for mask generator, weight on classification loss
        :exponential_loss_weight: for mask generator, weight on exponential loss.
        :reconstruction_loss: weight for L1 reconstruction loss
        :disc_loss: weight for GAN loss on inpainter
        :eval_data:
        :logdir:
        """
        assert tf.executing_eagerly(), "eager execution must be enabled first"
        self.epoch = 0
        self.eval_pos = eval_pos
        self.eval_neg = eval_neg
        if logdir is not None:
            self._summary_writer = tf.contrib.summary.create_file_writer(logdir,
                                            flush_millis=10000)
            self._summary_writer.set_as_default()
            
        self.global_step = tf.train.get_or_create_global_step()
        self._steps_per_epoch = steps_per_epoch
        self._batch_size = batch_size
        self.inp_losses = []
        self.inp_accs = []
        self.disc_losses = []
        self.disc_accs = []
        self.recon_loss = []
        self._mask_generator_trainer = build_mask_generator_trainer(mask_generator, 
                                                                    classifier, 
                                                      inpainter, lr,
                                                      class_loss_weight=class_loss_weight,
                                                      exponential_loss_weight=exponential_loss_weight)
        self._inpainter_trainer = build_inpainter_trainer(inpainter, 
                                                          discriminator, lr,
                                                          reconstruction_loss=reconstruction_loss,
                                                          disc_loss=disc_loss)
        self._discriminator_trainer = build_discriminator_trainer(inpainter, 
                                                    discriminator, lr)
        self._inpainter = inpainter
        self._classifier = classifier
        self._maskgen = mask_generator
        self._discriminator = discriminator
        self._ds_mask = mask_trainer_dataset
        self._ds_img = image_dataset
        
        
    def fit(self, epochs=1):
        """
        
        """
        inpaint = True

        # for each epoch
        for e in tqdm(range(epochs)):
            # one training epoch on mask generator
            self._mask_generator_trainer.fit(self._ds_mask, 
                               #validation_data=(imgs, (labs, ms)),
                               steps_per_epoch=self._steps_per_epoch,
                               epochs=1,
                               batch_size=self._batch_size,
                               initial_epoch=self.epoch,
                               verbose=0)
            # one training epoch on the inpainter and discriminator:
    
            # generate some mask samples for the mask buffer
            mask_buffer = self._maskgen.predict(self._ds_img, 
                                          steps=self._steps_per_epoch, 
                                          batch_size=self._batch_size,
                                          verbose=0)    
            # now one training epoch on inpainter/discriminator
            step = 0
            for im in self._ds_img:
                im = im.numpy()
                sample_indices = np.random.choice(np.arange(im.shape[0]), 
                                          replace=False,
                                          size=im.shape[0])
                mask_sample = mask_buffer[sample_indices,:,:,:]
                mask_target = np.zeros(mask_sample.shape, dtype=np.int64)
                x = (im, mask_sample)
                y = (im, mask_target)
        
                # one batch on inpainter
                if inpaint:
                    inp_loss, recon_loss, *_ =  self._inpainter_trainer.train_on_batch(x,y)
                    self.inp_losses.append(inp_loss)
                    self.recon_loss.append(recon_loss)
                    inpaint = False
                # one batch on discriminator    
                else:
                    #disc_loss, disc_acc = self._discriminator_trainer.train_on_batch(x,mask_target)
                    disc_loss, disc_acc = self._discriminator_trainer.train_on_batch(x,mask_sample)
                    self.disc_losses.append(disc_loss)
                    self.disc_accs.append(disc_acc)
                    inpaint = True
                step += 1
                if step > self._steps_per_epoch:
                    break
            self.epoch += 1
            if (self.eval_pos is not None) & (self.eval_neg is not None):
                self.evaluate()
    
    def evaluate(self):
        """
        Run a set of evaluation metrics. Requires validation data and a log
        directory to be set.
        """
        x_pos = self.eval_pos
        x_neg = self.eval_neg
        N_pos = x_pos.shape[0]
        # label positive examples "0"
        classvals = np.zeros(N_pos, dtype=np.int64)
        emptymask = np.zeros((N_pos, x_pos.shape[1], x_pos.shape[2],1))
        # evaluate the mask generator on a POSITIVE example
        maskgen_losses = self._mask_generator_trainer.evaluate(x_pos, (classvals, emptymask), verbose=0)
        # predict masks for the NEGATIVE evaluation data
        pred_mask = self._maskgen.predict(x_neg).astype(np.int64)
        # evaluate the inpainter
        inpainter_losses = self._inpainter_trainer.evaluate([x_neg, pred_mask],
                                            [x_neg, np.zeros(pred_mask.shape, dtype=np.int64)], verbose=0)
        
        
        # also visualize predictions for the positive cases
        predicted_masks = self._maskgen.predict(x_pos)
        predicted_inpaints = self._inpainter.predict(x_pos*(1-predicted_masks))
        reconstructed = x_pos*(1-predicted_masks) + \
                        predicted_masks*predicted_inpaints
    
        rgb_masks = np.concatenate([predicted_masks]*3, -1)
        concatenated = np.concatenate([x_pos, rgb_masks, 
                                       predicted_inpaints, reconstructed],
                                axis=2)
        # record everything
        with tf.contrib.summary.always_record_summaries():
            tf.contrib.summary.image("input__mask__inpainted__reconstructed", 
                             concatenated, max_images=5,
                             step=self.global_step)
   
            tf.contrib.summary.scalar("mask_generator_total_loss", maskgen_losses[0])
            tf.contrib.summary.scalar("mask_generator_classifier_loss", maskgen_losses[1])
            tf.contrib.summary.scalar("mask_generator_exponential_loss", maskgen_losses[2])
            tf.contrib.summary.scalar("mask_generator_classifier_accuracy", maskgen_losses[3])
            tf.contrib.summary.scalar("inpainter_total_loss", inpainter_losses[0])
            tf.contrib.summary.scalar("inpainter_reconstruction_L1_loss", inpainter_losses[1])
            tf.contrib.summary.scalar("inpainter_discriminator_GAN_loss", inpainter_losses[2])

        self.global_step.assign_add(1)


