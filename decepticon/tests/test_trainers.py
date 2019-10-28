# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from decepticon._trainers import maskgen_training_step, inpainter_training_step
#from decepticon._trainers import discriminator_training_step

tf.enable_v2_behavior()

"""

        BUILDING TEST COMPONENTS

I'm not sure what the best way is to test the training
infrastructure- let's define some shared fake model pieces
so that we can make sure the training functions run without
error and return the right things.

"""
# optimizer
opt = tf.keras.optimizers.Adam(1e-4)
# basic input image
input_img = np.random.uniform(0, 1, size=(5,7,11,3)).astype(np.float32)
mask = np.random.uniform(0, 1, size=(5,7,11,1)).astype(np.float32)

# MASK GENERATOR
inpt = tf.keras.layers.Input((None, None, 3))
output = tf.keras.layers.Conv2D(1, 1, padding="same", activation="sigmoid")(inpt)
maskgen = tf.keras.Model(inpt, output)

# CLASSIFIER
net = tf.keras.layers.GlobalAvgPool2D()(inpt)
output = tf.keras.layers.Dense(2, activation="softmax")(net)
classifier = tf.keras.Model(inpt, output)

# INPAINTER
inp_inpt = tf.keras.layers.Input((None, None, 4))
output = tf.keras.layers.Conv2D(3, 1, activation="sigmoid")(inp_inpt)
inpainter = tf.keras.Model(inp_inpt, output)

# DISCRIMINATOR
output = tf.keras.layers.Conv2D(1, 1, activation="sigmoid")(inpt)
discriminator = tf.keras.Model(inpt, output)


def test_maskgen_training_step():
    cls_loss, exp_loss, prior_loss, tv_loss, loss, mask = maskgen_training_step(
                                            opt, input_img,
                                            maskgen, classifier, inpainter)
    
    maskshape = mask.get_shape().as_list()
    assert isinstance(mask, tf.Tensor)
    assert maskshape == [5,7,11,1]
    assert cls_loss.numpy().dtype == np.float32
    assert exp_loss.numpy().dtype == np.float32
    assert loss.numpy().dtype == np.float32
    
    
def test_inpainter_training_step():
    recon_loss, disc_loss, style_loss, tv_loss, loss, d_loss = inpainter_training_step(
                                            opt, opt, input_img, mask,
                                            inpainter, discriminator)
    assert recon_loss.numpy().dtype == np.float32
    assert disc_loss.numpy().dtype == np.float32
    assert loss.numpy().dtype == np.float32
    
    
"""
obsolete, as discriminator_training_step() has been rolled into
inpainter_training_step()

def test_discriminator_training_step():
    loss = discriminator_training_step(opt, input_img, mask,
                                       inpainter, discriminator)
    assert loss.numpy().dtype == np.float32
""" 
