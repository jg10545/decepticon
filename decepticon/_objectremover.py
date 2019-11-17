# -*- coding: utf-8 -*-
import os
import matplotlib.pyplot as plt

from decepticon._util import _remove_objects
from decepticon._models import build_mask_generator, build_inpainter

class ObjectRemover(object):
    """
    Class to automate inference from a saved model.
    
    """


    def __init__(self, logdir=None, maskgen=None, inpainter=None, downsample=2):
        """
        Pass a log directory from a previous decepticon run, paths to 
        inpainter and mask generator models, or Keras models directly.
        
        If you pass a string- a new model will be generated and then load_weights()
        will be called, instead of calling tf.keras.models.load_model() directly.
        This is a workaround for issues with saving nested models. 
        
        :logdir: string; path to log directory
        :maskgen: string; path to mask generator save, or keras model. Supercedes
            the mask generator found in logdir.
        :inpainter: string; path to inpainter save, or keras model. Supercedes
            the inpainter found in logdir.
        :downsample: downsampling kwarg for initializing new models
        """
        if (logdir is not None)&(maskgen is None):
            maskgen = os.path.join(logdir, "mask_generator.h5")
        if (logdir is not None)&(inpainter is None):
            inpainter = os.path.join(logdir, "inpainter.h5")
            
        if isinstance(maskgen, str):
            #maskgen = tf.keras.models.load_model(maskgen)
            maskfile = maskgen
            maskgen = build_mask_generator(downsample=downsample)
            maskgen.load_weights(maskfile)
            
        if isinstance(inpainter, str):
            #inpainter = tf.keras.models.load_model(inpainter)
            paintfile = inpainter
            inpainter = build_inpainter(downsample=downsample)
            inpainter.load_weights(paintfile)
        
        self.maskgen = maskgen
        self.inpainter = inpainter
        
    def __call__(self, img, return_all=False):
        """
        Input a path to an image or a PIL Image object, and return
        an Image object with the object removal model applied
        """
        return _remove_objects(img, self.maskgen, self.inpainter, 
                               return_all)

    def plot(self, img):
        """
        Use matplotlib to draw a plot showing the original image,
        mask, inpainted image, and reconstructed image.
        """
        img, mask, inpainted, reconstructed = self(img, True)
        
        plt.subplot(2,2,1)
        plt.imshow(img)
        plt.axis("off")
        plt.title("original", fontsize=14)
        
        plt.subplot(2,2,2)
        plt.imshow(mask[:,:,0])
        plt.imshow(img, alpha=0.1)
        plt.axis("off")
        plt.title("mask", fontsize=14)
        
        plt.subplot(2,2,3)
        plt.imshow(inpainted)
        plt.axis("off")
        plt.title("inpainted", fontsize=14)
        
        plt.subplot(2,2,4)
        plt.imshow(reconstructed)
        plt.axis("off")
        plt.title("reconstructed", fontsize=14)
