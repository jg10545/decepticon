# -*- coding: utf-8 -*-
import numpy as np
#import tensorflow as tf
#import yaml
#import os
from PIL import Image

#from decepticon._trainers import Trainer


def _load_to_array(img):
    """
    input a file path, PIL image or numpy array;
    return a numpy array
    """
    if isinstance(img, str):
        img = Image.open(img)
    if isinstance(img, Image.Image):
        img = np.array(img)
    if img.dtype == np.uint8:
        img = img.astype(np.float32)/255
    return img
    

def _remove_objects(img, maskgen, inpainter, return_all=False):
    """
    Run the object removal model on an image, returning results
    as a PIL Image.
    
    :img: PIL Image object or path to image
    :maskgen: Keras model for mask generator
    :inpainter: Keras model for inpainter
    :return_all: if True, return the mask, inpainted image, and
        reconstructed image
    """
    if isinstance(img, str):
        img = Image.open(img)
            
    # now some funky stuff- 
    W, H = img.size
    Wnew, Hnew = img.size

    if Wnew%8 != 0: Wnew = 8*(Wnew//8 + 1)
    if Hnew%8 != 0: Hnew = 8*(Hnew//8 + 1)
    img_arr = np.expand_dims(
        np.array(img.crop([0, 0, Wnew, Hnew])), 0).astype(np.float32)/255
        
    mask = maskgen.predict(img_arr)
    masked_img = np.concatenate([
                img_arr*(1-mask),
                mask], -1)
    inpainted = inpainter.predict(masked_img)
    reconstructed = img_arr*(1-mask) + inpainted*mask
        
    recon_img =  Image.fromarray((255*reconstructed[0]).astype(np.uint8))
    recon_img = recon_img.crop([0, 0, W, H])
    
    if return_all:
        return img, mask[0], inpainted[0], recon_img
    else:
        return recon_img
    