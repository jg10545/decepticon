# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import yaml
import os
from PIL import Image

from decepticon._trainers import Trainer


def load_trainer_from_saved(posfiles, negfiles, old_dir, new_dir, 
                           **kwargs):
    """
    Build a Trainer object using the log directory from a previous
    Trainer. Useful for transfer learning experiments.
    
    :posfiles: list of paths to positive files
    :negfiles: list of paths to negative files
    :old_dir: directory to load from
    :new_dir: directory to save to
    :kwargs: pass any keyword arguments to override values
        in the original directory
    """
    # load config.yml find
    config = yaml.load(open(os.path.join(old_dir, "config.yml")),
                  Loader=yaml.FullLoader)
    # load saved models
    saved_models = {
        "mask_generator":"mask_generator.h5",
        "inpainter":"inpainter.h5",
        "discriminator":"discriminator.h5",
        "classifier":"classifier.h5",
        "maskdisc":"mask_discriminator.h5"
    }
    files_in_dir = list(os.listdir(old_dir))
    for m in saved_models:
        if saved_models[m] in files_in_dir:
            config[m] = tf.keras.models.load_model(
                        os.path.join(old_dir, saved_models[m]))
            
    for k in kwargs:
        config[k] = kwargs[k]
        
    return Trainer(posfiles, negfiles, 
                   logdir=new_dir, **config)


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
    