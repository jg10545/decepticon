# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from PIL import Image

tf.enable_eager_execution()

from decepticon._util import _load_to_array, _remove_objects


"""

        BUILDING TEST COMPONENTS

define some shared fake model pieces 
"""
# MASK GENERATOR
inpt = tf.keras.layers.Input((None, None, 3))
output = tf.keras.layers.Conv2D(1, 1, padding="same", activation="sigmoid")(inpt)
maskgen = tf.keras.Model(inpt, output)
# INPAINTER
inp_inpt = tf.keras.layers.Input((None, None, 4))
output = tf.keras.layers.Conv2D(3, 1, activation="sigmoid")(inp_inpt)
inpainter = tf.keras.Model(inp_inpt, output)




def test_load_to_array():
    test_arr = np.ones((10, 10), dtype=np.uint8)
    test_img = Image.fromarray(test_arr)
    
    for test in [test_arr, test_img]:
        output = _load_to_array(test)
        assert isinstance(output, np.ndarray)
        assert output.dtype == np.float32
        assert (output >= 0).all()
        assert (output <= 1).all()
        
        
        
def test_remove_objects(test_png_path):
    img = Image.open(test_png_path)
    reconstructed = _remove_objects(test_png_path, maskgen, inpainter)
    
    assert isinstance(reconstructed, Image.Image)
    assert img.size == reconstructed.size
        