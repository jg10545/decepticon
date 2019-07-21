# -*- coding: utf-8 -*-
import tensorflow as tf
from decepticon._models import ResidualBlock


def test_residual_block_output_shape():
    inpt = tf.keras.layers.Input((None, None, 5))
    mod = ResidualBlock(5)
    output = mod(inpt)
    
    assert inpt.shape.as_list() == output.shape.as_list()