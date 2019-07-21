# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from decepticon._layers import InstanceNormalizationLayer



def test_instance_normalization_layer():
    # dimensions of a batch of images
    N = 2
    H = 5
    W = 7
    C = 3

    # build a simple model that runs inputs through the layer
    inpt = tf.keras.layers.Input((None, None, C))
    net = InstanceNormalizationLayer()(inpt)
    model = tf.keras.Model(inpt, net)
    
    # generate fake data and run it through
    test_in = np.random.normal(0, 1, (N,H,W,C))
    for i, av, std in [(0,1,3), (1,-1,5)]:
        test_in[i,:,:,:] *= std
        test_in[i,:,:,:] += av
    
    test_out = model.predict(test_in)
    assert isinstance(test_out, np.ndarray)
    for i in range(2):
        assert round(test_out[i,:,:,:].mean(), 3) == 0
        assert round(test_out[i,:,:,:].var(), 3) == 1