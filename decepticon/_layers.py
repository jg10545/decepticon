"""
                _layers.py
                
Custom Keras layers go here.
"""
import tensorflow as tf
import tensorflow.keras.backend as K

EPSILON = 1e-8



class InstanceNormalizationLayer(tf.keras.layers.Layer):
    """
    Instance Normalization, as described in
    
    "Instance Normalization: The Missing Ingredient for 
    Fast Stylization" by Ulyanov, Vedaldi, and Lempitsky
    
    This version assumes you're using the TensorFlow default
    of CHANNELS LAST (NHWC)
    """
    def __init__(self, **kwargs):
        super(InstanceNormalizationLayer, self).__init__(**kwargs)
        
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def build(self, input_shape):
        super(InstanceNormalizationLayer, self).build(input_shape)
        
    def call(self, inputs):
        
        mu_raw = K.mean(inputs, axis=[1,2])
        mu = tf.expand_dims(tf.expand_dims(mu_raw, axis=1), axis=1)
        
        sig_sq_raw = K.var(inputs, axis=[1,2])
        sig_sq = tf.expand_dims(tf.expand_dims(sig_sq_raw, axis=1), 
                                axis=1)
        
        return (inputs - mu)/K.sqrt(sig_sq + EPSILON)