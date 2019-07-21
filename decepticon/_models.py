"""

            _models.py
            
            
Code for Keras models here

"""

import tensorflow as tf

from decepticon._layers import InstanceNormalizationLayer



class ResidualBlock(tf.keras.Model):
    """
    Residual block as described in Choi et al's StarGAN paper
    """
    
    def __init__(self, filters, kernel_size=3):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = tf.keras.layers.Conv2D(filters, kernel_size,
                                           padding="same")
        self.instance_norm = InstanceNormalizationLayer()
        self.relu = tf.keras.layers.Activation("relu")
        self.conv2 = tf.keras.layers.Conv2D(filters, kernel_size,
                                           padding="same")
        self.add = tf.keras.layers.Add()

    def call(self, inputs):
        net = self.conv1(inputs)
        net = self.instance_norm(net)
        net = self.relu(net)
        net = self.conv2(net)
        net = self.instance_norm(net)
        net = self.add([net, inputs])
        return net