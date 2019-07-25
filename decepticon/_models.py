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
    
    
def modified_vgg19(weights="imagenet"):
    """
    Rebuild VGG19 without the last two max pooling layers (mask
    generator architecture in section 2 of Shetty et al's supplementary
    material)
    """
    fcn_orig = tf.keras.applications.VGG19(weights=weights, 
                                           include_top=False)
    
    inpt = tf.keras.layers.Input((None, None, 3))
    net = inpt
    for l in fcn_orig.layers:
        if l.name not in ["block3_pool", "block4_pool", "block5_pool"]:
            net = l(net)
    return tf.keras.Model(inpt, net)



class MaskGeneratorHead(tf.keras.Model):
    """
    Output layers for the Mask Generator from section 2 of
    Shetty et al's supplementary material
    
    :target_classes: 20 in paper
    :downsample: reduce the number of kernels in each layer
        by this factor
    """
    
    def __init__(self, target_classes=1, downsample=1):
        super(MaskGeneratorHead, self).__init__()
        
        if target_classes == 1:
            k = 1
        else:
            k = target_classes + 1
        
        self._layers = [
            tf.keras.layers.Conv2D(int(512/downsample), kernel_size=3,
                                  padding="same"),
            tf.keras.layers.LeakyReLU(0.1),
            ResidualBlock(int(512/downsample)),
            tf.keras.layers.Conv2D(int(256/downsample), kernel_size=3,
                                  padding="same"),
            tf.keras.layers.LeakyReLU(0.1),
            ResidualBlock(int(256/downsample)),
            tf.keras.layers.Conv2D(int(128/downsample), kernel_size=3,
                                  padding="same"),
            tf.keras.layers.LeakyReLU(0.1),
            ResidualBlock(int(128/downsample)),
            tf.keras.layers.Conv2DTranspose(k, kernel_size=7,
                            strides=4,
                            activation=tf.keras.activations.sigmoid,
                            padding="same")   
        ]

    def call(self, inputs):
        net = inputs
        for l in self._layers:
            net = l(net)
        return net


class InpainterDownsampler(tf.keras.Model):
    """
    note- final layer changed from paper from 512 to 256
    to align with residual layers for bottleneck
    """
    def __init__(self,  downsample=1):
        super(InpainterDownsampler, self).__init__()

        
        self._layers = [
            tf.keras.layers.Conv2D(int(64/downsample), kernel_size=4,
                                  padding="same"),
            InstanceNormalizationLayer(),
            tf.keras.layers.LeakyReLU(0.1),
            tf.keras.layers.Conv2D(int(128/downsample), kernel_size=4,
                                  strides=2, padding="same"),
            InstanceNormalizationLayer(),
            tf.keras.layers.LeakyReLU(0.1),
            tf.keras.layers.Conv2D(int(256/downsample), kernel_size=4,
                                  strides=2, padding="same"),
            InstanceNormalizationLayer(),
            tf.keras.layers.LeakyReLU(0.1),
            tf.keras.layers.Conv2D(int(256/downsample), kernel_size=4,
                                  strides=2, padding="same"),
            InstanceNormalizationLayer(),
            tf.keras.layers.LeakyReLU(0.1)
        ]

    def call(self, inputs):
        net = inputs
        for l in self._layers:
            net = l(net)
        return net


class InpainterBottleneck(tf.keras.Model):
    """
    
    """ 
    def __init__(self,  downsample=1, num_residuals=6):
        super(InpainterBottleneck, self).__init__()

        
        self._layers = [
            ResidualBlock(int(256/downsample)) 
                for _ in range(num_residuals)
        ]

    def call(self, inputs):
        net = inputs
        for l in self._layers:
            net = l(net)
        return net

class InpainterUpsampler(tf.keras.Model):
    """
    
    """
    
    def __init__(self,  downsample=1):
        super(InpainterUpsampler, self).__init__()

        
        self._layers = [
            tf.keras.layers.UpSampling2D(size=2, 
                                         interpolation="bilinear"),
            tf.keras.layers.Conv2D(int(256/downsample), kernel_size=3,
                                  padding="same"),
            InstanceNormalizationLayer(),
            tf.keras.layers.LeakyReLU(0.1),
            
            tf.keras.layers.UpSampling2D(size=2, 
                                         interpolation="bilinear"),
            tf.keras.layers.Conv2D(int(128/downsample), kernel_size=3,
                                  padding="same"),
            InstanceNormalizationLayer(),
            tf.keras.layers.LeakyReLU(0.1),
            
            tf.keras.layers.UpSampling2D(size=2, 
                                         interpolation="bilinear"),
            tf.keras.layers.Conv2D(int(64/downsample), kernel_size=3,
                                  padding="same"),
            InstanceNormalizationLayer(),
            tf.keras.layers.LeakyReLU(0.1),
            
            tf.keras.layers.Conv2D(3, kernel_size=7,
                            strides=1,
                            activation=tf.keras.activations.sigmoid,
                            padding="same")
        ]
        
    def call(self, inputs):
        net = inputs
        for l in self._layers:
            net = l(net)
        return net

def build_inpainter(downsample=1):
    """
    Build the full inpainting network
    
    :downsample: reduce number of kernels by this factor
    """
    downsampler = InpainterDownsampler(downsample=downsample)
    bottleneck = InpainterBottleneck(downsample=downsample)
    upsampler = InpainterUpsampler(downsample=downsample)
    
    inpt = tf.keras.layers.Input((None, None, 3))
    net = downsampler(inpt)
    net = bottleneck(net)
    net = upsampler(net)
    
    return tf.keras.Model(inpt, net)  



class ClassificationHead(tf.keras.Model):
    """
    Output layers for the object classifier from section 2 of
    Shetty et al's supplementary material
    
    :target_classes: 20 in paper
    :downsample: reduce the number of kernels in each layer
        by this factor
    """
    
    def __init__(self, target_classes=1, downsample=1):
        super(ClassificationHead, self).__init__()
        # note- same padding is important here; otherwise for small
        # images the FCN downsampling can lead to negative dimensions
        # in this network
        self._layers = [
            
            tf.keras.layers.Conv2D(int(512/downsample), kernel_size=3,
                                   padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(0.1),
            
            tf.keras.layers.Conv2D(int(512/downsample), kernel_size=3,
                                   padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(0.1),
            
            tf.keras.layers.GlobalMaxPool2D(),
            tf.keras.layers.Dense(target_classes+1,
                                 activation=tf.keras.activations.softmax)
        ]
        
    def call(self, inputs):
        net = inputs
        for l in self._layers:
            net = l(net)
        return net


def build_classifier(fcn=None, target_classes=1, downsample=1):
    """
    Build the object classifier
    
    :fcn: fully-convolutional network. If not specified, uses 
        VGG19 pretrained on ImageNet
    :target_classes: number of object classes
    :downsample: factor to downsample kernels by
    :fcn_trainable:
    """
    if fcn is None:
        fcn = tf.keras.applications.VGG19(weights="imagenet", 
                                          include_top=False)

    head = ClassificationHead(target_classes, downsample)
    
    inpt = tf.keras.layers.Input((None, None, 3))
    net = fcn(inpt)
    #net = tf.keras.layers.GlobalMaxPool2D()(net)
    #net = tf.keras.layers.Dense(target_classes+1, 
    #                            activation=tf.keras.activations.softmax)(net)
    net = head(net)
    
    return tf.keras.Model(inpt, net)


def build_mask_generator(fcn=None, target_classes=1, downsample=1):
    """
    Build the mask generator
    
    :fcn: fully-convolutional network. If not specified, uses 
        modified VGG19 pretrained on ImageNet
    :target_classes: number of object classes
    :downsample: factor to downsample kernels by
    """
    if fcn is None:
        fcn = modified_vgg19()
        
    head = MaskGeneratorHead(target_classes, downsample)
    
    inpt = tf.keras.layers.Input((None, None, 3))
    net = fcn(inpt)
    net = head(net)
    
    return tf.keras.Model(inpt, net)



class LocalDiscriminator(tf.keras.Model):
    """
    I'm not sure whether this is exactly the modified patchGAN
    used in Shetty et al- the supplementary material doesn't specify
    and the code has several versions.
    
    :downsample: reduce the number of kernels in each layer
        by this factor
    """
    
    def __init__(self, downsample=1):
        super(LocalDiscriminator, self).__init__()
        
        
        self._layers = [
            # 128 -> 64
            tf.keras.layers.Conv2D(int(64/downsample), kernel_size=4,
                                  strides=2, padding="same"),
            tf.keras.layers.LeakyReLU(0.2),
            # 64 -> 32
            tf.keras.layers.Conv2D(int(128/downsample), kernel_size=4,
                                  strides=2, padding="same"),
            tf.keras.layers.LeakyReLU(0.2),
            # 32 -> 16
            tf.keras.layers.Conv2D(int(256/downsample), kernel_size=4,
                                  strides=2, padding="same"),
            tf.keras.layers.LeakyReLU(0.2),
            
            # 16 -> 32
            tf.keras.layers.Conv2DTranspose(int(256/downsample),
                                           kernel_size=4,
                                           strides=2,
                                           padding="same"),
            tf.keras.layers.LeakyReLU(0.2),
            # 32 -> 64
            tf.keras.layers.Conv2DTranspose(int(128/downsample),
                                           kernel_size=4,
                                           strides=2,
                                           padding="same"),
            tf.keras.layers.LeakyReLU(0.2),
            # 64 -> 128
            tf.keras.layers.Conv2DTranspose(int(64/downsample),
                                           kernel_size=4,
                                           strides=2,
                                           padding="same"),
            tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.Conv2D(1, kernel_size=7,
                                  padding="same",
                                  activation=tf.keras.activations.sigmoid)
        ]

    def call(self, inputs):
        net = inputs
        for l in self._layers:
            net = l(net)
        return net