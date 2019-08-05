"""

            _models.py
            
            
Code for Keras models here

"""

import tensorflow as tf

from decepticon._layers import InstanceNormalizationLayer

def _compose(net, model, trainable=True):
    """
    Hack for keras isue #10074
    """
    for l in model.layers:
        # don't chain together input layers, since that causes
        # some problems later
        if not isinstance(l, tf.keras.layers.InputLayer):
            net = l(net)
            l.trainable = trainable
    return net

    
def ResidualBlock(filters, kernel_size=3):
    """
    Residual block as described in Choi et al's StarGAN paper
    """
    inpt = tf.keras.layers.Input((None, None, filters))
    net = tf.keras.layers.Conv2D(filters, kernel_size,
                                           padding="same")(inpt)
    net =  InstanceNormalizationLayer()(net)
    net = tf.keras.layers.Activation("relu")(net)
    net = tf.keras.layers.Conv2D(filters, kernel_size,
                                           padding="same")(net)
    net =  InstanceNormalizationLayer()(net)
    net = tf.keras.layers.Activation("relu")(net)
    net = tf.keras.layers.Add()([inpt, net])

    return tf.keras.Model(inpt, net)
    
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




def MaskGeneratorHead(input_shape=(None, None, 512), downsample=1, target_classes=1):
    """
    Output layers for the Mask Generator from section 2 of
    Shetty et al's supplementary material
    
    :target_classes: 20 in paper
    :downsample: reduce the number of kernels in each layer
        by this factor
    """
    if target_classes == 1:
        k = 1
    else:
        k = target_classes + 1
        
    inpt = tf.keras.layers.Input(input_shape)
    net = tf.keras.layers.Conv2D(int(512/downsample), kernel_size=3,
                                  padding="same")(inpt)
    net = tf.keras.layers.LeakyReLU(0.1)(net)
    net = ResidualBlock(int(512/downsample))(net)
    net = tf.keras.layers.Conv2D(int(256/downsample), kernel_size=3,
                                  padding="same")(net)
    net = tf.keras.layers.LeakyReLU(0.1)(net)
    net = ResidualBlock(int(256/downsample))(net)
    net = tf.keras.layers.Conv2D(int(128/downsample), kernel_size=3,
                                  padding="same")(net)
    net = tf.keras.layers.LeakyReLU(0.1)(net)
    net = ResidualBlock(int(128/downsample))(net)
    net = tf.keras.layers.Conv2DTranspose(k, kernel_size=7,
                            strides=4,
                            activation=tf.keras.activations.sigmoid,
                            padding="same")(net)
    return tf.keras.Model(inpt, net)




def InpainterDownsampler(input_shape=(None, None, 3), downsample=1):
    """
    note- final layer changed from paper from 512 to 256
    to align with residual layers for bottleneck
    """
    inpt = tf.keras.layers.Input(input_shape)
    net = tf.keras.layers.Conv2D(int(64/downsample), kernel_size=4,
                                  padding="same")(inpt)
    net = InstanceNormalizationLayer()(net)
    net = tf.keras.layers.LeakyReLU(0.1)(net)
    net = tf.keras.layers.Conv2D(int(128/downsample), kernel_size=4,
                                  strides=2, padding="same")(net)
    net = InstanceNormalizationLayer()(net)
    net = tf.keras.layers.LeakyReLU(0.1)(net)
    net = tf.keras.layers.Conv2D(int(256/downsample), kernel_size=4,
                                  strides=2, padding="same")(net)
    net = InstanceNormalizationLayer()(net)
    net = tf.keras.layers.LeakyReLU(0.1)(net)
    net = tf.keras.layers.Conv2D(int(256/downsample), kernel_size=4,
                                  strides=2, padding="same")(net)
    net = InstanceNormalizationLayer()(net)
    net = tf.keras.layers.LeakyReLU(0.1)(net)
    
    return tf.keras.Model(inpt, net)



def InpainterBottleneck(input_shape=(None, None, 256), downsample=1, num_residuals=6):
    """
    
    """ 
    inpt = tf.keras.layers.Input(input_shape)
    net = inpt
    for _ in range(num_residuals):
        net = ResidualBlock(int(256/downsample))(net)
    
    return tf.keras.Model(inpt, net)
        
    
  
def InpainterUpsampler(input_shape=(None, None, 256), downsample=1):
    """
    
    """
    inpt = tf.keras.layers.Input(input_shape)
    net = tf.keras.layers.UpSampling2D(size=2, 
                                         interpolation="bilinear")(inpt)
    net = tf.keras.layers.Conv2D(int(256/downsample), kernel_size=3,
                                  padding="same")(net)
    net = InstanceNormalizationLayer()(net)
    net = tf.keras.layers.LeakyReLU(0.1)(net)
    net = tf.keras.layers.UpSampling2D(size=2, 
                                         interpolation="bilinear")(net)
    net = tf.keras.layers.Conv2D(int(128/downsample), kernel_size=3,
                                  padding="same")(net)
    net = InstanceNormalizationLayer()(net)
    net = tf.keras.layers.LeakyReLU(0.1)(net)
    net = tf.keras.layers.UpSampling2D(size=2, 
                                         interpolation="bilinear")(net)
    net = tf.keras.layers.Conv2D(int(64/downsample), kernel_size=3,
                                  padding="same")(net)
    net = InstanceNormalizationLayer()(net)
    net = tf.keras.layers.LeakyReLU(0.1)(net)
    net = tf.keras.layers.Conv2D(3, kernel_size=7,
                            strides=1,
                            activation=tf.keras.activations.sigmoid,
                            padding="same")(net)
    return tf.keras.Model(inpt, net)


def build_inpainter(input_shape=(None, None, 3), downsample=1):
    """
    Build the full inpainting network
    
    :downsample: reduce number of kernels by this factor
    """
    downsampler = InpainterDownsampler(input_shape, downsample=downsample)
    bottleneck = InpainterBottleneck(downsampler.output_shape[1:], downsample=downsample)
    upsampler = InpainterUpsampler(bottleneck.output_shape[1:], downsample=downsample)
    
    for model in [downsampler, bottleneck, upsampler]:
        for l in model.layers:
            l._name = "inpainter_" + l.name
    
    inpt = tf.keras.layers.Input(input_shape)
    #net = downsampler(inpt)
    #net = bottleneck(net)
    #net = upsampler(net)
    net = _compose(inpt, downsampler)
    net = _compose(net, bottleneck)
    net = _compose(net, upsampler)
    #return tf.keras.Model(inpt, net) 
    model = tf.keras.Model(inpt, net)
    #for l in model.layers:
    #    l._name = "inpainter_" + l.name
    return model


def ClassificationHead(input_shape=(None, None, 512), target_classes=1, downsample=1):
    """
    Output layers for the object classifier from section 2 of
    Shetty et al's supplementary material
    
    :target_classes: 20 in paper
    :downsample: reduce the number of kernels in each layer
        by this factor
    """
    inpt = tf.keras.layers.Input(input_shape)
    net = tf.keras.layers.Conv2D(int(512/downsample), kernel_size=3,
                                   padding="same")(inpt)
    net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.LeakyReLU(0.1)(net)
    net = tf.keras.layers.Conv2D(int(512/downsample), kernel_size=3,
                                   padding="same")(net)
    net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.LeakyReLU(0.1)(net)
    net = tf.keras.layers.GlobalMaxPool2D()(net)
    net = tf.keras.layers.Dense(target_classes+1,
                                 activation=tf.keras.activations.softmax)(net)
    return tf.keras.Model(inpt, net)


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

    head = ClassificationHead(fcn.output_shape[1:], target_classes, downsample)
    
    for l in fcn.layers:
        l._name = "classifier_" + l.name
    for l in head.layers:
        l._name = "classifier_" + l.name
    
    inpt = tf.keras.layers.Input((None, None, 3))
    net = _compose(inpt, fcn)
    net = _compose(net, head)
   
    #return tf.keras.Model(inpt, net)
    model = tf.keras.Model(inpt, net)
    #for l in model.layers:
    #    l._name = "classifier_" + l.name
    return model



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
        
    head = MaskGeneratorHead(fcn.output_shape[1:], downsample, target_classes)
    
    for l in fcn.layers:
        l._name = "maskgen_" + l.name
        
    for l in head.layers:
        l._name = "maskgen_" + l.name
    
    inpt = tf.keras.layers.Input((None, None, 3))
    net = _compose(inpt, fcn)
    net = _compose(net, head)
    
    #return tf.keras.Model(inpt, net)
    model = tf.keras.Model(inpt, net)
    #for l in model.layers:
    #    l._name = "maskgen_" + l.name
    return model



    

def LocalDiscriminator(input_shape=(None, None, 3), downsample=1):
    """
    I'm not sure whether this is exactly the modified patchGAN
    used in Shetty et al- the supplementary material doesn't specify
    and the code has several versions.
    
    :downsample: reduce the number of kernels in each layer
        by this factor
    """
    inpt = tf.keras.layers.Input(input_shape)
    # 128 -> 64
    net = tf.keras.layers.Conv2D(int(64/downsample), kernel_size=4,
                                  strides=2, padding="same")(inpt)
    net = tf.keras.layers.LeakyReLU(0.2)(net)
    # 64 -> 32
    net = tf.keras.layers.Conv2D(int(128/downsample), kernel_size=4,
                                  strides=2, padding="same")(net)
    net = tf.keras.layers.LeakyReLU(0.2)(net)
    # 32 -> 16
    net = tf.keras.layers.Conv2D(int(256/downsample), kernel_size=4,
                                  strides=2, padding="same")(net)
    net = tf.keras.layers.LeakyReLU(0.2)(net)
    # 16 -> 32
    net = tf.keras.layers.Conv2DTranspose(int(256/downsample),
                                           kernel_size=4,
                                           strides=2,
                                           padding="same")(net)
    net = tf.keras.layers.LeakyReLU(0.2)(net)
    # 32 -> 64
    net = tf.keras.layers.Conv2DTranspose(int(128/downsample),
                                           kernel_size=4,
                                           strides=2,
                                           padding="same")(net)
    net = tf.keras.layers.LeakyReLU(0.2)(net)
    # 64 -> 128
    net = tf.keras.layers.Conv2DTranspose(int(64/downsample),
                                           kernel_size=4,
                                           strides=2,
                                           padding="same")(net)
    net = tf.keras.layers.LeakyReLU(0.2)(net)
    net = tf.keras.layers.Conv2D(1, kernel_size=7,
                                  padding="same",
                                  activation=tf.keras.activations.sigmoid)(net)
    return tf.keras.Model(inpt, net)

    
def build_discriminator(input_shape=(None, None, 3), downsample=1):
    """
    Build the real/fake discriminator
    
    """
    
    inpt = tf.keras.layers.Input(input_shape)
    disc = LocalDiscriminator(input_shape, downsample)
    for l in disc.layers:
        l._name = "discriminator_" + l.name
    net = _compose(inpt, disc)
    
    #return tf.keras.Model(inpt, net)
    model = tf.keras.Model(inpt, net)
    #for l in model.layers:
    #    l._name = "discriminator_" + l.name
    return model
