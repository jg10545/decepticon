# -*- coding: utf-8 -*-
import tensorflow as tf
from decepticon._models import ResidualBlock
from decepticon._models import modified_vgg19
from decepticon._models import MaskGeneratorHead
from decepticon._models import InpainterDownsampler
from decepticon._models import InpainterBottleneck
from decepticon._models import InpainterUpsampler
from decepticon._models import build_inpainter
from decepticon._models import build_classifier
from decepticon._models import build_mask_generator
from decepticon._models import LocalDiscriminator
from decepticon._models import build_discriminator


def test_ResidualBlock_output_shape():
    inpt = tf.keras.layers.Input((None, None, 5))
    mod = ResidualBlock(5)
    output = mod(inpt)
    
    assert inpt.shape.as_list() == output.shape.as_list()
    
    
def test_modified_vgg_output_shape():
    fcn = modified_vgg19()
    assert fcn.output.get_shape().as_list() == [None, None, None, 512]
    
    
def test_MaskGeneratorHead_output_shape():
    maskhead = MaskGeneratorHead()

    inpt = tf.keras.layers.Input((32,32,512))
    net = maskhead(inpt)
    assert net.get_shape().as_list() == [None, 128, 128, 1]

def test_MaskGeneratorHead_multiclass_output_shape():
    maskhead = MaskGeneratorHead(target_classes=5)

    inpt = tf.keras.layers.Input((32,32,512))
    net = maskhead(inpt)
    assert net.get_shape().as_list() == [None, 128, 128, 6]
    
    
def test_InpainterDownsampler_output_shape():
    downsampler = InpainterDownsampler()

    inpt = tf.keras.layers.Input((128,128,3))
    output = downsampler(inpt)
    assert output.get_shape().as_list() == [None, 16, 16, 256]
    
def test_InpainterBottleneck_output_shape():
    bottleneck = InpainterBottleneck()

    inpt = tf.keras.layers.Input((16,16,256))
    output = bottleneck(inpt)
    assert output.get_shape().as_list() == [None, 16, 16, 256]
    
def test_InpainterUpsampler_output_shape():
    upsample = InpainterUpsampler()

    inpt = tf.keras.layers.Input((16,16,256))
    output = upsample(inpt)
    assert output.get_shape().as_list() == [None, 128, 128, 3]
    
def test_build_inpainter_output_shape():
    inpainter = build_inpainter()
    assert inpainter.input.get_shape().as_list() == inpainter.output.get_shape().as_list()
    
   
def test_build_classifier_input_output_shapes():
    classifier = build_classifier()
    assert classifier.input.get_shape().as_list() == [None, None, None, 3]
    assert classifier.output.get_shape().as_list() == [None, 2]


def test_build_mask_generator_input_output_shapes():
    inpt = tf.keras.layers.Input((128, 128, 3))
    maskgen = build_mask_generator()
    output = maskgen(inpt)

    outshape = output.get_shape().as_list()
    inshape = inpt.get_shape().as_list()
    assert outshape[:3] == inshape[:3]
    assert inshape[-1] == 3
    assert outshape[-1] == 1



def test_local_discriminator_input_output_shapes():
    inpt = tf.keras.layers.Input((128, 128, 3))
    disc = LocalDiscriminator()
    output = disc(inpt)

    outshape = output.get_shape().as_list()
    inshape = inpt.get_shape().as_list()
    assert outshape[:3] == inshape[:3]
    assert inshape[-1] == 3
    assert outshape[-1] == 1
    
    
def test_build_discriminator():
    inpt = tf.keras.layers.Input((128,128,3))
    disc = build_discriminator()
    output = disc(inpt)
    
    outshape =  output.get_shape().as_list()
    inshape = inpt.get_shape().as_list()
    assert outshape[:-1] == inshape[:-1]
    assert outshape[-1] == 1









