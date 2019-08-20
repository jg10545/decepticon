"""

            _synthetic.py
            
code for generating a simple synthetic test dataset

"""

import numpy as np
import tensorflow as tf



def _rand_rectangle(H,W):
    """
    pick a random rectangle in a (H,W) image
    """
    dh = np.random.randint(int(H/16)+1, int(H/4))
    dw = np.random.randint(int(W/16)+1, int(W/4))
    top = np.random.randint(0, H-dh-1)
    left = np.random.randint(0, W-dw-1)
    return top, left, top+dh, left+dw


def _generate_img(imshape, label, num_rectangles=25, num_empty=0):
    """
    generate an image that's random noise with some random 
    rectangular patches
    """
    img = np.random.uniform(0, 1, size=(imshape[0], imshape[1], 3))
    
    if label == 0:
        m = 2
    else:
        m = 3
    for n in range(num_rectangles):
        top, left, bottom, right = _rand_rectangle(imshape[0], imshape[1])
        img[top:bottom, left:right, np.random.randint(0,m)] = 0
        
    for n in range(num_empty):
        top, left, bottom, right = _rand_rectangle(imshape[0], imshape[1])
        img[top:bottom, left:right, :] = 0
        
    return img





def build_synthetic_dataset(prob=0.5, imshape=(128,128), num_rectangles=25, 
                            num_empty=0, return_labels=True):
    """
    Create a tf.data.Dataset object that generates random images.
    Class 0 has blue and magenta patches, while class 1 has yellow as well.
    
    :prob: probability of generating a positive case. 
    :imshape: image dimensions
    :num_rectangles: how many rectangles to draw per image
    :num_empty: number of empty rectangles to draw (for pretraining object
                classifier)
    :return_labels: whether dataset generates (image, label) tuples or just images
    """
    def _example_generator():
        while True:
            y = np.random.choice([0, 1], p=[1-prob, prob])
            img = _generate_img(imshape, y, num_rectangles, num_empty)
            if return_labels:
                yield img, y
            else:
                yield img
    if return_labels:
        ds = tf.data.Dataset.from_generator(_example_generator, (tf.float32, tf.int64),
                                   output_shapes=((imshape[0], imshape[1], 3), 
                                                  ()))
    else:
        ds = tf.data.Dataset.from_generator(_example_generator, tf.float32,
                                   output_shapes=(imshape[0], imshape[1], 3))
    return ds


def _deprecated_build_synthetic_dataset(imshape=(128,128), num_rectangles=25, num_empty=0,
                            mask_train=False, imgs_only=False):
    """
    Create a tf.data.Dataset object that generates random
    image/label pairs. Class 0 has blue and magenta patches,
    while class 1 has yellow as well.
    
    :num_rectangles: how many rectangles to draw per image
    :num_empty: number of empty rectangles to draw (for pretraining object
                classifier)
    :mask_train: for training the mask generator- always use positive
            samples but label them incorrectly (should they all be positive?).
            also return a blank mask for exponential loss
    :imgs_only: only return images
    """
    def _example_generator():
        while True:
            if mask_train:
                y = (0, np.zeros((imshape[0], imshape[1],1), dtype=np.int64))
                yield _generate_img(imshape, 1, num_rectangles, num_empty), y
            elif imgs_only:
                #label = np.random.randint(0,2)
                label = 0
                yield _generate_img(imshape, label, num_rectangles, num_empty)
            else:
                label = np.random.randint(0,2)
                yield _generate_img(imshape, label, num_rectangles, num_empty), label
    if mask_train:
        ds = tf.data.Dataset.from_generator(_example_generator, (tf.float32, (tf.int64, tf.int64)),
                                   output_shapes=((imshape[0], imshape[1], 3), 
                                                  ((), (imshape[0], imshape[1], 1))))
    elif imgs_only:
        ds = tf.data.Dataset.from_generator(_example_generator, tf.float32,
                                   output_shapes=(imshape[0], imshape[1], 3))
    else:
        ds = tf.data.Dataset.from_generator(_example_generator, (tf.float32, tf.int64),
                                   output_shapes=((imshape[0], imshape[1], 3), ()))
    return ds