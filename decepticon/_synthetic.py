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

