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


def _generate_img(imshape, label, num_rectangles=25):
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
        
    return img


def build_synthetic_dataset(imshape=(128,128)):
    """
    Create a tf.data.Dataset object that generates random
    image/label pairs. Class 0 has blue and magenta patches,
    while class 1 has yellow as well.
    """
    def _example_generator():
        while True:
            label = np.random.randint(0,2)
            yield _generate_img(imshape, label), label
            
    ds = tf.data.Dataset.from_generator(_example_generator, (tf.float32, tf.int64),
                                   output_shapes=((imshape[0], imshape[1], 3), ()))
    return ds