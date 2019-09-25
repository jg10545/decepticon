import numpy as np
import tensorflow as tf


def random_mask_generator(imsize, intensity=3):
    """
    Generator that yields artificial masks to train the WGAN.
    
    :imsize: 2-tuple of image dimensions
    :intensity: poisson intensity for number of circles to draw
    """
    xx, yy = np.meshgrid(np.arange(imsize[0]), np.arange(imsize[1]))

    while True:
        num_circles = np.random.poisson(3)
        mask = np.zeros(imsize, dtype=np.float32)
        for n in range(num_circles):
            x0 = np.random.randint(0, imsize[1])
            y0 = np.random.randint(0, imsize[0])
            r = np.random.uniform(10, (imsize[0]+imsize[1])/8)
            clip = (xx - x0)**2 +(yy - y0)**2 <= r**2
            mask[clip] = 1
            
        yield mask