# -*- coding: utf-8 -*-
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

def _circle_mask_generator(imshape, intensity=3):
    """
    Generator that yields circular masks
    
    :imsize: 2-tuple of image dimensions
    :intensity: poisson intensity for number of circles to draw
    """
    xx, yy = np.meshgrid(np.arange(imshape[0]), np.arange(imshape[1]))

    while True:
        num_circles = np.random.poisson(3)
        mask = np.zeros(imshape, dtype=np.float32)
        for n in range(num_circles):
            x0 = np.random.randint(0, imshape[1])
            y0 = np.random.randint(0, imshape[0])
            r = np.random.uniform(10, (imshape[0]+imshape[1])/8)
            clip = (xx - x0)**2 +(yy - y0)**2 <= r**2
            mask[clip] = 1
            
        yield np.expand_dims(mask, 2)
        
        
def _splotch_mask_generator(imshape):
    """
    Generator that yields splotchy masks
    
    :imshape: 2-tuple of image dimensions
    """
    xx, yy = np.meshgrid(np.arange(imshape[0]), np.arange(imshape[1]))
    # length scale
    scale = (imshape[0]+imshape[1])/30
    while True:
        num_circles = np.random.negative_binomial(1, 0.2)
        mask = np.zeros(imshape, dtype=np.float32)
        x0 = np.random.randint(0, imshape[1])
        y0 = np.random.randint(0, imshape[0])
        for n in range(num_circles):
            a = 10 + np.random.exponential(scale)
            b = 10 + np.random.exponential(scale)
            clip = ((xx - x0)/a)**2 + ((yy - y0)/b)**2 <= 1
            mask[clip] = 1
            x0 += np.random.randint(-a,a)
            y0 += np.random.randint(-b,b)
            
        yield np.expand_dims(mask, 2)
        
def circle_mask_dataset(imshape=(80,80),intensity=3,
                        batch_size=32, prefetch=True):
    """
    Build a dataset for generating samples from a mask prior.
    
    :imshape: dimensions of image patches
    :intensity: poisson intensity
    :batch_size: size of batches
    """
    def _gen():
        #return _circle_mask_generator(imshape, intensity)
        return _splotch_mask_generator(imshape)
        
    maskgen_ds = tf.data.Dataset.from_generator(_gen, 
                                            tf.float32,
                                            (imshape[0],imshape[1],1))
    if batch_size > 0:
        maskgen_ds = maskgen_ds.batch(batch_size)
    if prefetch:
        maskgen_ds = maskgen_ds.prefetch(1)
    return maskgen_ds

def _mask_img(img, num_empty):
    """
    Cut num_empty random rectangles out of an image
    """
    img = img.copy()
    for n in range(num_empty):
        top, left, bottom, right = _rand_rectangle(img.shape[0], img.shape[1])
        img[top:bottom, left:right, :] = 0
    return img
    


def _load_img(f):
    raw_img = tf.io.read_file(f)
    decoded = tf.io.decode_image(raw_img)
    return tf.cast(decoded, tf.float32)/255


def _augment(im):
    """
    Quick-and-dirty random augmentation. ripped off from patchwork
    """
    # built-in methods
    im = tf.image.random_brightness(im, 0.2)
    im = tf.image.random_contrast(im, 0.4, 1.4)
    im = tf.image.random_flip_left_right(im)
    #im = tf.image.random_flip_up_down(im)
       
    # some augmentation can put values outside unit interval
    im = tf.minimum(im, 1)
    im = tf.maximum(im, 0)
    return im


def image_loader_dataset(filepaths, batch_size=64, repeat=False, shuffle=True, 
                        augment=True, num_parallel_calls=None, prefetch=True):
    """
    Barebones function for building a tensorflow Dataset to load,
    shuffle, and batch images from disk.
    
    :filepaths: list of paths to files
    :batch_size: size of batches; set to None to skip batching
    :repeat: whether to repeat when iteration reaches the end
    :shuffle: whether to shuffle the order in which files are loaded
    :augment: whether to augment the data
    :num_parallel_calls: number of threads to use for loading/decoding images
    """
    ds = tf.data.Dataset.from_tensor_slices(filepaths)
    
    if shuffle:
        ds = ds.shuffle(len(filepaths))
        
    ds = ds.map(_load_img, num_parallel_calls=num_parallel_calls)
    
    if augment:
        ds = ds.map(_augment, num_parallel_calls=num_parallel_calls)
    
    if repeat:
        ds = ds.repeat()    
    if batch_size:
        ds = ds.batch(batch_size)
    if prefetch:
        ds = ds.prefetch(1)
    
    return ds




def classifier_training_dataset(pos_files, neg_files, imshape=(80,80),
                               num_empty=10, shuffle=1000,
                                batch_size=32, intensity=3,
                                num_parallel_calls=None,
                                prefetch=True):
    """
    Build a dataset for pretraining the classifier using images masked
    with random circles
    
    :pos_files: list of paths to image patches containing the objects to be removed
    :neg_files: list of paths to empty image patches
    :imshape: dimensions of image patches
    :num_empty: how many empty patches to add per image
    :shuffle: size of shuffle queue
    :batch_size: size of batches
    :intensity: Poisson intensity
    :num_parallel_calls: number of threads for map operations
    :prefetch: whether to prefetch the next batch
    """
    # define a dataset that generates 
    circle_ds = circle_mask_dataset(imshape, intensity, 
                                    batch_size=0, prefetch=False)
    
        
    filenames_with_labels = [(x,0) for x in neg_files] + \
                [(x,1) for x in pos_files]
    np.random.shuffle(filenames_with_labels)
    ds = tf.data.Dataset.zip((
            tf.data.Dataset.from_tensor_slices(
                [x[0] for x in filenames_with_labels]),
            tf.data.Dataset.from_tensor_slices(
                [x[1] for x in filenames_with_labels])))
    # load the images
    ds = ds.map(lambda x,y: (_load_img(x), y), num_parallel_calls=num_parallel_calls)
    # augment
    ds = ds.map(lambda x,y: (_augment(x), y), num_parallel_calls=num_parallel_calls)
    # zip together images and masks
    ds = tf.data.Dataset.zip((ds, circle_ds))
    ds = ds.map(lambda x,m: (x[0]*(1-m), x[1]))
    ds = ds.repeat()
    if shuffle:
        ds = ds.shuffle(shuffle)
    if batch_size:
        ds = ds.batch(batch_size)
    if prefetch:
        ds = ds.prefetch(1)
        
    return ds



def inpainter_training_dataset(neg_files, imshape=(80,80),
                               num_empty=10, shuffle=1000,
                                batch_size=32, num_parallel_calls=None):
    """
    Build a dataset for pretraining the inpainter using images with random
    rectangular masks.
    
    :neg_files: list of paths to empty image patches
    :imshape: dimensions of image patches
    :num_empty: how many empty patches to add per image
    :shuffle: size of shuffle queue
    :batch_size: size of batches
    """
    def _random_mask_generator():
        while True:
            img = np.ones((imshape[0], imshape[1],1)).astype(np.float32)
            yield _mask_img(img, num_empty)
        

    ds = tf.data.Dataset.from_tensor_slices(neg_files)
    # load the images
    ds = ds.map(_load_img)
    # augment
    ds = ds.map(_augment)
    # generate random masks
    maskgen_ds = tf.data.Dataset.from_generator(_random_mask_generator, 
                                            tf.float32,
                                            (imshape[0],imshape[1],1))
    # randomly mask the images
    ds = tf.data.Dataset.zip((ds, maskgen_ds))
    ds = ds.map(lambda x,y: (x*y, x), num_parallel_calls=num_parallel_calls)
    ds = ds.repeat()
    if shuffle:
        ds = ds.shuffle(shuffle)
    if batch_size:
        ds = ds.batch(batch_size)
        
    return ds