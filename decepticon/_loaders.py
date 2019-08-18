# -*- coding: utf-8 -*-
import tensorflow as tf



def _load_img(f):
    raw_img = tf.io.read_file(f)
    return tf.io.decode_image(raw_img)





def image_loader_dataset(filepaths, batchsize=64, repeat=True, shuffle=1000, 
                        num_parallel_calls=None):
    """
    Barebones function for building a tensorflow Dataset to load,
    shuffle, and batch images from disk.
    
    :filepaths: list of paths to files
    :batchsize: size of batches; set to None to skip batching
    :repeat: whether to repeat when iteration reaches the end
    :shuffle: size of shuffle queue. set to False to skip shuffling
    :num_parallel_calls: number of threads to use for loading/decoding images
    """
    ds = tf.data.Dataset.from_tensor_slices(filepaths)
    ds = ds.map(_load_img, num_parallel_calls=num_parallel_calls)
    
    if repeat:
        ds = ds.repeat()    
    if shuffle:
        ds = ds.shuffle(shuffle)
    if batchsize:
        ds = ds.batch(batchsize)
    
    return ds.prefetch(1)
