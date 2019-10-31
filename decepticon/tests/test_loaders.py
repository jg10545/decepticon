# -*- coding: utf-8 -*-
import tensorflow as tf
from decepticon.loaders import image_loader_dataset, circle_mask_dataset

tf.enable_eager_execution()


def test_image_loader_dataset_loads_correctly(test_png_path):
    # build dataset
    ds = image_loader_dataset([test_png_path], batch_size=2, repeat=True, 
                              shuffle=True)
    # pull a batch
    for x in ds:
        batch = x.numpy()
        break
    
    assert batch.shape == (2, 32, 32, 3)
    
def test_circle_mask_dataset():
    ds = circle_mask_dataset((10,10), batch_size=0)
    assert isinstance(ds, tf.data.Dataset)
    
    for x in ds:
        x = x.numpy()
        break
    assert x.shape == (10,10,1)