import numpy as np
import tensorflow as tf
tf.enable_eager_execution()

from decepticon._synthetic import build_synthetic_dataset



def test_build_synthetic_dataset_classifier_training_mode(imshape=(64,64)):
    ds = build_synthetic_dataset(prob=0.5, imshape=imshape, num_empty=10,
                                 return_labels=True)
    for img, label in ds:
        img = img.numpy()
        label = label.numpy()
        break
    
    assert isinstance(label, np.int64)
    assert isinstance(img, np.ndarray)
    assert img.shape == (imshape[0], imshape[1], 3)
    
    

def test_build_synthetic_dataset_inpainter_training_mode(imshape=(64,64)):
    ds = build_synthetic_dataset(prob=0.0, imshape=imshape, num_empty=0,
                                 return_labels=False)
    for img in ds:
        img = img.numpy()
        break
    
    assert isinstance(img, np.ndarray)
    assert img.shape == (imshape[0], imshape[1], 3)