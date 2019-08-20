import numpy as np
import tensorflow as tf
tf.enable_eager_execution()

from decepticon._synthetic import build_synthetic_dataset, _rand_rectangle
from decepticon._synthetic import _generate_img


def test_rand_rectangle():
    top, left, bottom, right = _rand_rectangle(128,128)
    
    assert (top >=0)&(top < 128)
    assert (left >=0)&(left < 128)
    assert (bottom >=0)&(bottom < 128)&(bottom > top)
    assert (right >=0)&(right < 128)&(right > left)


def test_generate_img_label_1():
    img = _generate_img((32, 48), 1, num_rectangles=25, num_empty=0)
    
    assert isinstance(img, np.ndarray)
    assert img.shape == (32, 48, 3)
    assert (img >= 0).all()
    assert (img <= 1).all()



def test_generate_img_label_0():
    img = _generate_img((32, 48), 0, num_rectangles=25, num_empty=0)
    
    assert isinstance(img, np.ndarray)
    assert img.shape == (32, 48, 3)
    assert (img >= 0).all()
    assert (img <= 1).all()


def test_generate_img_label_empty():
    img = _generate_img((32, 48), 1, num_rectangles=25, num_empty=10)
    
    assert isinstance(img, np.ndarray)
    assert img.shape == (32, 48, 3)
    assert (img >= 0).all()
    assert (img <= 1).all()

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