import numpy as np
import tensorflow as tf

from decepticon._synthetic import build_synthetic_dataset


def test_build_synthetic_dataset(imshape=(64,64)):
    ds = build_synthetic_dataset(imshape)
    n = ds.make_one_shot_iterator().get_next()
    with tf.Session() as sess:
        img, label = sess.run(n)
        
    assert isinstance(label, np.int64)
    assert isinstance(img, np.ndarray)
    assert img.shape == (imshape[0], imshape[1], 3)