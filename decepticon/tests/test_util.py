# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image

from decepticon._util import _load_to_array


def test_load_to_array():
    test_arr = np.ones((10, 10), dtype=np.uint8)
    test_img = Image.fromarray(test_arr)
    
    for test in [test_arr, test_img]:
        output = _load_to_array(test)
        assert isinstance(output, np.ndarray)
        assert output.dtype == np.float32
        assert (output >= 0).all()
        assert (output <= 1).all()