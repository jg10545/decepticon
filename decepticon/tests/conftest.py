# -*- coding: utf-8 -*-
import os
import pytest


@pytest.fixture()
def test_png_path():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(this_dir, "fixtures", "test_png.png")
