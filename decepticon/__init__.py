# -*- coding: utf-8 -*-

"""Top-level package for decepticon."""

__author__ = """Joe Gezo"""
__email__ = 'joe@gezo.net'
__version__ = '0.1.0'


from decepticon._models import build_inpainter, build_classifier
from decepticon._models import build_mask_generator 
from decepticon._models import build_discriminator, build_mask_discriminator
from decepticon._synthetic import build_synthetic_dataset, build_inpainter_dataset
from decepticon._trainers import Trainer
import decepticon.loaders