==========
decepticon
==========

teaching machines to lie


.. image:: https://img.shields.io/pypi/v/decepticon.svg
        :target: https://pypi.python.org/pypi/decepticon

.. image:: https://img.shields.io/travis/jg10545/decepticon.svg
        :target: https://travis-ci.org/jg10545/decepticon

.. image:: https://readthedocs.org/projects/decepticon/badge/?version=latest
        :target: https://decepticon.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status


* Free software: MIT license
* Documentation: https://decepticon.readthedocs.io.

This repository contains a tensorflow/keras implementation of the elegant object removal model described in `Adversarial Scene Editing: Automatic Object Removal from Weak Supervision <https://arxiv.org/abs/1806.01911>`_ by Shetty, Fritz, and Schiele.


Usage
-----

**Data**

**Models**

Shetty *et al*'s model has several components; `decepticon` expects a `keras` Model object for each:

+--------+-----------+-----+------+----+
| Component | Description | Input | Output | Code |
+========+===========+=====+======+====+
| Component | Description | Input | Output | Code |
+--------+-----------+-----+------+----+

| **mask generator** | fully-convolutional network that generates a mask from an input image | ``(None, None, None, 3)`` | ``(None, None, None, 1)`` | ``decepticon.build_mask_generator()`` |
+--------+-----------+-----+------+----+



* **mask generator:** a fully-convolutional network that inputs a ``(None, None, None, 3)`` tensor containing a batch of images and returns a ``(None, None, None, 1)`` tensor containing a batch of masks. Use ``decepticon.build_mask_generator()`` to initialize the model from the paper.
* **classifier:** a convnet that inputs a ``(None, None, None, 3)`` tensor containing a batch of images and outputs



Features
--------

* TODO

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
