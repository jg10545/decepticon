![](docs/logo.png)

# decepticon

teaching machines to lie


![](https://img.shields.io/pypi/v/decepticon.svg)

![](https://img.shields.io/travis/jg10545/decepticon.svg)

![](https://readthedocs.org/projects/decepticon/badge/?version=latest)


* Free software: MIT license

This repository contains a tensorflow/keras implementation of the elegant object removal model described in [Adversarial Scene Editing: Automatic Object Removal from Weak Supervision](https://arxiv.org/abs/1806.01911) by Shetty, Fritz, and Schiele.

We've been testing using Python 3.6 and TensorFlow 1.14.

## Usage

### Data

So far we've only tested single-class object removal. You'll need two lists of filepaths- one to image patches containing objects and one without. All patches should be prepared to the same size.

### Models

Shetty *et al*'s model has several components; `decepticon` expects a `keras` Model object for each. You can reproduce the models from the paper or substitute your own so long as they have the correct inputs and outputs:

| **Component** | **Description** | **Input Shape** | **Output Shape** | **Code** |
| ---- | ---- | ---- | ---- | ---- |
| **mask generator** | fully-convolutional network that generates a mask from an input image | `(None, None, None, 3)` | `(None, None, None, 1)` | `decepticon.build_mask_generator()` |
| **classifier** | standard convolutional classifier that maps an image to a probability over categories| `(None, None, None, 3)` | `(None, num_classes+1)` | `decepticon.build_classifier()` |
| **inpainter** | fully-convolutional network that inputs a partially-masked image and attempts to generate the original unmasked version (like a  denoising autoencoder or [context encoder](https://arxiv.org/abs/1604.07379))| `(None, None, None, 3)` | `(None, None, None, 3)` | `decepticon.build_inpainter()` |
| **discriminator** | fully-convolutional network that inputs an image and makes a pixel-wise assessment about whether the image is real or fake| `(None, None, None, 3)` | `(None, None, None, 1)` | `decepticon.build_discriminator()` |
| **mask discriminator** | *not yet implemented*|  |  | |

If you're training on a consumer GPU you may run into memory limitations using the models from the paper and a reasonable batch size- if you pass the keyword argument `downsample=n` to any of the above functions, the number of filters in every hidden convolutional layer will be reduced by a factor of `n`.

### Pretraining

#### Classifier

The image classifier is trained on randomly-masked images:

```{python}
# initialize a classifier
classifier = decepticon.build_classifier()
classifier.compile(tf.keras.optimizers.Adam(1e-3),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=["accuracy"])
                  
# create a tensorflow dataset that generates randomly-masked batches
class_ds = decepticon.loaders.classifier_training_dataset(posfiles, negfiles, batch_size=32)
# train with the normal keras API
classifier.fit(class_ds, steps_per_epoch=250, epochs=5)
```

#### Inpainter

We've had better luck with the end-to-end model if the inpainter is pretrained for a few epochs.

```{python}
# initialize an inpainter
inpainter = decepticon.build_inpainter()
inpainter.compile(tf.keras.optimizers.Adam(1e-3),
                 loss=tf.keras.losses.mae)
                 
# pretrain only on randomly-masked negative patches, so it
# never learns to inpaint the objects you want to remove
inpaint_ds = decepticon.loaders.inpainter_training_dataset(negfiles)
inpainter.fit(inpaint_ds, steps_per_epoch=250, epochs=5)
```

### Training

The `Trainer` class manages the alternating-epoch (mask generator vs inpainter) and alternating-batch (inpainter and discriminator) training.

```{python}
# initialize Keras models for mask generator and discriminator
maskgen = decepticon.build_mask_generator()
disc = decepticon.build_discriminator()

# pass the model components and lists of file paths to
# a trainer function. also set training parameters and
# weights for all terms in the loss function here.
trainer = decepticon.build_image_file_trainer(posfiles, negfiles,
                                   classifier, inpainter,
                                    disc,maskgen,
                                    steps_per_epoch=1000,
                                    lr=1e-5,
                                    logdir="/path/to/log/directory/",
                                    batch_size=32,
                                    exponential_loss_weight=1,
                                    disc_weight=10,
                                   )

# run training loop
trainer.fit(10)
```

### Monitoring in TensorBoard

If a log directory is specified, models and TensorBoard logs will be saved there.

All the terms in the loss function will be recorded as scalars:

![](docs/tensorboard-l1-loss.png)

Histograms will record classification and discriminator probabilities on reconstructed examples, to visualize how well the system is fooling them:

![](docs/tensorboard-histogram.png)

Images will record examples of raw images, mask generator and inpainter outputs for them, and the reconstructed image:

![](docs/tensorboard-image.png)


### Inference

The end-to-end object removal network is stored as a Keras model in `trainer.full_model`; run inference using the normal `predict` interface.

## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage) project template.
