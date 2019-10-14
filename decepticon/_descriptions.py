# markdown descriptions for tensorboard outputs go here
from collections import defaultdict

maskgen = """
# Mask generator total loss

Total loss used in updating mask generator. Weighted sum of:
    
    * Classification loss
    * Exponential loss
    * Mask-discriminator "prior" loss
"""

cls_loss = """
# Classifier loss for mask generator

Crossentropy loss for class 0- helps train mask generator by reducing the classifier's assessed probability that it contains an object.
"""

exp_loss = """
# Exponential loss for mask generator

Helps bias mask generator by favoring smaller masks
"""

prior_loss = """
# Discriminator loss for mask generator

Helps bias mask generator toward compact masks
"""


inpainter_total_loss="""
# Inpainter total loss

Weighted sum of:
    
    * Reconstruction loss
    * Style loss
    * Discriminator loss
"""


recon_loss = """
# Reconstruction loss for inpainter

L1-norm between reconstructed image and ground-truth
"""

disc_loss = """
# Discriminator loss for inpainter

Weighted sum of pixel-wise crossentropy loss from the discriminator
"""
style_loss = """
# Style loss for inpainter

Helps train the inpainter to generate realistic images using neural style transfer loss
"""

maskdisc_loss = """
# Loss for mask discriminator
"""

image_display="""
# Object removal components

From left to right:
    
    * original image
    * generated mask
    * inpainter outputs
    * reconstructed image
"""

reconstructed_classifier_score="""
# Reconstructed classifier score

For evaluation images- histogram of classifier outputs.

If we've done everything right, should be near zero.
"""


disc_score_real="""
# Discriminator score on real pixels

Pixel-wise score for real parts of the image
"""


disc_score_fake="""
# Discriminator score on fake pixels

Pixel-wise score for fake parts of the image
"""

loss_descriptions = defaultdict(str)
loss_descriptions["mask_generator_total_loss"] = maskgen
loss_descriptions["mask_generator_classifier_loss"] = cls_loss
loss_descriptions["mask_generator_exponential_loss"] = exp_loss
loss_descriptions["mask_generator_prior_loss"] = prior_loss
loss_descriptions["inpainter_style_loss"] = style_loss
loss_descriptions["inpainter_total_loss"] = inpainter_total_loss
loss_descriptions["inpainter_reconstruction_L1_loss"] = recon_loss
loss_descriptions["discriminator_GAN_loss"] = disc_loss
loss_descriptions["mask_discriminator_loss"] = maskdisc_loss