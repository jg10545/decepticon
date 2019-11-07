import tensorflow as tf
import tensorflow.keras.backend as K

def exponential_loss(y_true, y_pred):
    """
    Loss function for mask training- exp(sum(mask)). 
    
    Assumes the mask is a tensor that looks like:
    
    (batch_size, height, width, 1)
    
    It'll need to be updated if we want to do multiclass removal.
    
    Also I'm using the mean and sum so the behavior shouldn't
    depend on image size.
    """
    y_pred = tf.cast(y_pred, tf.float32)
    mask_sums = tf.reduce_mean(y_pred, axis=[1,2,3])
    return K.exp(mask_sums)



def least_squares_gan_loss(y_true, y_pred):
    """
    Equation 7 from Shetty et al's paper. Output swapped so
    m instead of ~m is class 1 because I find it way more
    intuitive. Also using sigmoid instead of tanh outputs,
    out of habit.
    
    y_true would be the mask and y_pred is the pixelwise
    real/fake prediction from the discriminator. both 
    should be rank-4 tensors; (batch, H, W, 1)
    """
    mask = tf.cast(y_true, tf.float32) # 1 in inpainted regions
    inverse_mask = 1 - mask # 0 in inpainted regions
    
    # i'm adding a 1 to each sum since they're in the 
    # denominator, in case we ever get an empty mask
    mask_sum = tf.reduce_sum(mask, axis=[1,2,3]) + 1
    inv_mask_sum = tf.reduce_sum(inverse_mask, axis=[1,2,3]) + 1
    
    # for masked areas- penalize deviations from 1
    masked_area_loss = tf.reduce_sum(
        mask*K.square(y_pred - 1), axis=[1,2,3]
    )
    # for unmasked areas- penalize deviations from 0
    unmasked_area_loss = tf.reduce_sum(
        inverse_mask*K.square(y_pred), axis=[1,2,3]
    )
    return masked_area_loss/mask_sum + unmasked_area_loss/inv_mask_sum


def compute_gram_matrix(x):
    """
    Input a (batch_size, H, W, C) tensor; return a (C,C)
    Gram matrix normalized using equation 4 from Gatys'
    paper (4 * (H^2) * (W^2)) (since this is computed per matrix, 
    and style loss uses squared-error loss, we compute 2*H*W here)
    
    I've also scaled by N^2 since we're computing this for an
    entire batch
    """
    assert K.ndim(x) == 4
    N, H, W, C = x.get_shape().as_list()
    # flatten to a matrix
    x_flat = tf.reshape(x, [-1, C])
    # compute matrix of kernel correlations and normalize
    gram = tf.matmul(x_flat, x_flat, transpose_a=True)
    norm = 2*H*W*(N**2) 
    return gram/norm

def build_style_model():
    """
    Macro to generate a Keras model for computing the style
    representations of an image.
    """
    # load VGG19 and reform as a model that outputs the 
    # conv blocks used for style transfer
    vgg = tf.keras.applications.VGG19(weights="imagenet", include_top=False)
    output_dict = dict([(l.name, l.output) for l in vgg.layers])
    feature_layers = ['block1_conv1', 'block2_conv1',
                  'block3_conv1', 'block4_conv1',
                  'block5_conv1']
    style_model = tf.keras.Model(vgg.inputs, [output_dict[l] for l in feature_layers])
    return style_model

def compute_style_loss(x, y, style_model):
    """
    Input two batches of images and a style model and
    compute the squared-error loss between the style representations'
    Gram matrices.
    
    Note that we're using K.mean() instead of K.sum() (like the style loss
    implementation on the Keras website). Since the style loss is only
    one term out of several in the loss function, it's more important that 
    the natural values be on the unit scale to compare with the others.
    
    :x: (N, H, W, C) batch of images
    :y: (N, H, W, C) batch of images to compare with x
    :style_model: a Keras model that outputs a list of style representations
    """
    # each of these will be a list of rank-4 tensors
    x_style = style_model(x)
    y_style = style_model(y)
    
    loss = 0
    # for each layer
    for a,b in zip(x_style, y_style):
        # compute gram matrices
        x_gram = compute_gram_matrix(a)
        y_gram = compute_gram_matrix(b)
        # record loss
        loss += K.mean(K.square(x_gram - y_gram)) 
        
    return loss



def pixelwise_variance(x):
    """
    Compute the average variance-per-pixel for
    a batch of images (to make sure outputs aren't
    mode-collapsing)
    
    :x: an image with batch dimension 0
    """
    pixelwise_mean = tf.reduce_mean(x, 0)
    meandiff = x - tf.expand_dims(pixelwise_mean, 0)
    return tf.reduce_mean(meandiff**2)


def total_variation_loss(x):
    """
    Input a batch of images; return mean total variation
    loss.
    
    Check out https://en.wikipedia.org/wiki/Total_variation_denoising
    """
    vl = tf.image.total_variation(x)
    return tf.reduce_mean(vl)


def compute_gradient_penalty(real, fake, discriminator):
    """
    As described in "Improved Training of Wasserstein GANs"
    """
    eps = tf.random.uniform((), minval=0, maxval=1)
    with tf.GradientTape() as tape:
        # interpolate image batch
        x = eps*real + (1-eps)*fake
        tape.watch(x)
        disc_out = discriminator(x)
        # average outputs for pixelwise discriminator
        if len(disc_out.get_shape()) > 2:
            disc_out = tf.reduce_mean(disc_out, (1,2,3))
    # compute gradient with respect to interpolated image batch
    grad = tape.gradient(disc_out, x)
    grad_norm = tf.math.sqrt(tf.reduce_sum(tf.math.square(grad), (1,2,3)))
    return tf.reduce_mean(tf.square(grad_norm - 1))
        
        
        
        
        
        
        