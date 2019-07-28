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