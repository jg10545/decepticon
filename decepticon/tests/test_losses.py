import numpy as np
import tensorflow as tf
tf.enable_eager_execution()


from decepticon._losses import exponential_loss, least_squares_gan_loss



def test_empty_mask_output_value():
    zerotest = tf.constant(np.zeros((5,4,4,1), dtype=np.int64))
    exploss = exponential_loss(zerotest, zerotest)
    exploss = exploss.numpy()
    assert (exploss == np.ones(5)).all()
    
    
def test_full_mask_output_value():
    onetest = tf.constant(np.ones((7,4,4,1), dtype=np.int64))
    exploss_one = exponential_loss(onetest, onetest)
    exploss_one = exploss_one.numpy()
    assert np.max(np.abs(exploss_one - np.exp(1)*np.ones(7))) < 1e-5
    
    
def test_gan_loss_no_error():
    mask_test = tf.constant(np.ones((5,2,2,1), dtype=np.int64))
    no_error_loss = least_squares_gan_loss(mask_test, 
                                       tf.cast(mask_test, tf.float32))
    no_error_loss = no_error_loss.numpy()
    assert (no_error_loss == np.zeros(5)).all()
    
def test_gan_loss_one_pixel_wrong():
    mask_test2 = np.zeros((1,2,2,1), dtype=np.int64)
    mask_test2[0,0,0,0] = 1
    mask_test2 = tf.constant(mask_test2)
    preds = tf.constant(np.zeros((1,2,2,1), dtype=np.float32))

    one_spot_wrong_loss = least_squares_gan_loss(mask_test2, preds)
    one_spot_wrong_loss = one_spot_wrong_loss.numpy()
    assert one_spot_wrong_loss[0] == 0.5
    