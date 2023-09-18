import tensorflow as tf

def L1_LOSS(fusion, ori):
    maes = tf.losses.absolute_difference(ori, fusion)
    maes_loss = tf.reduce_mean(maes)
    return maes_loss


