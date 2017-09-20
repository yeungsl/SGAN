"""
Stacked GAN, translated from https://github.com/xunhuang1995/SGAN (theano implementation)
Tensorflow version
"""
import numpy as np
import tensorflow as tf

## Some helper functions?
def centered_softplus(x):
    return tf.nn.softplus(x) - tf.convert_to_tensor(np.log(2.))

def log_sum_exp(x, axis=1):
    m = tf.reduce_max(x, axis=axis)
    return m + tf.reduce_logsumexp(x-tf.expand_dims(m, 1), axis=axis)

def adam_updates(params, cost, lr=0.001, mom1=0.9, mom2=0.999):
    updates = []
    grads = tf.gradients(cost, params)
    t = tf.Variable(1.0, dtype=tf.float32)
    for p, g in zip(params, grads):
        
