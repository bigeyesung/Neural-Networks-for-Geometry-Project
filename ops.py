# Import python dependencies
import tensorflow as tf
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Create lots of weights and biases & Initialize with a small positive number as we will use ReLU
def weight(shape, layer_name, weight_initializer=None,reuse=False):
    if weight_initializer is None:
        weight_initializer = tf.orthogonal_initializer(gain=0.6,
                                                       seed=41,
                                                       dtype=tf.float32)

                                                           with tf.name_scope(layer_name):
        with tf.variable_scope(layer_name, reuse=reuse):
            weights = tf.get_variable(layer_name + "_W", shape=shape,
                                      dtype=tf.float32, initializer=weight_initializer)
                                          tf.summary.histogram(layer_name, weights)
    return weights

    def bias(shape, layer_name,reuse=False):
    bias_init = tf.constant_initializer(0.01)

     with tf.name_scope(layer_name):
        with tf.variable_scope('', reuse=reuse):
            biases = tf.get_variable(layer_name + '_b',  shape=shape,
                                     dtype=tf.float32, initializer=bias_init)  # default initialier: glorot_uniform_initializer
    return biases

    def conv3d(x,filtertype , stride, padding):
    return tf.nn.conv3d(x, filter=filtertype, strides=[1, stride[0], stride[1], stride[2], 1], padding=padding)

    def max_pool3d(x, kernel, stride, padding):
    return tf.nn.max_pool3d(x, ksize=kernel, strides=stride, padding=padding)

    def avg_pool3d(x, kernel, stride, padding):
    return tf.nn.avg_pool3d(x, ksize=kernel, strides=stride, padding=padding)

    def relu(x):
    return tf.nn.relu(x)

    def batch_norm(x):
    epsilon = 1e-3
    batch_mean, batch_var = tf.nn.moments(x, [0])
    return tf.nn.batch_normalization(x, batch_mean, batch_var, None, None, epsilon)

    def l2_normalize(x):
    return tf.nn.l2_normalize(x, axis=1, epsilon=1e-12, name=None)

    def dropout(x,dropout_rate=0.7):
    return tf.nn.dropout(x,keep_prob=dropout_rate,noise_shape=None,seed=None,name=None)