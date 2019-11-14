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