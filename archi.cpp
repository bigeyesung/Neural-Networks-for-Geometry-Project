# Import python dependencies
import tensorflow as tf
import numpy as np

# Import custom functions
from core import ops

def network_architecture(x_anc,x_pos, dropout_rate, config, reuse=False):

    # Join the 3DSmoothNet structure with the desired output dimension
    net_structure = [1, 32, 32, 64, 64, 128, 128]
    outputDim = config.output_dim
    channels = [item for sublist in [net_structure, [outputDim]] for item in sublist]

    # In the third layer stride is 2
    stride = np.ones(len(channels))
    stride[2] = 2

    # Apply dropout in the 6th layer
    dropout_flag = np.zeros(len(channels))
    dropout_flag[5] = 1