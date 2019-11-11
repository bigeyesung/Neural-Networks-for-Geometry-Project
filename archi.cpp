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