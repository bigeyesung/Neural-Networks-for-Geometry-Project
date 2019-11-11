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

     # Initalize data
    input_anc = x_anc
    input_pos = x_pos
    layer_index = 0

    # Loop over the desired layers
    with tf.name_scope('3DIM_cnn') as scope:
        for layer in np.arange(0, len(channels)-2):
            scope_name = "3DIM_cnn" + str(layer_index+1)
            with tf.name_scope(scope_name) as inner_scope:
                input_anc, input_pos = conv_block(input_anc, input_pos, [channels[layer], channels[layer + 1]],
                                                  dropout_flag[layer], dropout_rate, layer_index,
                                                  stride_input=stride[layer], reuse=reuse)

            layer_index += 1