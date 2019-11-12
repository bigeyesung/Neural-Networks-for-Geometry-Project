import numpy as np
import tensorflow as tf
import glob
import time
import os
import datetime
from core import ops

from tqdm import trange

class NetworkBuilder(object):
    """Network builder class """

    def __init__(self, config):

        self.config = config

        # Initialize tensorflow session
        self._init_tensorflow()

        # Build the network
        self._build_placeholder()
        self._build_data_loader()
        self._build_model()
        self._build_loss()
        self._build_optim()
        self._build_summary()
        self._build_writer()

    def _init_tensorflow(self):
        # Initialize tensorflow and let the gpu memory to grow
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True

        self.sess = tf.Session(config=tf_config)

    def _build_placeholder(self):

        # Create placeholders for the input to the siamese network
        self.anchor_input = tf.placeholder(dtype=tf.float32, shape=[None, int(np.cbrt(self.config.input_dim)),
                                                                    int(np.cbrt(self.config.input_dim)),
                                                                    int(np.cbrt(self.config.input_dim)), 1],
                                           name='X_reference')

        self.positive_input = tf.placeholder(dtype=tf.float32, shape=[None, int(np.cbrt(self.config.input_dim)),
                                                                      int(np.cbrt(self.config.input_dim)),
                                                                      int(np.cbrt(self.config.input_dim)), 1],
                                             name='X_positive')

                # Global step for optimization
        self.global_step = tf.Variable(0, trainable=False)

            def _build_data_loader(self):

        if not os.path.exists(self.config.training_data_folder):
            print('Error directory: {}'.format(self.config.training_data_folder))
            raise ValueError('The training directory {} does not exist.'.format(self.config.training_data_folder))

                    # Get name of all tfrecord files
        training_data_files = glob.glob(self.config.training_data_folder + '*.tfrecord')
        nr_training_files = len(training_data_files)
        print('Number of training files: {}'.format(nr_training_files))