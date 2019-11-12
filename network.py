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

                # Creates a data set that reads all of the examples from file names.
        dataset = tf.data.TFRecordDataset(training_data_files)

        # Parse the record into tensors.
        dataset = dataset.map(ops._parse_function)

                # Shuffle the data set
        dataset = dataset.shuffle(buffer_size=self.config.shuffle_size_TFRecords)

        # Repeat the input indefinitely
        dataset = dataset.repeat()

                # Generate batches
        dataset = dataset.batch(self.config.batch_size)
        dataset = dataset.prefetch(self.config.batch_size * 2)

        # Create a one-shot iterator
        iterator = dataset.make_one_shot_iterator()
        self.anc_training_batch, self.pos_training_batch = iterator.get_next()

            def _build_model(self):
        """Build 3DSmoothNet network for testing."""

        # -------------------- Network archintecture --------------------
        from core.architecture import network_architecture

                # Build graph
        print("Building the 3DSmoothNet graph")

        self.keep_probability = tf.placeholder(tf.float32)

        # Build network for training usinf the tf_records files
        self.anchor_output, self.positive_output = network_architecture(self.anc_training_batch,
                                                                        self.pos_training_batch,
                                                                        self.keep_probability, self.config)

                                                                                # Build network for testing and validation that uses the placeholders for data input
        self.test_anchor_output, self.test_positive_output = network_architecture(self.anchor_input,
                                                                                  self.positive_input,
                                                                                  self.keep_probability, self.config,
                                                                                  reuse=True)