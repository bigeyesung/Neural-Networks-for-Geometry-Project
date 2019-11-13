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

                                                                                      def _build_loss(self):
        # Import the loss function
        from core import loss

        # Create mask for the batch_hard loss
        positiveIDS = np.arange(self.config.batch_size)
        positiveIDS = tf.reshape(positiveIDS, [self.config.batch_size])

        self.dists = loss.cdist(self.anchor_output, self.positive_output, metric='euclidean')
        self.losses = loss.LOSS_CHOICES['batch_hard'](self.dists, positiveIDS,
                                                      self.config.margin, batch_precision_at_k=None)

        # tf.summary.scalar("loss", self.losses)

            def _build_optim(self):
        # Build the optimizer
        starter_learning_rate = self.config.learning_rate
        learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                                   self.config.decay_step, self.config.decay_rate,
                                                   staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)

        # Merge all summary op
        self.summary_op = tf.summary.merge_all()

        # Adam optimization, with the adaptable learning_rate
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.losses, global_step=self.global_step)
        self.optimization_parameters = [optimizer, self.losses, self.summary_op]

            def _build_summary(self):
        """Build summary ops."""

        # Merge all summary op
        self.summary = tf.summary.merge_all()

            def _build_writer(self):
        self.saver = tf.train.Saver()
        self.saver = tf.train.Saver(max_to_keep=self.config.max_epochs)

        self.time_stamp_format = "%f_%S_%H_%M_%d_%m_%Y"
        time_stamp = datetime.datetime.now().strftime(self.time_stamp_format)
        self.base_file_name = 'lr_{}_batchSize_{}_outDim_{}_{}'.format(self.config.learning_rate, self.config.batch_size,
                                                                       self.config.output_dim, time_stamp)

                # Initlaize writer for the tensorboard
        if not os.path.exists(self.config.log_path + '/{}_dim/'.format(self.config.output_dim)):
            os.makedirs(self.config.log_path + '/{}_dim/'.format(self.config.output_dim))
            print('Created a folder: {}'.format(self.config.log_path +
                                                '/{}_dim/'.format(self.config.output_dim)))

        # Check which saved files are already existing
        output_dir = os.listdir(self.config.log_path + '/{}_dim/'.format(self.config.output_dim))
        temp_names = [d.split('_') for d in output_dir]
        temp_names = list(map(int, [item[-1] for item in temp_names]))