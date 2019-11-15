
import numpy as np
import os
import pickle
from os.path import isfile, join, sep
import subprocess
from glob import glob
import tensorflow as tf

def npy_to_tfrecords(features,pairs,output_file):
    # write records to a tfrecords file
    writer = tf.python_io.TFRecordWriter(output_file)