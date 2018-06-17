import numpy as np
import pandas as pd
import tensorflow as tf
import random as rn
import os

os.environ['PYTHONHASHSEED'] = '0'

np.random.seed(42)
rn.seed(42)

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

from keras import backend as K

tf.set_random_seed(42)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

from keras.models import Sequential
from keras.callbacks import Callback
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.optimizers import SGD
from keras.utils import plot_model

from sklearn.metrics import roc_auc_score
from util import load_config, read_data

import logging
import datetime
import dill

if __name__ == '__main__':
    logging.basicConfig(
        filename=os.path.join(os.path.dirname(os.path.abspath('__file__')), 'logs',
                              'fmt_' + str(datetime.date.today()) + '.txt')
    )
    logger = logging.getLogger('fmt')

    config = load_config()
    dat = read_data(config, 'adult')
    print(dat.head(5))
