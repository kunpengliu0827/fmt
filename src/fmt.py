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

from keras.models import Sequential, Model
from keras.callbacks import Callback
from keras.layers import Dense, Dropout, Activation, BatchNormalization, Input
from keras.layers.advanced_activations import PReLU
from keras.optimizers import SGD
from keras.utils import plot_model

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from util import load_config, read_data
import pdb

import logging
import datetime
import dill


class FMT():
    def __init__(self, input_shape, filter_output_shape, filter_params, predictor_params,
                 corrector_params, optimizer_params, weight):
        self.weight = weight
        self.input_shape, self.filter_output_shape = None, None
        self.n_s_class, self.n_y_class = None, None

        self.input_shape = input_shape
        self.filter_output_shape = filter_output_shape

        filter_layer = self.build_filter_layer(*filter_params)
        predictor_layer = self.build_predictor_layer(*predictor_params)
        corrector_layer = self.build_corrector_layer(*corrector_params)

        # create the filter model first; keep filter trainable and both predictor and corrector non-trainable
        input_filter = Input(shape=self.input_shape)
        predictor_layer.trainable = False
        corrector_layer.trainable = False
        self.filter = Model(inputs=input_filter, outputs=[predictor_layer(filter_layer(input_filter)),
                                                          corrector_layer(filter_layer(input_filter))])

        self.filter.summary()

        # next create the predictor model; keep filter non-trainable
        filter_layer.trainable, predictor_layer.trainable = False, True
        self.predictor = Model(inputs=input_filter, outputs=predictor_layer(filter_layer(input_filter)))
        self.predictor.summary()

        # next create the corrector model; keep filter non-trainable
        filter_layer.trainable, corrector_layer.trainable = False, True
        self.corrector = Model(inputs=input_filter, outputs=corrector_layer(filter_layer(input_filter)))
        self.corrector.summary()

        lr, decay, momentum, nesterov = optimizer_params

        self.filter_optimizer, self.predictor_optimizer, self.corrector_optimizer = [
            SGD(lr=lr, decay=decay, momentum=momentum, nesterov=nesterov) for _ in range(3)]

        self.predictor.compile(optimizer=self.predictor_optimizer, loss='categorical_crossentropy')
        self.corrector.compile(optimizer=self.corrector_optimizer, loss='categorical_crossentropy')

        self.filter.compile(optimizer=self.filter_optimizer,
                            loss=['categorical_crossentropy', 'categorical_crossentropy'],
                            loss_weights=self.weight)

    def build_filter_layer(self, h1, l1, h2, l2):
        model = Sequential()
        model.add(Dense(h1, input_dim=self.input_shape, kernel_initializer='normal'))
        if l1 == 'PReLU':
            model.add(PReLU())
        else:
            model.add(Activation(l1))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(h2, kernel_initializer='normal'))
        if l2 == 'PReLU':
            model.add(PReLU())
        else:
            model.add(Activation(l2))
        model.add(Dropout(0.5))
        model.add(Dense(self.filter_output_shape))
        return model

    def build_predictor_layer(self, h1, l1, h2, l2):
        model = Sequential()
        model.add(Dense(h1, input_dim=self.filter_output_shape, kernel_initializer='normal'))
        if l1 == 'PReLU':
            model.add(PReLU())
        else:
            model.add(Activation(l1))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(h2, kernel_initializer='normal'))
        if l2 == 'PReLU':
            model.add(PReLU())
        else:
            model.add(Activation(l2))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        return model

    def build_corrector_layer(self, h1, l1, h2, l2):
        model = Sequential()
        model.add(Dense(h1, input_dim=self.filter_output_shape, kernel_initializer='normal'))
        if l1 == 'PReLU':
            model.add(PReLU())
        else:
            model.add(Activation(l1))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(h2, kernel_initializer='normal'))
        if l2 == 'PReLU':
            model.add(PReLU())
        else:
            model.add(Activation(l2))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        return model

    def train(self, nepochs):
        for epoch in range(nepochs):
            print('Epoch {0}/{1}'.format(epoch, nepochs))


if __name__ == '__main__':
    logging.basicConfig(
        filename=os.path.join(os.path.dirname(os.path.abspath('__file__')), 'logs',
                              'fmt_' + str(datetime.date.today()) + '.txt')
    )
    logger = logging.getLogger('fmt')

    config = load_config()
    df = read_data(config, 'adult')
    logger.info("{0}: shape of data read - {1}".format(datetime.datetime.now(), df.shape))

    sensitive_features = ['race', 'sex', 'race-sex']
    response = ['income-per-year']
    normal_features = [e for e in df.columns if e not in (sensitive_features + response)]
    df['y_sensitive'] = df['race'].map(str) + '-' + df['sex'].map(str)

    test_ratio = 0.25

    X_train, X_test, y_train, y_test, x_sensitive_train, x_sensitive_test = train_test_split(
        np.array(df[normal_features]), np.array(df[response]), np.array(df['y_sensitive']), test_size=test_ratio,
        random_state=42)

    logger.info("{0}: train with data of shape {1}".format(datetime.datetime.now(), X_train.shape))
    logger.info("{0}: test with data of shape {1}".format(datetime.datetime.now(), X_test.shape))

    filter_output_shape = 10

    filter_params = [50, 'PReLU', 50, 'PReLU']
    corrector_params = [5, 'PReLU', 5, 'PReLU']
    predictor_params = [5, 'PReLU', 5, 'PReLU']
    optimizer_params = [0.025, 1e-6, 0.95, False]

    weight = [1.0, 1.0]

    model = FMT(input_shape=X_train.shape[0], filter_output_shape=filter_output_shape, filter_params=filter_params,
                corrector_params=corrector_params, predictor_params=predictor_params, optimizer_params=optimizer_params,
                weight=weight)

    for m in ['model' + '.' + e for e in ['filter', 'predictor', 'corrector']]:
        plot_model(eval(m), show_layer_names=True, show_shapes=True, to_file=os.path.join(os.path.dirname(
            os.path.abspath('__file__')
        ), 'output', m + '_plot.png'))
