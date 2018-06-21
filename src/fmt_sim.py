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
from keras.utils.generic_utils import Progbar

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from collections import defaultdict

from util import load_config, read_data

import pdb
import sys
import logging
import datetime
import dill
import argparse


class FMT():
    def __init__(self, input_shape, num_y_class, num_sensitive_x_class, filter_output_shape, filter_params,
                 predictor_params,
                 corrector_params, optimizer_params, weight):
        self.weight = weight
        self.input_shape, self.filter_output_shape = None, None
        self.n_y_class, self.num_sensitive_x_class = num_y_class, num_sensitive_x_class

        self.input_shape = input_shape
        self.filter_output_shape = filter_output_shape

        lr, decay, momentum, nesterov = optimizer_params

        self.filter_optimizer, self.predictor_optimizer, self.corrector_optimizer = [
            SGD(lr=lr, decay=decay, momentum=momentum, nesterov=nesterov) for _ in range(3)]

        input_filter = Input(shape=(self.input_shape,))

        filter_layer = self.build_filter_layer(input_filter, *filter_params)
        predictor_layer = self.build_predictor_layer(filter_layer, *predictor_params)
        corrector_layer = self.build_corrector_layer(filter_layer, *corrector_params)

        # create the filter model first; keep filter trainable and both predictor and corrector non-trainable
        self.filter = Model(inputs=input_filter, outputs=[predictor_layer, corrector_layer])

        self.filter.compile(optimizer=self.filter_optimizer,
                            loss=['sparse_categorical_crossentropy', 'sparse_categorical_crossentropy'],
                            metrics=['accuracy'],
                            loss_weights=self.weight)
        # next create the predictor model; keep filter non-trainable
        self.predictor = Model(inputs=input_filter, outputs=predictor_layer)

        self.predictor.compile(optimizer=self.predictor_optimizer, loss='sparse_categorical_crossentropy',
                               metrics=['accuracy'])
        # next create the corrector model; keep filter non-trainable
        self.corrector = Model(inputs=input_filter, outputs=corrector_layer)
        self.corrector.compile(optimizer=self.corrector_optimizer, loss='sparse_categorical_crossentropy',
                               metrics=['accuracy'])

        self.filter.summary()
        self.predictor.summary()
        self.corrector.summary()
        # pdb.set_trace()

    def build_filter_layer(self, input_layer, h1, l1, h2, l2):
        name_ = 'filter'
        layer = Dense(h1, kernel_initializer='normal', name=name_ + '_1')(input_layer)
        if l1 == 'PReLU':
            layer = PReLU(name=name_ + '_2')(layer)
        else:
            layer = Activation(l1, name=name_ + '_2')(layer)
        layer = BatchNormalization(name=name_ + '_3')(layer)
        layer = Dropout(0.5, name=name_ + '_4')(layer)
        layer = Dense(h2, kernel_initializer='normal', name=name_ + '_5')(layer)
        if l2 == 'PReLU':
            layer = PReLU(name=name_ + '_6')(layer)
        else:
            layer = Activation(l1, name_ + '_6')(layer)
        layer = Dropout(0.5, name=name_ + '_7')(layer)
        layer = Dense(self.filter_output_shape, kernel_initializer='normal', name=name_ + '_8')(layer)
        return layer

    def build_predictor_layer(self, filter_layer, h1, l1, h2, l2):
        layer = Dense(h1, kernel_initializer='normal')(filter_layer)
        if l1 == 'PReLU':
            layer = PReLU()(layer)
        else:
            layer = Activation(l1)(layer)
        layer = BatchNormalization()(layer)
        layer = Dropout(0.5)(layer)
        layer = Dense(h2, kernel_initializer='normal')(layer)
        if l2 == 'PReLU':
            layer = PReLU()(layer)
        else:
            layer = Activation(l1)(layer)
        layer = Dropout(0.5)(layer)
        layer = Dense(self.n_y_class, kernel_initializer='normal')(layer)
        return layer

    def build_corrector_layer(self, filter_layer, h1, l1, h2, l2):
        layer = Dense(h1, kernel_initializer='normal')(filter_layer)
        if l1 == 'PReLU':
            layer = PReLU()(layer)
        else:
            layer = Activation(l1)(layer)
        layer = BatchNormalization()(layer)
        layer = Dropout(0.5)(layer)
        layer = Dense(h2, kernel_initializer='normal')(layer)
        if l2 == 'PReLU':
            layer = PReLU()(layer)
        else:
            layer = Activation(l1)(layer)
        layer = Dropout(0.5)(layer)
        layer = Dense(self.num_sensitive_x_class, kernel_initializer='normal')(layer)
        return layer

    def train(self, num_epochs, batch_size, X_train, X_test, y_train, y_test, x_sensitive_train, x_sensitive_test):
        train_history = defaultdict(list)
        test_history = defaultdict(list)
        for epoch in range(num_epochs):

            print('Epoch {0}/{1}'.format(epoch, num_epochs))
            logger.info('Epoch {0}/{1}'.format(epoch, num_epochs))

            num_batches = int(X_train.shape[0] / batch_size)
            progress_bar = Progbar(target=num_batches)
            epoch_filter_loss, epoch_predictor_loss, epoch_corrector_loss = [], [], []

            for index in range(num_batches):
                X_train_batch = X_train[index * batch_size:(index + 1) * batch_size]
                y_train_batch = y_train[index * batch_size:(index + 1) * batch_size]
                x_sensitive_train_batch = x_sensitive_train[index * batch_size:(index + 1) * batch_size]

                epoch_filter_loss.append(
                    self.filter.train_on_batch(X_train_batch, [y_train_batch, x_sensitive_train_batch]))
                progress_bar.update(index + 1)

            print('Testing for epoch {}:'.format(epoch))
            logger.info('Testing for epoch {}:'.format(epoch))

            train_predictor_loss = self.predictor.evaluate(X_train, y_train,
                                                           verbose=False)  # np.mean(np.array(epoch_predictor_loss), axis=0)
            train_corrector_loss = self.corrector.evaluate(X_train, x_sensitive_train,
                                                           verbose=False)  # np.mean(np.array(epoch_predictor_loss), axis=0)

            test_predictor_loss = self.predictor.evaluate(X_test, y_test, verbose=False)
            test_corrector_loss = self.corrector.evaluate(X_test, x_sensitive_test, verbose=False)

            train_history['predictor'].append(train_predictor_loss)
            train_history['corrector'].append(train_corrector_loss)

            test_history['predictor'].append(test_predictor_loss)
            test_history['corrector'].append(test_corrector_loss)
            # pdb.set_trace()
            print('{0:<22s} | {1:4s} | {2:15s} '.format(
                'component', *self.predictor.metrics_names))
            print('-' * 60)
            logger.info('{0:<22s} | {1:4s} | {2:15s} '.format(
                'component', *self.predictor.metrics_names))
            logger.info('-' * 60)

            ROW_FMT = '{0:<22s} | {1:<4.2f} | {2:<15.4f}'
            print(ROW_FMT.format('predictor (train)',
                                 *train_history['predictor'][-1]))
            print(ROW_FMT.format('predictor (test)',
                                 *test_history['predictor'][-1]))
            print(ROW_FMT.format('corrector (train)',
                                 *train_history['corrector'][-1]))
            print(ROW_FMT.format('corrector (test)',
                                 *test_history['corrector'][-1]))

            logger.info(ROW_FMT.format('predictor (train)',
                                       *train_history['predictor'][-1]))
            logger.info(ROW_FMT.format('predictor (test)',
                                       *test_history['predictor'][-1]))
            logger.info(ROW_FMT.format('corrector (train)',
                                       *train_history['corrector'][-1]))
            logger.info(ROW_FMT.format('corrector (test)',
                                       *test_history['corrector'][-1]))


def model_arg(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weight', type=float, nargs=2, help='weights for predictor and corrector')
    parsed_args = parser.parse_args(args)
    return [parsed_args.weight]


if __name__ == '__main__':
    weight, = model_arg(sys.argv[1:])
    logging.basicConfig(
        filename=os.path.join(os.path.dirname(os.path.abspath('__file__')), 'logs',
                              'fmt_' + str(datetime.date.today()) + '.txt'), level=logging.INFO
    )
    logger = logging.getLogger('fmt')

    config = load_config()
    df = read_data(config, 'adult')
    logger.info("{0}: shape of data read - {1}".format(datetime.datetime.now(), df.shape))

    sensitive_features = ['race', 'sex', 'race-sex']
    response = ['income-per-year']
    normal_features = [e for e in df.columns if e not in (sensitive_features + response)]
    le = LabelEncoder()
    df['y_sensitive'] = le.fit_transform(df['race'].map(str) + '-' + df['sex'].map(str))

    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    test_ratio = 0.25

    X_train, X_test, y_train, y_test, x_sensitive_train, x_sensitive_test = train_test_split(
        np.array(df[normal_features]), np.array(df[response]), np.array(df[['y_sensitive']]), test_size=test_ratio,
        random_state=42)

    scaler = MinMaxScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    logger.info("\n" * 5)
    logger.info("{0}: train with data of shape {1}".format(datetime.datetime.now(), X_train.shape))
    logger.info("{0}: test with data of shape {1}".format(datetime.datetime.now(), X_test.shape))

    filter_output_shape = 10

    filter_params = [50, 'PReLU', 50, 'PReLU']
    corrector_params = [5, 'PReLU', 5, 'PReLU']
    predictor_params = [5, 'PReLU', 5, 'PReLU']
    optimizer_params = [0.025, 1e-6, 0.95, False]

    model = FMT(input_shape=X_train.shape[1], num_y_class=len(np.unique(df[response])),
                num_sensitive_x_class=len(np.unique(df[['y_sensitive']])), filter_output_shape=filter_output_shape,
                filter_params=filter_params,
                corrector_params=corrector_params, predictor_params=predictor_params, optimizer_params=optimizer_params,
                weight=weight)

    for m in ['model' + '.' + e for e in ['filter', 'predictor', 'corrector']]:
        plot_model(eval(m), show_layer_names=True, show_shapes=True, to_file=os.path.join(os.path.dirname(
            os.path.abspath('__file__')
        ), 'output', m + '_plot.png'))

    model.train(num_epochs=20, batch_size=256, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                x_sensitive_train=x_sensitive_train, x_sensitive_test=x_sensitive_test)

