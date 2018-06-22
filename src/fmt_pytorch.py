import random
import os
import sys
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchsummary import summary
import numpy as np
import datetime
import argparse
import pdb

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import logging

from collections import defaultdict

torch.manual_seed(42)
from util import load_config, read_data


class Filter(nn.Module):
    def __init__(self, D_in, D_out, h1, h2):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(D_in, h1),
            nn.PReLU(),
            nn.BatchNorm1d(h1),
            nn.Dropout(0.5),
            nn.Linear(h1, h2),
            nn.PReLU(),
            nn.BatchNorm1d(h2),
            nn.Dropout(0.5),
            nn.Linear(h2, D_out),
            nn.PReLU(),
        )

    def forward(self, *input):
        return self.main(*input)


class Predictor(nn.Module):
    def __init__(self, D_in, n_class, h1, h2):
        super().__init__()
        self.fc1 = nn.Linear(D_in, h1)
        self.main = nn.Sequential(
            nn.Linear(D_in, h1),
            nn.PReLU(),
            nn.BatchNorm1d(h1),
            nn.Dropout(0.5),
            nn.Linear(h1, h2),
            nn.PReLU(),
            nn.BatchNorm1d(h2),
            nn.Dropout(0.5),
            nn.Linear(h2, n_class),
            nn.Sigmoid()
        )

    def forward(self, *input):
        return self.main(*input)


class Corrector(nn.Module):
    def __init__(self, D_in, n_class, h1, h2):
        super().__init__()
        self.fc1 = nn.Linear(D_in, h1)
        self.main = nn.Sequential(
            nn.Linear(D_in, h1),
            nn.PReLU(),
            nn.BatchNorm1d(h1),
            nn.Dropout(0.5),
            nn.Linear(h1, h2),
            nn.PReLU(),
            nn.BatchNorm1d(h2),
            nn.Dropout(0.5),
            nn.Linear(h2, n_class),
            nn.Sigmoid()
        )

    def forward(self, *input):
        return self.main(*input)


class FMT():
    def __init__(self, input_shape, num_y_class, num_sensitive_x_class, filter_output_shape, filter_params,
                 predictor_params,
                 corrector_params, weight):
        self.weight = weight

        self.netFilter = Filter(input_shape, filter_output_shape, *filter_params)

        self.netPredictor = Predictor(filter_output_shape, num_y_class, *predictor_params)

        self.netCorrector = Corrector(filter_output_shape, num_sensitive_x_class, *corrector_params)

        self.predictorCriterion = nn.CrossEntropyLoss()
        self.correctorCriterion = nn.CrossEntropyLoss()

        self.optimizerFilter = optim.SGD(self.netFilter.parameters(), lr=1e-4, momentum=0.9)
        self.optimizerPredictor = optim.SGD(self.netPredictor.parameters(), lr=1e-4, momentum=0.9)
        self.optimizerCorrector = optim.SGD(self.netCorrector.parameters(), lr=1e-4, momentum=0.9)

        print('---------- Networks architecture -------------')
        print_network(self.netFilter)
        print_network(self.netPredictor)
        print_network(self.netCorrector)
        print('-----------------------------------------------')

    def train(self, num_epochs, batch_size, X_train, X_test, y_train, y_test, x_sensitive_train, x_sensitive_test):
        train_history = defaultdict(list)
        test_history = defaultdict(list)
        for epoch in range(num_epochs):

            print('Epoch {0}/{1}'.format(epoch, num_epochs))
            logger.info('Epoch {0}/{1}'.format(epoch, num_epochs))

            num_batches = int(X_train.shape[0] / batch_size)
            epoch_filter_loss, epoch_predictor_loss, epoch_corrector_loss = [], [], []

            for index in range(num_batches):
                X_train_batch = X_train[index * batch_size:(index + 1) * batch_size]
                y_train_batch = y_train[index * batch_size:(index + 1) * batch_size]
                x_sensitive_train_batch = x_sensitive_train[index * batch_size:(index + 1) * batch_size]

                X_train_tensor = Variable(torch.FloatTensor(X_train_batch), requires_grad=False)
                X_test_tensor = Variable(torch.FloatTensor(X_test), requires_grad=False)

                y_train_tensor = Variable(torch.FloatTensor(y_train_batch).long(), requires_grad=False)
                y_test_tensor = Variable(torch.FloatTensor(y_test).long(), requires_grad=False)
                x_sensitive_train_tensor = Variable(torch.FloatTensor(x_sensitive_train_batch).long(),
                                                    requires_grad=False)
                x_sensitive_test_tensor = Variable(torch.FloatTensor(x_sensitive_test).long(), requires_grad=False)

                # train predictor
                self.optimizerPredictor.zero_grad()
                filter_out = self.netFilter(X_train_tensor)
                predictor_out = self.netPredictor(filter_out)
                # pdb.set_trace()
                predictor_loss = self.predictorCriterion(predictor_out, y_train_tensor)
                epoch_predictor_loss.append(predictor_loss.item())
                predictor_loss.backward()
                self.optimizerPredictor.step()

                # train corrector
                corrector_out = self.netCorrector(filter_out)
                corrector_loss = self.correctorCriterion(corrector_out, x_sensitive_train_tensor)
                epoch_corrector_loss.append(corrector_loss.item())
                corrector_loss.backward()
                self.optimizerCorrector.step()

                # train filter

                filter_loss = self.predictorCriterion(predictor_out, y_train_tensor) * self.weight[0] + \
                              self.correctorCriterion(corrector_out, x_sensitive_train_tensor) * self.weight[1]
                epoch_filter_loss.append(filter_loss.item())
                filter_loss.backward()
                self.optimizerFilter.step()
                # pdb.set_trace()

            # print('Testing for epoch {}:'.format(epoch))
            # logger.info('Testing for epoch {}:'.format(epoch))
            #
            # train_predictor_loss = self.predictor.evaluate(X_train, y_train,
            #                                                verbose=False)  # np.mean(np.array(epoch_predictor_loss), axis=0)
            # train_corrector_loss = self.corrector.evaluate(X_train, x_sensitive_train,
            #                                                verbose=False)  # np.mean(np.array(epoch_predictor_loss), axis=0)
            #
            # test_predictor_loss = self.predictor.evaluate(X_test, y_test, verbose=False)
            # test_corrector_loss = self.corrector.evaluate(X_test, x_sensitive_test, verbose=False)
            #
            # train_history['predictor'].append(train_predictor_loss)
            # train_history['corrector'].append(train_corrector_loss)
            #
            # test_history['predictor'].append(test_predictor_loss)
            # test_history['corrector'].append(test_corrector_loss)
            # # pdb.set_trace()
            # print('{0:<22s} | {1:4s} | {2:15s} '.format(
            #     'component', *self.predictor.metrics_names))
            # print('-' * 60)
            # logger.info('{0:<22s} | {1:4s} | {2:15s} '.format(
            #     'component', *self.predictor.metrics_names))
            # logger.info('-' * 60)
            #
            # ROW_FMT = '{0:<22s} | {1:<4.2f} | {2:<15.4f}'
            # print(ROW_FMT.format('predictor (train)',
            #                      *train_history['predictor'][-1]))
            # print(ROW_FMT.format('predictor (test)',
            #                      *test_history['predictor'][-1]))
            # print(ROW_FMT.format('corrector (train)',
            #                      *train_history['corrector'][-1]))
            # print(ROW_FMT.format('corrector (test)',
            #                      *test_history['corrector'][-1]))
            #
            # logger.info(ROW_FMT.format('predictor (train)',
            #                            *train_history['predictor'][-1]))
            # logger.info(ROW_FMT.format('predictor (test)',
            #                            *test_history['predictor'][-1]))
            # logger.info(ROW_FMT.format('corrector (train)',
            #                            *train_history['corrector'][-1]))
            # logger.info(ROW_FMT.format('corrector (test)',
            #                            *test_history['corrector'][-1]))


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print("Total number of parameters: {0}".format(num_params))


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

    filter_output_shape = 30
    filter_params = [50, 50]
    predictor_params = [10, 10]
    corrector_params = [10, 10]

    model = FMT(input_shape=X_train.shape[1], num_y_class=len(np.unique(df[response])),
                num_sensitive_x_class=len(np.unique(df[['y_sensitive']])), filter_output_shape=filter_output_shape,
                filter_params=filter_params,
                corrector_params=corrector_params, predictor_params=predictor_params,
                weight=weight)

    model.train(num_epochs=20, batch_size=256, X_train=X_train, X_test=X_test, y_train=y_train.ravel(),
                y_test=y_test.ravel(),
                x_sensitive_train=x_sensitive_train.ravel(), x_sensitive_test=x_sensitive_test.ravel())
