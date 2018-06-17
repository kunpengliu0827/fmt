from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
import yaml
import os


class AUCHistory(Callback):
    def __init__(self, validation_data, logger):
        self.X_val, self.Y_val = validation_data
        self.logger = logger

    def on_train_begin(self, logs=None):
        self.auc = []

    def on_epoch_begin(self, epoch, logs=None):
        epoch += 1
        Y_pred = self.model.predict_proba(self.X_val, verbose=0)
        current_auc = roc_auc_score(self.Y_val, Y_pred)
        self.auc.append(current_auc)
        self.logger.info("epoch {0:d}: AUC {1:0.6f}".format(epoch, current_auc))
        print("epoch {0:d}: AUC {1:0.6f}".format(epoch, current_auc))


def load_config():
    with open(os.path.join(os.path.dirname(os.path.abspath('__file__')), 'config.yaml'), 'r') as f:
        config = yaml.load(f)
    return config


def read_data(config, which_data):
    dat = pd.read_csv(config['data'][which_data])
    return dat
