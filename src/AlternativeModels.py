import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pdb


def LR(x_train, x_test, y_train, y_test, x_sensitive_train, x_sensitive_test):
    clf_predictor = LogisticRegression()
    clf_corrector = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    clf_predictor.fit(x_train, y_train)
    clf_corrector.fit(x_train, x_sensitive_train)
    y_pred = clf_predictor.predict(x_test)
    x_sensitive_pred = clf_corrector.predict(x_test)
    y_acc = accuracy_score(y_test, y_pred)
    x_sensitive_acc = accuracy_score(x_sensitive_test, x_sensitive_pred)
    print("predictor: {0:<4.4f}\t corrector:{1:<4.4f}".format(y_acc, x_sensitive_acc))
