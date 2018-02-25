from __future__ import division
import numpy as np


class Metrics(object):
    @staticmethod
    def rmsle(y_true, y_pred):
        n = len(y_true)
        return np.sqrt((1 / n) * np.sum((np.log(1 + y_pred) - np.log(1 + y_true)) ** 2))