from unittest import TestCase
from utils.metrics import Metrics
import numpy as np

class TestMetrics(TestCase):
    def test_rmsle(self):
        rmsle = Metrics().rmsle
        y_true = np.array([0, np.exp(2) - 1])
        y_pred = np.array([0, np.exp(1) - 1])

        result = rmsle(y_true, y_pred)
        self.assertEqual(result, np.sqrt(.5))
