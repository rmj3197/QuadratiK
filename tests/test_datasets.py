"""
Testing loading datasets with different arguments
"""

import numpy as np
import pandas as pd
import unittest
from QuadratiK.datasets import load_wireless_data


class TestLoadWirelessData(unittest.TestCase):

    def test_return_X_y(self):
        X, y = load_wireless_data(return_X_y=True, as_dataframe=False)
        self.assertEqual(X.shape[0], 2000)
        self.assertEqual(y.shape[0], 2000)
        self.assertTrue(isinstance(X, np.ndarray))
        self.assertTrue(isinstance(y, np.ndarray))

    def test_as_dataframe(self):
        df = load_wireless_data(return_X_y=False, as_dataframe=True)
        self.assertEqual(df.shape[0], 2000)
        self.assertEqual(df.shape[1], 8)
        self.assertTrue(isinstance(df, pd.DataFrame))

    def test_scaled(self):
        X, _ = load_wireless_data(return_X_y=True, as_dataframe=False)
        X_scaled, _ = load_wireless_data(
            return_X_y=True, scaled=True, as_dataframe=False
        )
        self.assertTrue(
            np.allclose(X / np.linalg.norm(X, axis=1, keepdims=True), X_scaled)
        )

    def test_desc(self):
        descr, _ = load_wireless_data(desc=True)
        self.assertTrue(isinstance(descr, str))

    def test_scaled_as_numpy(self):
        X, y = load_wireless_data(scaled=True, return_X_y=True)
        self.assertEqual(X.shape[0], 2000)
        self.assertEqual(y.shape[0], 2000)
        self.assertTrue(isinstance(X, np.ndarray))
        self.assertTrue(isinstance(y, np.ndarray))

    def test_desc_with_Xy(self):
        descr, X, y = load_wireless_data(desc=True, return_X_y=True)
        self.assertEqual(X.shape[0], 2000)
        self.assertEqual(y.shape[0], 2000)
        self.assertTrue(isinstance(descr, str))

    def test_as_numpy(self):
        df = load_wireless_data(as_dataframe=False)
        self.assertEqual(df.shape[0], 2000)
        self.assertEqual(df.shape[1], 8)
        self.assertTrue(isinstance(df, np.ndarray))
