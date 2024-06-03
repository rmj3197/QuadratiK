"""
Testing loading datasets with different arguments
"""

import numpy as np
import pandas as pd
import unittest
from QuadratiK.datasets import (
    load_wireless_data,
    load_wisconsin_breast_cancer_data,
    load_wine_data,
)


class TestLoadWirelessData(unittest.TestCase):

    def test_return_X_y(self):
        X1, y1 = load_wireless_data(return_X_y=True, as_dataframe=False)
        X2, y2 = load_wisconsin_breast_cancer_data(return_X_y=True, as_dataframe=False)
        X3, y3 = load_wine_data(return_X_y=True, as_dataframe=False)
        self.assertTrue(isinstance(X1, np.ndarray))
        self.assertTrue(isinstance(X2, np.ndarray))
        self.assertTrue(isinstance(X3, np.ndarray))
        self.assertTrue(isinstance(y1, np.ndarray))
        self.assertTrue(isinstance(y2, np.ndarray))
        self.assertTrue(isinstance(y3, np.ndarray))

    def test_as_dataframe(self):
        df1 = load_wireless_data(return_X_y=False, as_dataframe=True)
        df2 = load_wisconsin_breast_cancer_data(return_X_y=False, as_dataframe=True)
        df3 = load_wine_data(return_X_y=False, as_dataframe=True)
        self.assertTrue(isinstance(df1, pd.DataFrame))
        self.assertTrue(isinstance(df2, pd.DataFrame))
        self.assertTrue(isinstance(df3, pd.DataFrame))

    def test_scaled(self):
        X1, _ = load_wireless_data(return_X_y=True, as_dataframe=False)
        X2, _ = load_wisconsin_breast_cancer_data(return_X_y=True, as_dataframe=False)
        X3, _ = load_wine_data(return_X_y=True, as_dataframe=False)

        X1_scaled, _ = load_wireless_data(
            return_X_y=True, scaled=True, as_dataframe=False
        )
        X2_scaled, _ = load_wisconsin_breast_cancer_data(
            return_X_y=True, scaled=True, as_dataframe=False
        )
        X3_scaled, _ = load_wine_data(return_X_y=True, scaled=True, as_dataframe=False)
        self.assertTrue(
            np.allclose(X1 / np.linalg.norm(X1, axis=1, keepdims=True), X1_scaled)
        )
        self.assertTrue(
            np.allclose(X2 / np.linalg.norm(X2, axis=1, keepdims=True), X2_scaled)
        )
        self.assertTrue(
            np.allclose(X3 / np.linalg.norm(X3, axis=1, keepdims=True), X3_scaled)
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
