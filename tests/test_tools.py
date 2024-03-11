"""
Tests the tools module
"""

import unittest
import numpy as np
import pandas as pd
from QuadratiK.tools import (
    stats,
    sample_hypersphere,
    qq_plot,
    sphere3d,
    plot_clusters_2d,
)


class TestTools(unittest.TestCase):
    def test_stats(self):
        X = np.random.randn(300, 2)
        Y = np.random.randn(300, 2)
        self.assertIsInstance(stats(X, Y), pd.DataFrame)

    def test_sample_hypersphere(self):
        with self.assertRaises(ValueError):
            sample_hypersphere(100, 3, [42])

    def test_qq_plot(self):
        X = pd.DataFrame(np.random.randn(100, 2))
        Y = pd.DataFrame(np.random.randn(100, 2))
        self.assertIsNotNone(qq_plot(X))
        self.assertIsNotNone(qq_plot(X, Y))

    def test_sphere_3d(self):
        X = pd.DataFrame(np.random.randn(300, 3))
        y1 = pd.DataFrame(np.repeat(np.arange(1, 4), repeats=100))
        y2 = pd.DataFrame(np.repeat(np.arange(1, 4), repeats=100))[0]
        y3 = np.repeat(np.arange(1, 4), repeats=100)
        y4 = np.repeat(np.arange(1, 4), repeats=100).reshape(1, -1)
        self.assertIsNotNone(sphere3d(X))
        self.assertIsNotNone(sphere3d(X, y1))
        self.assertIsNotNone(sphere3d(X, y2))
        self.assertIsNotNone(sphere3d(X, y3))
        self.assertIsNotNone(sphere3d(X, y4))

    def test_plot_clusters_2d(self):
        X = np.random.randn(300, 2)
        X = pd.DataFrame(X / np.linalg.norm(X, axis=1, keepdims=True))
        y1 = pd.DataFrame(np.repeat(np.arange(1, 4), repeats=100))
        y2 = pd.DataFrame(np.repeat(np.arange(1, 4), repeats=100))[0]
        y3 = np.repeat(np.arange(1, 4), repeats=100)
        y4 = np.repeat(np.arange(1, 4), repeats=100).reshape(1, -1)
        self.assertIsNotNone(plot_clusters_2d(X))
        self.assertIsNotNone(plot_clusters_2d(X, y1))
        self.assertIsNotNone(plot_clusters_2d(X, y2))
        self.assertIsNotNone(plot_clusters_2d(X, y3))
        self.assertIsNotNone(plot_clusters_2d(X, y4))
