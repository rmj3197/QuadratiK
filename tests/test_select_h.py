import unittest
import numpy as np
from QuadratiK.kernel_test import select_h


class TestSelectH(unittest.TestCase):
    def test_select_h_one_sample_skewness(self):
        np.random.seed(42)
        x = np.random.randn(200)
        h_sel, all_powers, plot = select_h(
            x=x, alternative="skewness", random_state=42, power_plot=True
        )
        self.assertIsInstance(h_sel, (int, float))
        self.assertIsNotNone(plot)

    def test_select_h_one_sample_location(self):
        np.random.seed(42)
        x = np.random.randn(200, 1)
        h_sel, all_powers, plot = select_h(
            x=x, alternative="location", random_state=42, power_plot=True
        )
        self.assertIsInstance(h_sel, (int, float))
        self.assertIsNotNone(plot)

    def test_select_h_one_sample_scale(self):
        np.random.seed(42)
        x = np.random.randn(200, 2)
        h_sel, all_powers, plot = select_h(
            x=x, alternative="scale", random_state=42, power_plot=True
        )
        self.assertIsInstance(h_sel, (int, float))
        self.assertIsNotNone(plot)

    def test_select_h_two_sample_skewness(self):
        np.random.seed(42)
        x = np.random.randn(200, 2)
        np.random.seed(56)
        y = np.random.randn(200, 2)
        h_sel, all_powers, plot = select_h(
            x=x, y=y, alternative="skewness", random_state=42, power_plot=True
        )
        self.assertIsInstance(h_sel, (int, float))
        self.assertIsNotNone(plot)

    def test_select_h_two_sample_location(self):
        np.random.seed(42)
        x = np.random.randn(200, 2)
        np.random.seed(56)
        y = np.random.randn(200, 2)
        h_sel, all_powers, plot = select_h(
            x=x, y=y, alternative="location", random_state=42, power_plot=True
        )
        self.assertIsInstance(h_sel, (int, float))
        self.assertIsNotNone(plot)

    def test_select_h_two_sample_scale(self):
        np.random.seed(42)
        x = np.random.randn(200, 2)
        np.random.seed(56)
        y = np.random.randn(200, 2)
        h_sel, all_powers, plot = select_h(
            x=x, y=y, alternative="scale", random_state=42, power_plot=True
        )
        self.assertIsInstance(h_sel, (int, float))
        self.assertIsNotNone(plot)

    def test_select_h_k_sample_location(self):
        np.random.seed(42)
        x = np.random.randn(100 * 3, 2)
        y = np.repeat(np.arange(1, 4), repeats=100)
        h_sel, all_powers = select_h(x=x, y=y, alternative="location", random_state=42)
        self.assertIsInstance(h_sel, (int, float))

    def test_select_h_k_sample_skewness(self):
        np.random.seed(42)
        x = np.random.randn(100 * 3, 2)
        y = np.repeat(np.arange(1, 4), repeats=100)
        h_sel, all_powers = select_h(x=x, y=y, alternative="skewness", random_state=42)
        self.assertIsInstance(h_sel, (int, float))

    def test_select_h_k_sample_scale(self):
        np.random.seed(42)
        x = np.random.randn(100 * 3, 2)
        y = np.repeat(np.arange(1, 4), repeats=100)
        h_sel, all_powers = select_h(x=x, y=y, alternative="scale", random_state=42)
        self.assertIsInstance(h_sel, (int, float))
