import unittest
import numpy as np
import pandas as pd
from QuadratiK.kernel_test import select_h


class TestSelectH(unittest.TestCase):
    def test_select_h_one_sample_skewness(self):
        np.random.seed(42)
        x = np.random.randn(200)
        h_sel, all_powers, plot = select_h(
            x=x, alternative="skewness", random_state=42, power_plot=True
        )
        self.assertIsInstance(h_sel, (int, float))
        self.assertIsInstance(all_powers, pd.DataFrame)
        self.assertIsNotNone(plot)

    def test_select_h_one_sample_location(self):
        np.random.seed(42)
        x = np.random.randn(200, 1)
        h_sel, all_powers, plot = select_h(
            x=x, alternative="location", random_state=42, power_plot=True
        )
        self.assertIsInstance(h_sel, (int, float))
        self.assertIsInstance(all_powers, pd.DataFrame)
        self.assertIsNotNone(plot)

    def test_select_h_one_sample_scale(self):
        np.random.seed(42)
        x = np.random.randn(200, 2)
        h_sel, all_powers, plot = select_h(
            x=x, alternative="scale", random_state=42, power_plot=True
        )
        self.assertIsInstance(h_sel, (int, float))
        self.assertIsInstance(all_powers, pd.DataFrame)
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
        self.assertIsInstance(all_powers, pd.DataFrame)
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
        self.assertIsInstance(all_powers, pd.DataFrame)
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
        self.assertIsInstance(all_powers, pd.DataFrame)
        self.assertIsNotNone(plot)

    def test_select_h_k_sample_location(self):
        np.random.seed(42)
        x = np.random.randn(100 * 3, 2)
        y = np.repeat(np.arange(1, 4), repeats=100)
        h_sel, all_powers, plot = select_h(
            x=x, y=y, alternative="location", random_state=42, power_plot=True
        )
        self.assertIsInstance(all_powers, pd.DataFrame)
        self.assertIsInstance(h_sel, (int, float))
        self.assertIsNotNone(plot)

    def test_select_h_k_sample_skewness(self):
        np.random.seed(42)
        x = np.random.randn(100 * 3, 2)
        y = np.repeat(np.arange(1, 4), repeats=100)
        h_sel, all_powers, plot = select_h(
            x=x, y=y, alternative="skewness", random_state=42, power_plot=True
        )
        self.assertIsInstance(all_powers, pd.DataFrame)
        self.assertIsInstance(h_sel, (int, float))
        self.assertIsNotNone(plot)

    def test_select_h_k_sample_scale(self):
        np.random.seed(42)
        x = np.random.randn(100 * 3, 3)
        y = np.repeat(np.arange(1, 4), repeats=100)
        h_sel, all_powers, plot = select_h(
            x=x, y=y, alternative="scale", random_state=42, power_plot=True
        )
        self.assertIsInstance(all_powers, pd.DataFrame)
        self.assertIsInstance(h_sel, (int, float))
        self.assertIsNotNone(plot)

    def test_select_h_inputs(self):
        x = pd.DataFrame(np.random.randn(100 * 3, 1))
        y = pd.DataFrame(np.repeat(np.arange(1, 4), repeats=100))

        with self.assertRaises(ValueError):
            select_h(x=x, y=y, alternative="scale", random_state=[42], power_plot=True)

        with self.assertRaises(TypeError):
            x_list = [1, 2, 3, 4]
            y_list = [1, 1, 2, 2]
            select_h(
                x=x_list,
                y=y_list,
                alternative="scale",
                random_state=42,
                power_plot=True,
            )

        with self.assertRaises(TypeError):
            x_df = pd.DataFrame(np.random.randn(100, 2))
            y_int = -1
            select_h(
                x=x_df,
                y=y_int,
                alternative="scale",
                random_state=42,
                power_plot=True,
            )

        with self.assertRaises(ValueError):
            select_h(x=x, y=y, alternative="some", random_state=42, power_plot=True)

        with self.assertRaises(ValueError):
            select_h(
                x=x,
                y=y,
                alternative="scale",
                random_state=42,
                power_plot=True,
                delta_dim=[1, 1],
            )

        with self.assertRaises(ValueError):
            x = np.random.randn(100 * 3, 1)
            select_h(
                x=x,
                y=None,
                alternative="scale",
                random_state=42,
                power_plot=True,
                delta_dim=[1, 1],
            )

        with self.assertRaises(ValueError):
            x_two_sample = pd.DataFrame(np.random.randn(100 * 3, 5))
            y_two_sample = pd.DataFrame(np.random.randn(100 * 3, 3))
            select_h(
                x=x_two_sample,
                y=y_two_sample,
                alternative="scale",
                random_state=42,
                power_plot=True,
            )
        with self.assertRaises(ValueError):
            x_two_sample = pd.DataFrame(np.random.randn(100 * 3, 3))
            y_two_sample = pd.DataFrame(np.random.randn(100 * 3, 3))
            select_h(
                x=x_two_sample,
                y=y_two_sample,
                alternative="scale",
                random_state=42,
                power_plot=True,
                delta_dim=[1, 1],
            )
