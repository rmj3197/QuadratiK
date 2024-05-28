import numpy as np
import pandas as pd
import unittest
from QuadratiK.kernel_test import KernelTest


class TestKernelTest(unittest.TestCase):

    def setUp(self):
        np.random.seed(81423)

    def test_datastructure(self):
        X1 = [1, 2, 3, 4]
        X2 = np.random.randn(100, 1)
        y = [1, 1, 2, 2]
        with self.assertRaises(TypeError):
            KernelTest(h=0.5, random_state=42).test(X1)
        with self.assertRaises(TypeError):
            KernelTest(h=0.5, random_state=42).test(X2, y)

    def test_normality(self):
        x = np.random.randn(100, 2)
        dataframe_x = pd.DataFrame(x)
        normality_test_numpy = KernelTest(h=0.5, random_state=42).test(x)
        normality_test_dataframe = KernelTest(h=0.5, random_state=42).test(dataframe_x)
        self.assertFalse(normality_test_numpy.un_h0_rejected_)
        self.assertTrue(
            isinstance(normality_test_numpy.un_test_statistic_, (int, float))
        )
        self.assertFalse(normality_test_dataframe.un_h0_rejected_)
        self.assertTrue(
            isinstance(normality_test_dataframe.un_test_statistic_, (int, float))
        )
        self.assertEqual(
            normality_test_numpy.un_h0_rejected_,
            normality_test_dataframe.un_h0_rejected_,
        )
        self.assertEqual(
            normality_test_numpy.un_test_statistic_,
            normality_test_dataframe.un_test_statistic_,
        )

        with self.assertRaises(ValueError):
            KernelTest(h=0.5, random_state=42, centering_type="not supported").test(x)

        x_1d = np.random.randn(100 * 3)
        normality_test_1d = KernelTest(
            method="subsampling", b=0.5, random_state=42, alternative="scale"
        ).test(x_1d)
        self.assertFalse(normality_test_1d.un_h0_rejected_)
        self.assertTrue(isinstance(normality_test_1d.un_test_statistic_, (int, float)))

    def test_twosample(self):
        x = np.random.randn(100, 2)
        y = np.random.randn(100, 2) + 2
        y_1 = np.random.randn(100, 1) + 2

        two_sample_test_subsampling = KernelTest(
            h=1.5, method="subsampling", b=0.5, random_state=42
        ).test(x, y)
        self.assertTrue(np.all(two_sample_test_subsampling.dn_h0_rejected_))
        self.assertTrue(
            isinstance(two_sample_test_subsampling.dn_test_statistic_, (int, float))
        )

        with self.assertRaises(ValueError):
            KernelTest(h=0.5, random_state=42, method="a different method").test(x, y)
        with self.assertRaises(ValueError):
            KernelTest(h=0.5, random_state=42, b=2).test(x, y)
        with self.assertRaises(ValueError):
            KernelTest(h=None, alternative=None, random_state=42).test(x, y)
        with self.assertRaises(ValueError):
            KernelTest(h=0.5, random_state=42, centering_type="not in list").test(x, y)
        with self.assertRaises(ValueError):
            KernelTest(h=1.5, method="subsampling", b=0.5, random_state=42).test(x, y_1)

        two_sample_test_h_selection = KernelTest(
            h=None,
            alternative="location",
            method="subsampling",
            b=0.5,
            centering_type="param",
            random_state=42,
        ).test(x, y)
        self.assertTrue(np.all(two_sample_test_h_selection.dn_h0_rejected_))
        self.assertTrue(
            isinstance(two_sample_test_h_selection.dn_test_statistic_, (int, float))
        )

    def test_ksample(self):
        x = np.random.randn(100 * 3, 2)
        x_1d = np.random.randn(100 * 3)
        y = np.repeat(np.arange(1, 4), repeats=100)
        y_dataframe = pd.DataFrame(np.repeat(np.arange(1, 4), repeats=100))
        y_diff_len = np.repeat(np.arange(1, 4), repeats=50)

        k_sample_test = KernelTest(
            h=1.5, method="subsampling", b=0.5, random_state=42
        ).test(x, y)
        self.assertFalse(np.all(k_sample_test.dn_h0_rejected_))
        self.assertTrue(isinstance(k_sample_test.dn_test_statistic_, (int, float)))

        k_sample_test_without_h = KernelTest(
            method="subsampling", b=0.5, random_state=42, alternative="location"
        ).test(x, y)
        self.assertFalse(np.all(k_sample_test_without_h.dn_h0_rejected_))
        self.assertTrue(
            isinstance(k_sample_test_without_h.dn_test_statistic_, (int, float))
        )

        k_sample_test_1d = KernelTest(
            h=1.5, method="subsampling", b=0.5, random_state=42
        ).test(x_1d, y_dataframe)
        self.assertFalse(np.all(k_sample_test_1d.dn_h0_rejected_))
        self.assertTrue(isinstance(k_sample_test_1d.dn_test_statistic_, (int, float)))

        with self.assertRaises(ValueError):
            KernelTest(h=1.5, method="subsampling", b=0.5, random_state=42).test(
                x, y_diff_len
            )
        with self.assertRaises(ValueError):
            KernelTest(h=1.5, method="subsampling", b=0.5, random_state=[42]).test(
                x_1d, y
            )

    def test_stats(self):
        X1 = np.random.randn(100 * 3, 2)
        Y1 = np.repeat(np.arange(1, 4), repeats=100)

        X2 = np.random.randn(100, 2)
        Y2 = np.random.randn(100, 2) + 2

        two_sample_test_perm_random_state = KernelTest(
            h=1.5, method="permutation", random_state=42
        ).test(X2, Y2)
        two_sample_test_boot_random_state = KernelTest(
            h=1.5, method="bootstrap", random_state=42
        ).test(X2, Y2)

        k_sample_test_sub = KernelTest(h=1.5, method="subsampling", b=0.5).test(X1, Y1)
        k_sample_test_perm = KernelTest(h=1.5, method="permutation").test(X1, Y1)
        k_sample_test_boot = KernelTest(h=1.5, method="bootstrap").test(X1, Y1)
        k_sample_test_perm_random_state = KernelTest(
            h=1.5, method="permutation", random_state=42
        ).test(X1, Y1)
        k_sample_test_boot_random_state = KernelTest(
            h=1.5, method="bootstrap", random_state=42
        ).test(X1, Y1)

        two_sample_test_sub = KernelTest(
            h=1.5, method="subsampling", b=0.5, random_state=42
        ).test(X2, Y2)
        two_sample_test_perm = KernelTest(h=1.5, method="permutation").test(X2, Y2)
        two_sample_test_boot = KernelTest(h=1.5, method="bootstrap").test(X2, Y2)

        self.assertTrue(isinstance(k_sample_test_sub.stats(), pd.DataFrame))
        self.assertTrue(isinstance(k_sample_test_perm.stats(), pd.DataFrame))
        self.assertTrue(isinstance(k_sample_test_boot.stats(), pd.DataFrame))
        self.assertTrue(
            isinstance(k_sample_test_perm_random_state.stats(), pd.DataFrame)
        )
        self.assertTrue(
            isinstance(k_sample_test_boot_random_state.stats(), pd.DataFrame)
        )
        self.assertTrue(isinstance(two_sample_test_sub.stats(), pd.DataFrame))
        self.assertTrue(isinstance(two_sample_test_perm.stats(), pd.DataFrame))
        self.assertTrue(isinstance(two_sample_test_boot.stats(), pd.DataFrame))
        self.assertTrue(
            isinstance(two_sample_test_perm_random_state.stats(), pd.DataFrame)
        )
        self.assertTrue(
            isinstance(two_sample_test_boot_random_state.stats(), pd.DataFrame)
        )

    def test_summary(self):
        x = np.random.randn(100 * 3, 2)
        y = np.repeat(np.arange(1, 4), repeats=100)
        k_sample_test = KernelTest(
            h=1.5, method="subsampling", b=0.5, random_state=42
        ).test(x, y)
        self.assertTrue(isinstance(k_sample_test.summary(), str))
        self.assertTrue(isinstance(k_sample_test.summary(print_fmt="html"), str))
