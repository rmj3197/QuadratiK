"""
Testing for Normality, Two Sample and K-Sample Tests
"""
import numpy as np
from QuadratiK.kernel_test import KernelTest


def test_normality():
    np.random.seed(81423)
    x = np.random.randn(100, 2)
    normality_test = KernelTest(h=0.5, random_state=42).test(x)
    assert not normality_test.h0_rejected_
    assert isinstance(normality_test.test_statistic_, (int, float))


def test_twosample():
    np.random.seed(81423)
    x = np.random.randn(100, 2)
    y = np.random.randn(100, 2) + 2
    two_sample_test = KernelTest(
        h=1.5, method="subsampling", b=0.5, random_state=42).test(x, y)
    assert two_sample_test.h0_rejected_
    assert isinstance(two_sample_test.test_statistic_, (int, float))


def test_ksample():
    np.random.seed(81423)
    x = np.random.randn(100*3, 2)
    y = np.repeat(np.arange(1, 4), repeats=100)
    k_sample_test = KernelTest(
        h=1.5, method="subsampling", b=0.5, random_state=42).test(x, y)
    assert not k_sample_test.h0_rejected_
    assert isinstance(k_sample_test.test_statistic_, np.ndarray)
