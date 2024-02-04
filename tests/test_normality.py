
"""
Testing for normality test.
"""

import numpy as np
from QuadratiK.kernel_test import KernelTest


def test_normality():
    np.random.seed(42)
    x = np.random.randn(100, 2)
    normality_test = KernelTest(h=0.5).test(x)
    assert not normality_test.h0_rejected_
