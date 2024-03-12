"""
Tests the uniformity of a set of points sampled from a
hypersphere using the Poisson Kernel Test.
"""
import numpy as np
from QuadratiK.poisson_kernel_test import PoissonKernelTest
from QuadratiK.tools import sample_hypersphere


def test_uniformity_test():
    x_sp = sample_hypersphere(npoints=100, ndim=3, random_state=42)
    uniformity_test = PoissonKernelTest(rho=0.8, random_state=42).test(x_sp)
    assert uniformity_test.u_statistic_un_,(int, float)
    assert uniformity_test.v_statistic_vn_,(int, float)
