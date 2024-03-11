"""
Tests the uniformity of a set of points sampled from a
hypersphere using the Poisson Kernel Test.
"""

import unittest
import numpy as np
import pandas as pd
from QuadratiK.poisson_kernel_test import PoissonKernelTest
from QuadratiK.tools import sample_hypersphere


class TestUniformityTest(unittest.TestCase):
    def test_uniformity_test(self):
        dat = np.random.randn(100)
        x_sp = dat / np.linalg.norm(dat)
        uniformity_test = PoissonKernelTest(rho=0.8, random_state=42).test(x_sp)
        self.assertIsInstance(uniformity_test.u_statistic_un_, (int, float))
        self.assertIsInstance(uniformity_test.u_statistic_cv_, (int, float))
        self.assertIsInstance(uniformity_test.u_statistic_h0_, np.bool_)
        self.assertIsInstance(uniformity_test.v_statistic_vn_, (int, float))

        self.assertIsInstance(uniformity_test.stats(), pd.DataFrame)
        self.assertIsInstance(uniformity_test.summary(print_fmt="html"), str)
        self.assertIsInstance(uniformity_test.summary(), str)

        with self.assertRaises(ValueError):
            PoissonKernelTest(rho=0.8, quantile=2).test(x_sp)

        with self.assertRaises(ValueError):
            PoissonKernelTest(rho=0.8, random_state=[42]).test(x_sp)

        with self.assertRaises(ValueError):
            PoissonKernelTest(rho=2, random_state=42).test(x_sp)

        with self.assertRaises(TypeError):
            x = [1, 2, 3, 4, 5]
            PoissonKernelTest(rho=0.8, random_state=42).test(x)

    def test_uniformity_no_rndm_state_dataframe(self):
        x_sp = pd.DataFrame(sample_hypersphere(npoints=100, ndim=3))
        uniformity_test = PoissonKernelTest(
            rho=0.8, random_state=None, num_iter=10, n_jobs=4
        ).test(x_sp)
        self.assertIsInstance(uniformity_test.u_statistic_un_, (int, float))
        self.assertIsInstance(uniformity_test.v_statistic_vn_, (int, float))
        self.assertIsInstance(uniformity_test.u_statistic_cv_, (int, float))
        self.assertIsInstance(uniformity_test.v_statistic_h0_, np.bool_)
