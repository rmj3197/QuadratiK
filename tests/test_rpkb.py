"""
Tests the PKBD class
"""

import unittest
import numpy as np
import pandas as pd
from QuadratiK.spherical_clustering import PKBD


class TestPKBD(unittest.TestCase):

    def test_rpkb(self):
        pkbd = PKBD()
        x_rejvmf = pkbd.rpkb(100, [1, 0, 0], 0.8, "rejvmf", random_state=42)
        x_rejacg = pkbd.rpkb(100, [1, 0, 0], 0.8, "rejacg", random_state=42)

        self.assertEqual(x_rejvmf.shape, (100, 3))
        self.assertEqual(x_rejacg.shape, (100, 3))
        self.assertIsInstance(x_rejvmf, np.ndarray)
        self.assertIsInstance(x_rejacg, np.ndarray)
        self.assertTrue(np.allclose(np.sum(x_rejvmf**2, axis=1), np.ones(100)))
        self.assertTrue(np.allclose(np.sum(x_rejacg**2, axis=1), np.ones(100)))

        with self.assertRaises(ValueError):
            pkbd.rpkb(100, np.array([1, 0, 0]), -1, "rejvmf", random_state=42)

        with self.assertRaises(ValueError):
            pkbd.rpkb(100, np.array([0, 0, 0]), 0.1, "rejvmf", random_state=42)

        with self.assertRaises(ValueError):
            pkbd.rpkb(100, np.array([1, 0, 0]), 0.1, "some method", random_state=42)

        with self.assertRaises(ValueError):
            pkbd.rpkb(100, np.array([1, 0, 0]), 0.1, "rejvmf", random_state=[42])

        with self.assertRaises(NotImplementedError):
            pkbd.rpkb(100, np.array([1, 0, 0]), 0.1, "rejpsaw", random_state=42)

        with self.assertRaises(ValueError):
            pkbd.rpkb(-100, np.array([1, 0, 0]), 0.1, "rejvmf", random_state=42)

    def test_dpkb(self):
        pkbd = PKBD()
        x = pkbd.rpkb(n=100, mu=np.array([1, 1, 1]), rho=0.9)
        density_x = pkbd.dpkb(
            x=pd.DataFrame(x), mu=np.array([1, 1, 1]), rho=0.9, logdens=True
        )
        density_y = pkbd.dpkb(x=pd.DataFrame(x), mu=np.array([1, 1, 1]), rho=0.9)

        self.assertEqual(density_x.shape[0], 100, "Dimensions are equal")
        self.assertEqual(density_y.shape[0], 100, "Dimensions are equal")

        with self.assertRaises(ValueError):
            pkbd.dpkb(x=x, mu=np.array([1]), rho=0.9)

        with self.assertRaises(ValueError):
            x_data = np.array([[1, 1, 1, 1, 1]]).reshape(-1, 1)
            pkbd.dpkb(x=x_data, mu=np.array([1, 1]), rho=0.9)

        with self.assertRaises(ValueError):
            pkbd.dpkb(x=x[:, 1], mu=np.array([1, 1]), rho=0.9)

        with self.assertRaises(ValueError):
            pkbd.dpkb(x=x, mu=np.array([1, 1]), rho=0.9)

        with self.assertRaises(ValueError):
            pkbd.dpkb(x=x, mu=np.array([1, 1, 1]), rho=-0.9)

        with self.assertRaises(ValueError):
            pkbd.dpkb(x=x, mu=np.array([0, 0, 0]), rho=0.9)
