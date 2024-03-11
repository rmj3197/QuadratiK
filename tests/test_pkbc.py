"""
Tests the functionality of the PKBC class 
generating data and checking the properties 
of the fitted clustering model.
"""

import unittest
import numpy as np
import pandas as pd
from QuadratiK.spherical_clustering import PKBC, PKBD


class TestPKBC(unittest.TestCase):
    def test_pkbc(self):
        pkbd = PKBD()
        x1 = pkbd.rpkb(100, np.array([1, 0, 0]), 0.8, "rejvmf", random_state=42)
        x2 = pkbd.rpkb(100, np.array([0, 1, 0]), 0.8, "rejacg", random_state=42)
        x3 = pkbd.rpkb(100, np.array([0, 1, 0.2]), 0.8, "rejacg", random_state=42)
        data = np.concatenate((x1, x2, x3), axis=0)
        pkbd_cluster_fit_numpy = PKBC(num_clust=3).fit(data)
        pkbd_cluster_fit_numpy_membership = PKBC(
            num_clust=3, stopping_rule="membership"
        ).fit(data)

        y_true = pd.DataFrame(np.repeat(np.arange(1, 4), repeats=100))

        self.assertEqual(len(pkbd_cluster_fit_numpy.labels_), 300)
        self.assertEqual(len(pkbd_cluster_fit_numpy.alpha_), 3)
        self.assertTrue(np.isclose(np.sum(pkbd_cluster_fit_numpy.alpha_), 1))
        self.assertEqual(len(pkbd_cluster_fit_numpy.rho_), 3)

        self.assertIsInstance(pkbd_cluster_fit_numpy.cosine_wcss_, (int, float))
        self.assertIsInstance(pkbd_cluster_fit_numpy.euclidean_wcss_, (int, float))

        self.assertIsInstance(pkbd_cluster_fit_numpy.stats(), pd.DataFrame)
        self.assertIsInstance(pkbd_cluster_fit_numpy.validation(), (int, float))
        self.assertIsInstance(pkbd_cluster_fit_numpy.validation(y_true=y_true), tuple)

        self.assertEqual(len(pkbd_cluster_fit_numpy_membership.labels_), 300)
        self.assertEqual(len(pkbd_cluster_fit_numpy_membership.alpha_), 3)
        self.assertTrue(np.isclose(np.sum(pkbd_cluster_fit_numpy_membership.alpha_), 1))
        self.assertEqual(len(pkbd_cluster_fit_numpy_membership.rho_), 3)
        self.assertIsInstance(pkbd_cluster_fit_numpy_membership.stats(), pd.DataFrame)

        self.assertIsInstance(
            pkbd_cluster_fit_numpy_membership.cosine_wcss_, (int, float)
        )
        self.assertIsInstance(
            pkbd_cluster_fit_numpy_membership.euclidean_wcss_, (int, float)
        )

        self.assertIsInstance(
            pkbd_cluster_fit_numpy_membership.validation(), (int, float)
        )
        self.assertIsInstance(
            pkbd_cluster_fit_numpy_membership.validation(y_true=y_true), tuple
        )

        with self.assertRaises(Exception):
            PKBC(num_clust=-1).fit(pd.DataFrame(data))

        with self.assertRaises(Exception):
            PKBC(num_clust=3, max_iter=-1).fit(data)

        with self.assertRaises(Exception):
            PKBC(num_clust=3, init_method="some").fit(data)

        with self.assertRaises(Exception):
            PKBC(num_clust=3, num_init=-1).fit(data)

        with self.assertRaises(Exception):
            PKBC(num_clust=3, random_state=[42]).fit(data)

        with self.assertRaises(ValueError):
            PKBC(num_clust=310).fit(data)

        with self.assertRaises(Exception):
            PKBC(num_clust=3, stopping_rule="some").fit(data)
