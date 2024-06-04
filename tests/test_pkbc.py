"""
Tests the functionality of the PKBC class 
generating data and checking the properties 
of the fitted clustering model.
"""

import unittest
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from QuadratiK.spherical_clustering import PKBC, PKBD


class TestPKBC(unittest.TestCase):
    def test_pkbc(self):
        pkbd = PKBD()
        x1 = pkbd.rpkb(100, np.array([1, 0, 0]), 0.8, "rejvmf", random_state=42)
        x2 = pkbd.rpkb(100, np.array([0, 1, 0]), 0.8, "rejacg", random_state=42)
        x3 = pkbd.rpkb(100, np.array([0, 1, 0.2]), 0.8, "rejacg", random_state=42)
        data = np.concatenate((x1, x2, x3), axis=0)
        prediction_data = data[:50]
        pkbd_cluster_fit_numpy = PKBC(num_clust=3, random_state=42).fit(data)
        pkbd_cluster_fit_numpy_membership = PKBC(
            num_clust=3,
            stopping_rule="membership",
            random_state=42,
        ).fit(data)

        y_true = pd.DataFrame(np.repeat(np.arange(1, 4), repeats=100))

        self.assertEqual(len(pkbd_cluster_fit_numpy.labels_[3]), 300)
        self.assertEqual(len(pkbd_cluster_fit_numpy.alpha_[3]), 3)
        self.assertTrue(np.isclose(np.sum(pkbd_cluster_fit_numpy.alpha_[3]), 1))
        self.assertEqual(len(pkbd_cluster_fit_numpy.rho_[3]), 3)

        self.assertIsInstance(pkbd_cluster_fit_numpy.cosine_wcss_[3], (int, float))
        self.assertIsInstance(pkbd_cluster_fit_numpy.euclidean_wcss_[3], (int, float))

        self.assertIsInstance(pkbd_cluster_fit_numpy.stats_clusters(3), pd.DataFrame)
        self.assertIsInstance(pkbd_cluster_fit_numpy.validation(), tuple)
        self.assertIsInstance(pkbd_cluster_fit_numpy.validation(y_true=y_true), tuple)
        self.assertIsInstance(pkbd_cluster_fit_numpy.predict(prediction_data, 3), tuple)

        self.assertEqual(len(pkbd_cluster_fit_numpy_membership.labels_[3]), 300)
        self.assertEqual(len(pkbd_cluster_fit_numpy_membership.alpha_[3]), 3)
        self.assertTrue(
            np.isclose(np.sum(pkbd_cluster_fit_numpy_membership.alpha_[3]), 1)
        )
        self.assertEqual(len(pkbd_cluster_fit_numpy_membership.rho_[3]), 3)
        self.assertIsInstance(
            pkbd_cluster_fit_numpy_membership.stats_clusters(3), pd.DataFrame
        )
        self.assertIsInstance(
            pkbd_cluster_fit_numpy_membership.predict(prediction_data, 3), tuple
        )

        self.assertIsInstance(
            pkbd_cluster_fit_numpy_membership.cosine_wcss_[3], (int, float)
        )
        self.assertIsInstance(
            pkbd_cluster_fit_numpy_membership.euclidean_wcss_[3], (int, float)
        )

        self.assertIsInstance(pkbd_cluster_fit_numpy_membership.validation(), tuple)
        self.assertIsInstance(
            pkbd_cluster_fit_numpy_membership.validation(y_true=y_true), tuple
        )

        self.assertIsInstance(pkbd_cluster_fit_numpy.plot(3), type(go.Figure()))
        self.assertIsInstance(pkbd_cluster_fit_numpy.plot(3, y_true), type(go.Figure()))
        self.assertIsInstance(pkbd_cluster_fit_numpy.summary(), str)

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

        with self.assertRaises(Exception):
            PKBC(num_clust=pd.DataFrame(3)).fit(data)

        with self.assertRaises(ValueError):
            X = np.random.randn(10, 2)
            pkbd_cluster_fit_numpy.predict(X, 3)

        pkbd = PKBD()
        x1 = pkbd.rpkb(100, np.array([1, 0]), 0.8, "rejvmf", random_state=42)
        x2 = pkbd.rpkb(100, np.array([0, 1]), 0.8, "rejacg", random_state=42)
        data_two = np.concatenate((x1, x2), axis=0)
        pkbd_cluster_two_dim = PKBC(num_clust=3, random_state=42).fit(data_two)
        y_true = pd.DataFrame(np.repeat(np.arange(1, 3), repeats=100))
        self.assertIsInstance(pkbd_cluster_two_dim.plot(3), type(plt.figure()))
        self.assertIsInstance(pkbd_cluster_two_dim.plot(3, y_true), type(plt.figure()))
