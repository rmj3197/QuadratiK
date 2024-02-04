
"""
Tests the functionality of the PKBC class 
generating data and checking the properties 
of the fitted clustering model.
"""
import numpy as np
from QuadratiK.spherical_clustering import PKBC, PKBD


def test_pkbc():
    pkbd = PKBD()
    x1 = pkbd.rpkb(100, np.array([1, 0, 0]), 0.8, "rejvmf", random_state=42)
    x2 = pkbd.rpkb(100, np.array([0, 1, 0]), 0.8, "rejacg", random_state=42)
    data = np.concatenate((x1, x2), axis=1)

    pkbd_cluster_fit = PKBC(num_clust=3).fit(data)

    assert len(pkbd_cluster_fit.labels_) == 100
    assert len(pkbd_cluster_fit.alpha_) == 3
    assert np.isclose(np.sum(pkbd_cluster_fit.alpha_), 1)
    assert len(pkbd_cluster_fit.rho_) == 3
