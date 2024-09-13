"""
Tests the rpkb random sampling with wrapped cauchy. 
"""

import numpy as np
from scipy.stats import wrapcauchy

from QuadratiK.spherical_clustering import PKBD


class TestApproximation:
    def test(self):
        n = 10000
        loc = np.array([-1, 0])
        rho = 0.6

        wrapped_cauchy = wrapcauchy.rvs(rho, loc=-np.pi, size=n)
        pkbd = PKBD().rpkb(n=10000, mu=loc, rho=rho)
        pkbd_angles = np.arctan2(pkbd[:, 1], pkbd[:, 0])

        quantiles = np.arange(0.05, 1, 0.05)

        wc_q = np.quantile(wrapped_cauchy, quantiles)
        pkbd_q = np.quantile(pkbd_angles, quantiles)

        np.testing.assert_allclose(wc_q, pkbd_q, atol=0.1)
