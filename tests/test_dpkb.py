"""
Tests the dpkb method of the PKBD class 
by generating a random dataset and 
calculating the density of each point.
"""
import numpy as np
from QuadratiK.spherical_clustering import PKBD


def test_dpkb():
    pkbd = PKBD()
    x = pkbd.rpkb(n=100, mu=np.array([1, 1, 1]), rho=0.9)
    density_x = pkbd.dpkb(x=x, mu=np.array([1, 1, 1]), rho=0.9)
    assert density_x.shape[0] == 100, "Dimensions are equal"
