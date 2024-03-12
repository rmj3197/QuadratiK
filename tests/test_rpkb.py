"""
Tests the rpkb method of the PKBD class by generating a random dataset
and checking if the dimensions of the output are correct.
"""
import numpy as np
from QuadratiK.spherical_clustering import PKBD


def test_rpkb():
    pkbd = PKBD()
    x_rejvmf = pkbd.rpkb(100, np.array([1, 0, 0]), 0.8, "rejvmf", random_state=42)
    x_rejacg = pkbd.rpkb(100, np.array([1, 0, 0]), 0.8, "rejacg", random_state=42)
    
    assert x_rejvmf.shape == (100, 3)
    assert x_rejacg.shape == (100, 3)
    assert isinstance(x_rejvmf, np.ndarray)
    assert isinstance(x_rejacg, np.ndarray)
    assert np.allclose(np.sum(x_rejvmf**2, axis=1), np.ones(100))
    assert np.allclose(np.sum(x_rejacg**2, axis=1), np.ones(100))
