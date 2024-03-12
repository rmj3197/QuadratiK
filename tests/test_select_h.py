"""
Tests the select_h algorithm
"""
import numpy as np
from QuadratiK.kernel_test import select_h


def test_select_h_two_sample():
    np.random.seed(42)
    x = np.random.randn(200,2)   
    np.random.seed(56)
    y = np.random.randn(200,2)
    h_sel, all_powers = select_h(x=x,y=y,alternative = "skewness",random_state = 42)
    assert isinstance(h_sel, (int, float))
    assert all_powers['power'].max() > 0.5
    

def test_select_h_k_sample():
    np.random.seed(42)
    x = np.random.randn(100*3, 2)
    y = np.repeat(np.arange(1, 4), repeats=150)
    h_sel, all_powers = select_h(x=x,y=y,alternative = "location",random_state = 42)
    assert isinstance(h_sel, (int, float))
    assert all_powers['power'].max() > 0.5