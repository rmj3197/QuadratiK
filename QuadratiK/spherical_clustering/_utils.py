"""
Contains utility functions for PKBC and PKBD.
"""

import numpy as np


def root_func(x, num_data, alpha_current_h, num_var, mu_denom_h, sum_h_weight_mat):
    """
    Function used for determing the root, used in PKBC.
    """
    return (
        (-2 * num_data * x * alpha_current_h) / (1 - x**2 + 10 ** (-15))
        + num_var * mu_denom_h
        - num_var * x * sum_h_weight_mat
    )


def c_d_lambda(beta, p, lamda):
    """
    Function used for determining root, used in PKBD.
    """
    return (
        -4 * (p - 1) * beta**3
        + (4 * p - lamda**2 * (p - 2) ** 2) * beta**2
        + 2 * p * (p - 2) * lamda**2 * beta
        - p**2 * lamda**2
    )


def calculate_wcss_euclidean(k, memb_best, dat, mu_best):
    """
    Used for computing the euclidean WCSS for a single cluster.
    """
    idx = np.where(memb_best == k)[0]
    return np.sum(np.linalg.norm(dat[idx] - mu_best[k], axis=1) ** 2)


def calculate_wcss_cosine(k, memb_best, dat, mu_best):
    """
    Used for computing the cosine WCSS for a single cluster.
    """
    idx = np.where(memb_best == k)[0]
    return np.sum(dat[idx] * mu_best[k])
