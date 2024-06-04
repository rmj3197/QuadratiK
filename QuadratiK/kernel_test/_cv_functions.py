"""
Critical value functions for Kernel Test
"""

from sklearn.utils.parallel import Parallel, delayed
import numpy as np
from ._utils import (
    normal_cv_helper,
    bootstrap_helper_twosample,
    permutation_helper_twosample,
    subsampling_helper_twosample,
    bootstrap_helper_ksample,
    subsampling_helper_ksample,
    permutation_helper_ksample,
)


def cv_twosample(
    num_iter,
    quantile,
    data_pool,
    size_x,
    size_y,
    h,
    method,
    b=0.9,
    random_state=None,
    n_jobs=8,
):
    """
    This function computes the critical value for two-sample kernel tests with
    centered Gaussian kernel using one of three methods: bootstrap, permutation, or subsampling.

    Parameters
    ----------
        num_iter : int
            The number of iterations to use for critical value estimation.

        quantile : float
            The quantile to use for critical value estimation.

        data_pool: numpy.ndarray
            ndarray containing the data to be used in the test.

        size_x : int
            The number of rows in the data_pool corresponding to group X.

        size_y : int
            The number of rows in the data_pool corresponding to group Y.

        h : float
            The tuning parameter for the kernel test.

        method : str
            Method to use for computing the critical value
            (one of bootstrap, permutation, or subsampling).

        b : float, optional
            Subsampling block size (only used if method is subsampling).

        random_state : int, None, optional.
            Seed for random number generation. Defaults to None.

        n_jobs : int, optional
            n_jobs specifies the maximum number of concurrently
            running workers. If 1 is given, no joblib parallelism
            is used at all, which is useful for debugging. For more
            information on joblib n_jobs refer to -
            https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html.

    Returns
    -------
        critical value : float
            Critical value for the specified dimension, size and quantile.

    References
    -----------
        Markatou Marianthi, Saraceno Giovanni, Chen Yang (2023). “Two- and k-Sample Tests Based on
        Quadratic Distances.” Manuscript, (Department of Biostatistics, University at Buffalo).

    """
    if method == "bootstrap":
        results = Parallel(n_jobs=n_jobs)(
            delayed(bootstrap_helper_twosample)(
                size_x, size_y, h, data_pool, i, random_state
            )
            for i in range(num_iter)
        )
    elif method == "permutation":
        results = Parallel(n_jobs=n_jobs)(
            delayed(permutation_helper_twosample)(
                size_x, size_y, h, data_pool, i, random_state
            )
            for i in range(num_iter)
        )
    elif method == "subsampling":
        results = Parallel(n_jobs=n_jobs)(
            delayed(subsampling_helper_twosample)(
                size_x, size_y, b, h, data_pool, i, random_state
            )
            for i in range(num_iter)
        )
    cv = np.quantile(results, quantile, axis=0)
    return cv


def cv_normality(
    size,
    h,
    mu_hat,
    sigma_hat,
    num_iter=500,
    quantile=0.95,
    random_state=None,
    n_jobs=8,
):
    """
    This function computes the empirical critical value for the
    Normality test based on the KBQD tests using the centered Gaussian kernel.

    For each replication, a sample from the d-dimensional Normal distribution with mean
    vector mu_hat and covariance matrix sigma_hat is generated and the KBQD test U-statistic
    for Normality is computed. After num_iter iterations, the critical value is selected as
    the quantile of the empirical distribution of the computed test statistics.

    Parameters
    ----------
        d : int
            The dimension of generated samples.

        size : int
            The number of observations to be generated.

        h : float
            The concentration parameter for the Gaussian kernel.

        mu_hat : numpy.ndarray
            Mean vector for the reference distribution.

        sigma_hat : numpy.array
            Covariance matrix of the reference distribution.

        num_iter : int, optional
            The number of replications.

        quantile : float, optional
            The quantile of the distribution used to select the critical value.

        random_state : int, None, optional.
            Seed for random number generation. Defaults to None.

        n_jobs : int, optional
            n_jobs specifies the maximum number of concurrently
            running workers. If 1 is given, no joblib parallelism
            is used at all, which is useful for debugging. For more
            information on joblib n_jobs refer to -
            https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html.

    Returns
    -------
        critical value : float
            Critical value for the specified dimension, size and quantile.
    """
    results = Parallel(n_jobs=n_jobs)(
        delayed(normal_cv_helper)(size, h, mu_hat, sigma_hat, i, random_state)
        for i in range(num_iter)
    )
    return np.quantile(results, quantile, axis=0)


def cv_ksample(
    x,
    y,
    h,
    num_iter=150,
    b=0.9,
    quantile=0.95,
    method="subsampling",
    random_state=None,
    n_jobs=8,
):
    """
    Compute the critical value for k-sample kernel tests.

    Parameters
    --------------
        x : numpy.ndarray
            Matrix containing the observations to be used in the k-sample test.

        y : numpy.ndarray
            Vector indicating the sample for each observation.

        h : float
            The tuning parameter for the test using the Gaussian kernel.

        num_iter : int, optional. Defaults to 150.
            The number of bootstrap/permutation/subsampling samples to generate.

        b : float, optional. Defaults to 0.9
            The subsampling block size (only used if `method` is "subsampling").

        quantile : float, optional. Defaults to 0.95
            The quantile of the bootstrap/permutation/subsampling distribution
            to use as the critical value.

        method : str
            The method to use for computing the critical value
            (one of "bootstrap", "permutation" or "subsampling").

        random_state : int, None, optional.
            Seed for random number generation. Defaults to None.

        n_jobs : int, optional
            n_jobs specifies the maximum number of concurrently
            running workers. If 1 is given, no joblib parallelism
            is used at all, which is useful for debugging. For more
            information on joblib n_jobs refer to -
            https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html.

    Returns
    ---------
        critical value : float
            A vector of two critical values corresponding to different formulations
            of the test statistics.

    References
    -----------
        Markatou Marianthi, Saraceno Giovanni, Chen Yang (2023). “Two- and k-Sample Tests Based on
        Quadratic Distances.” Manuscript, (Department of Biostatistics, University at Buffalo).
    """
    sizes = np.unique(y, return_counts=True)[1]
    n = len(y)
    k = len(sizes)
    cum_size = np.insert(np.cumsum(sizes), 0, 0)

    if method == "bootstrap":
        results = Parallel(n_jobs=n_jobs)(
            delayed(bootstrap_helper_ksample)(
                x, y, k, h, sizes, cum_size, i, random_state
            )
            for i in range(num_iter)
        )
    elif method == "subsampling":
        results = Parallel(n_jobs=n_jobs)(
            delayed(subsampling_helper_ksample)(
                x, y, k, h, sizes, b, cum_size, i, random_state
            )
            for i in range(num_iter)
        )
    elif method == "permutation":
        results = Parallel(n_jobs=n_jobs)(
            delayed(permutation_helper_ksample)(x, y, n, h, i, random_state)
            for i in range(num_iter)
        )

    cv = np.quantile(results, quantile, axis=0)
    return cv
