"""
Critical value for the uniformity test on the sphere based
on the centered poisson kernel tests
"""

from typing import Optional

import numpy as np
from numpy.random import SeedSequence, default_rng
from sklearn.utils.parallel import Parallel, delayed

from ._utils import poisson_cv_helper


def poisson_cv(
    d: int,
    size: int,
    rho: float,
    num_iter: int,
    quantile: float,
    random_state: Optional[int] = None,
    n_jobs: int = 8,
) -> float:
    """
    Perform a Poisson kernel-based test for uniformity multiple
    times and return the quantile of the results.

    Parameters
    --------------
    d : int
        The dimension of the observations.

    size : int
        The size of each sample to be generated on the Sphere.

    rho : float
        Concentration parameter of the Poisson kernel.

    num_iter : int
        The number of iterations to perform the Poisson kernel-based test.

    quantile : float
        The quantile value to be calculated from the results.

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
        Critical value for test of uniformity using the poisson kernel.

    References
    ----------
        Ding Yuxin, Markatou Marianthi, Saraceno Giovanni (2023). “Poisson Kernel-Based Tests for
        Uniformity on the d-Dimensional Sphere.” Statistica Sinica. doi: doi:10.5705/ss.202022.0347.
    """

    ss = SeedSequence(random_state)
    child_seeds = ss.spawn(num_iter)
    generators = [default_rng(s) for s in child_seeds]

    results = Parallel(n_jobs=n_jobs)(
        delayed(poisson_cv_helper)(size, d, rho, generators[i]) for i in range(num_iter)
    )
    return np.quantile(results, quantile)
