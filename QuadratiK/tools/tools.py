"""
Contains additional tools.
"""

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_random_state

from ._utils import _stats_helper


def stats(
    x: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    The stats function calculates statistics for one or multiple groups of data.

    Parameters
    ----------
    x : numpy.ndarray, pandas.DataFrame
        Data for which statistics is to be calculated.

    y : numpy.ndarray, pandas.DataFrame, optional
        The parameter `y` is an optional input that can be either another set of observations,
        or the associated labels for observations (data points).

    Returns
    -------
    summary statistics : pandas.DataFrame
        Summary statistics of the input data.

    Examples
    --------
    .. jupyter-execute::

        import numpy as np
        from QuadratiK.tools import stats
        np.random.seed(42)
        X = np.random.randn(100,4)
        print(stats(X))
    """

    x_stats = _stats_helper(x)

    if y is None:
        summary_stats_df = x_stats

    if y is not None:
        statistics = {}
        if len(np.unique(y)) > 10:
            y_stats = _stats_helper(y)

            pooled = np.concatenate((x, y))
            pooled_stats = _stats_helper(pooled)

            for feature in x_stats.columns:
                statistics[feature] = pd.concat(
                    [x_stats[[feature]], y_stats[[feature]], pooled_stats[[feature]]],
                    axis=1,
                )
                statistics[feature].columns = ["Group 1", "Group 2", "Overall"]

            summary_stats_df = pd.concat(
                statistics.values(), keys=statistics.keys(), axis=0
            )

        else:
            statistics = {}
            y = y.flatten()
            overall_stats = _stats_helper(x)
            groups = [x[y == m] for m in set(y)]
            group_stats = [pd.DataFrame(_stats_helper(dat)) for dat in groups]
            for col in range(x.shape[1]):
                statistics["Feature " + str(col)] = pd.concat(
                    [
                        pd.DataFrame(group_stats[i][["Feature " + str(col)]])
                        for i in range(len(set(y)))
                    ]
                    + [overall_stats["Feature " + str(col)]],
                    axis=1,
                )
                statistics["Feature " + str(col)].columns = [
                    ("Group " + str(int(k))) for k in set(y)
                ] + ["Overall"]

            summary_stats_df = pd.concat(
                statistics.values(), keys=statistics.keys(), axis=0
            )

    return summary_stats_df


def sample_hypersphere(
    npoints: int = 100, ndim: int = 3, random_state: int | None = None
) -> np.ndarray:
    """
    Generate random samples from the hypersphere.

    Parameters
    --------------
    npoints : int, optional.
        The number of points to generate.
        Default is 100.

    ndim : int, optional.
        The dimensionality of the hypersphere.
        Default is 3.

    random_state : int, None, optional.
        Seed for random number generation. Defaults to None.

    Returns
    ---------
    data on sphere : numpy.ndarray
        An array containing random vectors sampled uniformly
        from the surface of the hypersphere.

    Examples
    ---------
    .. jupyter-execute::

        from QuadratiK.tools import sample_hypersphere
        print(sample_hypersphere(5, 3, random_state = 42))
    """
    if not isinstance(random_state, (int, type(None))):
        raise ValueError("Please specify a integer or None random_state")

    generator = check_random_state(random_state)
    dat = generator.randn(npoints, ndim)
    dat = dat / np.linalg.norm(dat, axis=1, keepdims=True)
    return dat
