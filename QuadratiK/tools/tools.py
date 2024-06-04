"""
Contains additional tools. 
"""

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_random_state
from ._utils import _stats_helper


def stats(x, y=None):
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
    >>> import numpy as np
    >>> from QuadratiK.tools import stats
    >>> np.random.seed(42)
    >>> X = np.random.randn(100,4)
    >>> stats(X)
    ...           Feature 0  Feature 1  Feature 2  Feature 3
        Mean     -0.009811   0.033746   0.022496   0.043764
        Std Dev   0.868065   0.952234   1.044014   0.982240
        Median   -0.000248  -0.024646   0.068665   0.075219
        IQR       1.244319   1.111478   1.318245   1.506492
        Min      -2.025143  -1.959670  -3.241267  -1.987569
        Max       2.314659   3.852731   2.189803   2.720169
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


def sample_hypersphere(npoints=100, ndim=3, random_state=None):
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
    >>> from QuadratiK.tools import sample_hypersphere
    >>> sample_hypersphere(100,3,random_state = 42)
    ... array([[ 0.60000205, -0.1670153 ,  0.78237039],
    ...        [ 0.97717133, -0.15023209, -0.15022156], ........
    """
    if not isinstance(random_state, (int, type(None))):
        raise ValueError("Please specify a integer or None random_state")

    generator = check_random_state(random_state)
    dat = generator.randn(npoints, ndim)
    dat = dat / np.linalg.norm(dat, axis=1, keepdims=True)
    return dat
