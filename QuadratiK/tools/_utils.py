import time
import warnings
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from skfda.exploratory.stats import geometric_median
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

plt.ioff()


def _qq_plot_twosample(
    sample1: np.ndarray | pd.DataFrame, sample2: np.ndarray | pd.DataFrame
) -> plt.Figure:
    """
    Generate a Quantile-Quantile (QQ) plot for two samples.

    Parameters
    --------------
    sample1 : numpy.ndarray or pandas.DataFrame
        The first sample for comparison.
    sample2 : numpy.ndarray or pandas.DataFrame
        The second sample for comparison.

    Returns
    ---------
    figure : matplotlib.figure.Figure
        The QQ plot figure.
    """
    if isinstance(sample1, pd.DataFrame):
        sample1 = sample1.to_numpy()

    if isinstance(sample2, pd.DataFrame):
        sample2 = sample2.to_numpy()

    fig, axes = plt.subplots(
        nrows=sample1.shape[1], ncols=1, figsize=(6, sample1.shape[1] * 3)
    )

    for col in range(sample1.shape[1]):
        quantiles1 = np.quantile(sample1[:, col], np.arange(0, 1, 1 / sample1.shape[0]))
        quantiles2 = np.quantile(sample2[:, col], np.arange(0, 1, 1 / sample2.shape[0]))
        axes[col].plot(quantiles1, quantiles2, color="blue")

        model = LinearRegression()
        model.fit(quantiles1.reshape(-1, 1), quantiles2.reshape(-1, 1))
        predictions = model.predict(quantiles1.reshape(-1, 1))

        axes[col].plot(quantiles1, predictions, color="red", linestyle="--")
        axes[col].set_title("QQ Plot for feature: " + str(col))
        axes[col].set_xlabel("Q1")
        axes[col].set_ylabel("Q2")
        fig.suptitle("QQ Plots", fontsize=16)
    fig.subplots_adjust(hspace=0.5)
    return fig


def _qq_plot_onesample(
    sample1: np.ndarray | pd.DataFrame, dist: str = "norm"
) -> plt.Figure:
    """
    Generate a Quantile-Quantile (QQ) plot for a single sample against a specified distribution.

    Parameters
    --------------
    sample1 : numpy.ndarray or pandas.DataFrame
        The sample for comparison.

    dist : str, optional
        The distribution to compare against. Default is "norm" for the normal distribution.

    Returns
    ---------
    figure : matplotlib.figure.Figure
        The QQ plot figure.

    Notes
    --------------
    This function uses scipy.stats.probplot

    See Also
    --------------
    scipy.stats.probplot : Probability plot function.
    """
    if isinstance(sample1, pd.DataFrame):
        sample1 = sample1.to_numpy()

    fig, axes = plt.subplots(
        nrows=sample1.shape[1], ncols=1, figsize=(6, sample1.shape[1] * 3)
    )
    for col in range(sample1.shape[1]):
        stats_probplot_vals = stats.probplot(sample1[:, col], dist=dist)
        theoretical_quantiles = stats_probplot_vals[0][0]
        quantiles1 = stats_probplot_vals[0][1]
        axes[col].plot(theoretical_quantiles, quantiles1, color="blue")
        axes[col].plot(
            theoretical_quantiles,
            stats_probplot_vals[1][1]
            + stats_probplot_vals[1][0] * theoretical_quantiles,
            color="red",
            linestyle="--",
        )

        axes[col].plot(theoretical_quantiles, quantiles1, color="red", linestyle="--")
        axes[col].set_title(dist + " QQ Plot for feature: " + str(col))
        axes[col].set_xlabel("Theoretical Quantiles")
        axes[col].set_ylabel("Sample Quantiles")
        fig.suptitle("QQ Plots", fontsize=16)
    fig.subplots_adjust(hspace=0.5)
    return fig


def _spherical_pca(data: np.ndarray, scale: bool = False) -> dict:
    """
    Perform Spherical Principal Component Analysis (PCA). This is a Python
    implementation of PcaLocantore, the spherical PCA implementation in the
    R package rrcov.

    Parameters
    --------------
    data : numpy.ndarray
        The input data matrix of shape (n_samples, n_features).
    scale : bool, optional
        Whether to scale the data using the Median Absolute Deviation (MAD)
        before processing. Default is False.

    Returns
    ---------
    results : dict
        A dictionary containing:

        - `scores` : numpy.ndarray
            The principal component scores (centered data projected onto the spherical PCA loadings).
        - `loadings` : numpy.ndarray
            The principal component loadings (eigenvectors).
        - `eigenvalues` : numpy.ndarray
            The estimated eigenvalues (squared MAD of projections).

    See Also
    --------------
    PcaLocantore : Spherical PCA implementation in the R package rrcov.
    https://search.r-project.org/CRAN/refmans/rrcov/html/PcaLocantore-class.html

    sklearn.decomposition.PCA : Standard Principal Component Analysis.

    Examples
    --------
    >>> import numpy as np
    >>> from QuadratiK.tools import spherical_pca
    >>> data = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 100)
    >>> results = spherical_pca(data)
    """
    if scale:
        data = data / stats.median_abs_deviation(data, axis=0, scale="normal")

    spatial_median = geometric_median(data)
    centered_data = data - spatial_median
    projected_data = centered_data / np.linalg.norm(
        centered_data, axis=1, keepdims=True
    )
    pca = PCA()
    pca.fit(projected_data)
    loadings = pca.components_
    scores1 = projected_data @ loadings.T
    scores = centered_data @ loadings.T
    sdev = stats.median_abs_deviation(scores1, axis=0, scale="normal")
    orsdev = np.argsort(sdev)[::-1]
    sdev = sdev[orsdev]
    eigenvalues = sdev**2

    scores = scores[:, orsdev]
    loadings = loadings[orsdev, :]
    eigenvalues = sdev**2
    return {"scores": scores, "loadings": loadings, "eigenvalues": eigenvalues}


def _extract_3d(
    data: np.ndarray,
    use_dimensionality_reduction: bool = True,
    use_dimensionality_reduction_method: str = "spherical_pca",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract the first three principal components or features of the data
    for 3D visualization.

    Parameters
    --------------
    data : numpy.ndarray
        The input data matrix of shape (n_samples, n_features).
    use_dimensionality_reduction : bool, optional
        Whether to apply dimensionality reduction. If False, the first three
        columns of the data are used. Default is True.
    use_dimensionality_reduction_method : str, optional
        The method to use for dimensionality reduction ('spherical_pca' or 'pca').
        Default is 'spherical_pca'.

    Returns
    ---------
    components : tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple containing three arrays representing the first three
        principal components or features.

    See Also
    --------------
    sklearn.decomposition.PCA : Principal Component Analysis.
    _spherical_pca : Spherical Principal Component Analysis.
    """
    if use_dimensionality_reduction:
        if use_dimensionality_reduction_method == "spherical_pca":
            res = _spherical_pca(data, scale=False)
            data_pca = res["scores"][:, 0:3]
        elif use_dimensionality_reduction_method == "pca":
            pca = PCA(n_components=3)
            data_pca = pca.fit_transform(data)
        else:
            raise ValueError(
                f"Invalid dimensionality reduction method: {use_dimensionality_reduction_method}. "
                "Supported methods are 'spherical_pca' and 'pca'."
            )
    else:
        warnings.warn(
            "use_dimensionality_reduction is False, using only first 3 cols",
            stacklevel=2,
        )
        data_pca = data[:, 0:3]

    return (data_pca[:, 0], data_pca[:, 1], data_pca[:, 2])


def _stats_helper(dat: np.ndarray) -> pd.DataFrame:
    """
    Calculate descriptive statistics for each feature in the input data.

    Parameters
    --------------
    dat : numpy.ndarray
        The input data.

    Returns
    ---------
    descriptive statistics : pandas.DataFrame
        A DataFrame containing mean, standard deviation, median,
        interquartile range (IQR), minimum, and maximum values
        for each feature.
    """
    dat_mean = np.mean(dat, axis=0)
    dat_std = np.std(dat, axis=0, ddof=1)
    dat_median = np.median(dat, axis=0)
    dat_iqr = np.quantile(dat, 0.75, axis=0) - np.quantile(dat, 0.25, axis=0)
    dat_min = np.min(dat, axis=0)
    dat_max = np.max(dat, axis=0)
    dat_stats = pd.DataFrame([dat_mean, dat_std, dat_median, dat_iqr, dat_min, dat_max])
    dat_stats.columns = ["Feature " + str(i) for i in range(dat.shape[1])]
    dat_stats = dat_stats.set_axis(["Mean", "Std Dev", "Median", "IQR", "Min", "Max"])
    return dat_stats


F = TypeVar("F", bound=Callable[..., Any])


def class_method_call_timing(func: F) -> F:
    """
    Decorator to measure the execution time of a
    class method and store it in the instance.

    Parameters
    --------------
    func : callable
        The class method to be timed.

    Returns
    ---------
    callable
        The decorated class method.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        instance = args[0]
        instance.execution_time = execution_time
        return result

    return wrapper
