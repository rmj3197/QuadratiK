import time
from functools import wraps
import matplotlib.pyplot as plt

plt.ioff()
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression


def _qq_plot_twosample(sample1, sample2):
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
    plt.close()
    return fig


def _qq_plot_onesample(sample1, dist="norm"):
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
    plt.close()
    return fig


def _extract_3d(data):
    """
    Extract the first three principal components of the data and normalize them.

    Parameters
    --------------
    data : numpy.ndarray
        The input data.

    Returns
    ---------
    principal components : tuple
        A tuple containing three arrays representing the first three principal components.

    See Also
    --------------
    sklearn.decomposition.PCA : Principal Component Analysis.
    """
    pca = PCA(n_components=3)
    data_pca = pca.fit_transform(data)[:, 0:3]
    data_pca = data_pca / np.linalg.norm(data_pca, axis=1, keepdims=True)
    return (data_pca[:, 0], data_pca[:, 1], data_pca[:, 2])


def _stats_helper(dat):
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


def class_method_call_timing(func):
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
