"""
Contains the tuning parameter selection algorithm
"""

import numpy as np
import pandas as pd
from scipy.stats import skew, skewnorm
from sklearn.utils.parallel import Parallel, delayed
import matplotlib.pyplot as plt

from ._utils import stat_ksample, stat_two_sample, stat_normality_test
from ._cv_functions import cv_ksample, cv_twosample, cv_normality


def _objective_one_sample(
    alternative,
    delta,
    delta_dim,
    h,
    mean_dat,
    n,
    num_iter,
    quantile,
    rep_values,
    s_dat,
    skew_data,
    random_state,
    n_jobs=1,
):
    """
    Objective function using using the best
    h is chosen for one sample test

    Parameters
    ----------
        alternative : str
            family of alternative chosen for selecting h,
            must be one of "mean", "variance" and "skewness".

        delta : numpy.ndarray
            Array of parameter values indicating chosen alternatives.

        delta_dim : int, numpy.ndarray
            Array of coefficient of alternative with respect
            to each dimension.

        h : float
            Bandwidth for the kernel function.

        mean_dat : numpy.ndarray
            Means of the multivariate distribution to be used
            for determining the best h.

        n : int
            Number of observations in set of samples.

        num_iter : int
            The number of iterations to use for critical value estimation.

        quantile : float
            Quantile to use for critical value estimation.

        rep_values : int
            Number of the bootstrap replication.

        s_dat : numpy.ndarray
            Variances of the multivariate distribution to be used
            for determining the best h.

        skew_data : numpy.ndarray
            Skewness of the multivariate distribution to be used
            for determining the best h.

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
        List containing rep_values, delta, h and boolean
        representing the rejection/acceptance of null hypothesis.
    """
    dk = delta_dim * delta
    if alternative == "location":
        mean_tilde = mean_dat + dk
        s_tilde = s_dat
        skew_tilde = skew_data
    elif alternative == "scale":
        mean_tilde = mean_dat
        s_tilde = s_dat * dk
        skew_tilde = skew_data
    elif alternative == "skewness":
        mean_tilde = mean_dat
        skew_tilde = skew_data + dk
        s_tilde = s_dat

    if isinstance(random_state, int):
        random_state = random_state + int(rep_values)
    xnew = skewnorm.rvs(
        size=(n, len(mean_dat)),
        loc=mean_tilde,
        scale=s_tilde,
        a=skew_tilde,
        random_state=random_state,
    )

    statistic = stat_normality_test(xnew, h, np.array([mean_dat]), np.diag(s_dat))
    cv = cv_normality(
        n,
        h,
        np.array([mean_dat]),
        np.diag(s_dat),
        num_iter,
        quantile,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    h0 = statistic < cv
    return [rep_values, delta, h, h0[0]]


def _objective_two_sample(
    alternative,
    b,
    delta,
    delta_dim,
    h,
    m,
    mean_dat,
    method,
    n,
    num_iter,
    pooled,
    quantile,
    rep_values,
    s_dat,
    skew_data,
    d,
    random_state,
    n_jobs=1,
):
    """
    Objective function using using the best
    h is chosen for two sample test.

    Parameters
    ----------
        alternative : str
            family of alternative chosen for selecting h,
            must be one of "mean", "variance" and "skewness".

        b : float
            The size of the subsamples used in the subsampling algorithm.

        delta : numpy.ndarray
            Array of parameter values indicating chosen alternatives.

        delta_dim : int, numpy.ndarray
            Array of coefficient of alternative with respect
            to each dimension.

        h : float
            Bandwidth for the kernel function.

        m : int
            Number of observations in second set of samples.

        mean_dat : numpy.ndarray
            Means of the multivariate distribution to be used
            for determining the best h.

        method : str
            the method used for critical value estimation,
            must be one of "subsampling", "bootstrap", or "permutation".

        n : int
            Number of observations in first set of samples.

        num_iter : int
            The number of iterations to use for critical value estimation.

        pooled : numpy.ndarray
            Observations in first set and second
            set of samples combined together retaining the number of columns.

        quantile : float
            Quantile to use for critical value estimation.

        rep_values : int
            Number of the bootstrap replication.

        s_dat : numpy.ndarray
            Variances of the multivariate distribution to be used
            for determining the best h.

        skew_data : numpy.ndarray
            Skewness of the multivariate distribution to be used
            for determining the best h.

        d : int
            Dimension of the data.

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
        List containing rep_values, delta, h and boolean
        representing the rejection/acceptance of null hypothesis.
    """

    dk = delta_dim * delta
    if alternative == "location":
        mean_tilde = mean_dat + dk
        s_tilde = s_dat
        skew_tilde = skew_data
    elif alternative == "scale":
        mean_tilde = mean_dat
        s_tilde = s_dat * dk
        skew_tilde = skew_data
    elif alternative == "skewness":
        mean_tilde = mean_dat
        skew_tilde = skew_data + dk
        s_tilde = s_dat

    if isinstance(random_state, (int, np.int_)):
        random_state = random_state + int(rep_values)

    xnew = skewnorm.rvs(
        size=(n, len(mean_dat)),
        loc=mean_dat,
        scale=s_dat,
        a=skew_data,
        random_state=np.random.default_rng(random_state),
    )
    ynew = skewnorm.rvs(
        size=(m, len(mean_dat)),
        loc=mean_tilde,
        scale=s_tilde,
        a=skew_tilde,
        random_state=np.random.default_rng(random_state),
    )

    statistic = stat_two_sample(xnew, ynew, h, np.repeat(0, d), np.eye(d))
    cv = cv_twosample(
        num_iter,
        quantile,
        pooled,
        n,
        m,
        h,
        method,
        b,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    h0 = statistic[:2] < cv
    return [rep_values, delta, h, h0[0]]


def _objective_k_sample(
    alternative,
    num_iter,
    b,
    delta,
    delta_dim,
    h,
    k,
    mean_dat,
    method,
    n,
    quantile,
    rep_values,
    s_dat,
    skew_data,
    random_state,
    n_jobs=1,
):
    """
    Objective function using using the best
    h is chosen for k-sample test.

    Parameters
    ----------
        alternative : str
            family of alternative chosen for selecting h,
            must be one of "mean", "variance" and "skewness".

        num_iter : int
            The number of iterations to use for critical value estimation.

        b : float
            The size of the subsamples used in the subsampling algorithm.

        delta : numpy.ndarray
            Array of parameter values indicating chosen alternatives.

        delta_dim : int, numpy.ndarray
            Array of coefficient of alternative with respect
            to each dimension.

        h : float
            Bandwidth for the kernel function.

        k : int
            Number of classes (or groups) in the data.

        mean_dat : numpy.ndarray
            Means of the multivariate distribution to be used
            for determining the best h.

        method : str
            the method used for critical value estimation,
            must be one of "subsampling", "bootstrap", or "permutation".

        n : int
            Number of total observations in the provided data.

        quantile : float
            Quantile to use for critical value estimation.

        rep_values : int
            Number of the bootstrap replication.

        s_dat : numpy.ndarray
            Variances of the multivariate distribution to be used
            for determining the best h.

        skew_data : numpy.ndarray
            Skewness of the multivariate distribution to be used
            for determining the best h.

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
        List containing rep_values, delta, h and boolean
        representing the rejection/acceptance of null hypothesis.
    """
    dk = delta_dim * delta
    if alternative == "location":
        mean_tilde = mean_dat + dk
        s_tilde = s_dat
        skew_tilde = skew_data
    elif alternative == "scale":
        mean_tilde = mean_dat
        s_tilde = s_dat * dk
        skew_tilde = skew_data
    elif alternative == "skewness":
        mean_tilde = mean_dat
        skew_tilde = skew_data + dk
        s_tilde = s_dat

    if isinstance(random_state, (int, np.int_)):
        random_state = random_state + int(rep_values)

    nk = round(n / k)
    xnew = skewnorm.rvs(
        size=(nk * (k - 1), len(mean_dat)),
        loc=mean_dat,
        scale=s_dat,
        a=skew_data,
        random_state=random_state,
    )
    xk = skewnorm.rvs(
        size=(nk, len(mean_dat)),
        loc=mean_tilde,
        scale=s_tilde,
        a=skew_tilde,
        random_state=random_state,
    )

    xnew = np.concatenate((xnew, xk), axis=0)
    ynew = np.repeat(np.arange(1, k + 1), repeats=nk)
    statistic = stat_ksample(xnew, ynew, h)
    cv = cv_ksample(
        xnew,
        ynew,
        h,
        num_iter,
        b,
        quantile,
        method,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    h0 = statistic[:2] < cv
    return [rep_values, delta, h, h0[0]]


def select_h(
    x,
    y=None,
    alternative="location",
    method="subsampling",
    b=0.8,
    num_iter=150,
    delta_dim=1,
    delta=None,
    h_values=None,
    n_rep=50,
    n_jobs=8,
    quantile=0.95,
    k_threshold=10,
    power_plot=False,
    random_state=None,
):
    """
    This function computes the kernel bandwidth of the Gaussian kernel
    for the one sample, two-sample and k-sample kernel-based quadratic
    distance (KBQD) tests.

    The function performs the selection of the optimal value for the tuning
    parameter h of the normal kernel function, for the two-sample and k-sample
    KBQD tests. It performs a small simulation study, generating samples according
    to the family of a specified alternative, for the chosen values
    of h_values and delta.

    Parameters
    ----------
        x : numpy.ndarray or pandas.DataFrame
            Data set of observations from X.

        y : numpy.ndarray or pandas.DataFrame, optional
            Data set of observations from Y for two sample test
            or set of labels in case of k-sample test.

        alternative : str, optional
            Family of alternative chosen for selecting h,
            must be one of "location", "scale" and "skewness".
            Defaults to "location".

        method : str, optional.
            The method used for critical value estimation,
            must be one of "subsampling", "bootstrap", or "permutation".
            Defaults to "subsampling".

        b : float, optional.
            The size of the subsamples used in the subsampling algorithm.
            Defaults to 0.8 i.e. `0.8N` samples are used, where `N`
            represents the total sample size.

        num_iter : int, optional.
            The number of iterations to use for critical value estimation.
            Defaults to 150.

        delta_dim : int, numpy.ndarray, optional.
            Array of coefficient of alternative with respect to each dimension.
            Defaults to 1.

        delta : numpy.ndarray, optional.
            Array of parameter values indicating chosen alternatives.
            Defaults to None.

        h_values : numpy.ndarray, optional.
            Values of the tuning parameter used for the selection.
            Defaults to None.

        n_rep : int, optional. Defaults to 50.
            Number of bootstrap replications.

        n_jobs : int, optional.
            n_jobs specifies the maximum number of concurrently running workers.
            If 1 is given, no joblib parallelism is used at all,
            which is useful for debugging. For more information on joblib n_jobs
            refer to - https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html.
            Defaults to 8.

        quantile : float, optional.
            Quantile to use for critical value estimation. Defaults to 0.95.

        k_threshold : int.
            Maximum number of groups allowed. Defaults to 10.

        power_plot : boolean, optional.
            If True, plot is displayed the plot of power for
            values in h\\_values and delta. Defaults to False.

        random_state : int, None, optional.
            Seed for random number generation. Defaults to None.

    Returns
    -------
        h : float
            The selected value of tuning parameter h.

        h vs Power table : pandas.DataFrame
            A table containing the h, delta and corresponding powers.

    References
    -----------
        Markatou M., Saraceno G., Chen Y. (2023). “Two- and k-Sample Tests
        Based on Quadratic Distances. ”Manuscript, (Department of Biostatistics,
        University at Buffalo).

    Examples
    --------
    >>> import numpy as np
    >>> from QuadratiK.kernel_test import select_h
    >>> np.random.seed(42)
    >>> X = np.random.randn(200, 2)
    >>> np.random.seed(42)
    >>> y = np.random.randint(0, 2, 200)
    >>> h_selected, all_values, power_plot = select_h(
    ...    X, y, alternative='location', power_plot=True, random_state=42)
    >>> print("Selected h is: ", h_selected)
    ... Selected h is:  1.2
    """
    if not isinstance(random_state, (int, np.int_, type(None))):
        raise ValueError("Please specify a integer or None random_state")

    if isinstance(x, np.ndarray):
        if x.ndim == 1:
            x = np.array(x).reshape(-1, 1)
    elif isinstance(x, pd.DataFrame):
        x = x.to_numpy()
    else:
        raise TypeError("x must be a numpy array or a pandas DataFrame")

    if y is not None:
        if isinstance(y, (list, np.ndarray, pd.Series)):
            if y.ndim > 1:
                pass
            else:
                y = np.array(y).reshape(-1, 1)
        elif isinstance(y, pd.DataFrame):
            y = y.values
        else:
            raise TypeError(
                "y must be a list, numpy.ndarray, \
                    or a pandas DataFrame with one column (for k-sample case) or more"
            )
    else:
        pass

    if h_values is None:
        h_values = np.round(np.arange(0.4, 3.4, 0.4), 2)

    if delta is None:
        if alternative == "location":
            delta = np.array([0.2, 0.3, 0.4])
        elif alternative == "scale":
            delta = np.array([1.1, 1.3, 1.5])
        elif alternative == "skewness":
            delta = np.array([0.2, 0.3, 0.6])

    if (alternative is None) or (alternative not in ["location", "scale", "skewness"]):
        raise ValueError(
            "Please specify alternative from 'location', 'scale' or 'skewness'"
        )

    if y is not None:
        sizes = np.unique(y, return_counts=True)[1]
    else:
        sizes = []

    rep_values = np.arange(0, n_rep, 1)
    k = len(sizes)

    if k <= 1:
        n, d = x.shape
        if isinstance(delta_dim, (int, float)):
            delta_dim = np.ones(d)
        else:
            if not isinstance(delta_dim, (int, float)) or len(delta_dim) != d:
                raise ValueError(
                    "delta_dim must be 1 or a numeric Array of \
                        length equal to the number of columns of pooled."
                )

        mean_dat = np.mean(x, axis=0)
        s_dat = np.diag(np.cov(x, rowvar=False).reshape(x.shape[1], x.shape[1]))
        skew_data = skew(x)
        all_parameters = np.array(np.meshgrid(h_values, delta, rep_values)).T.reshape(
            -1, 3
        )

        all_results = {}
        for delta_val in delta:
            parameters = all_parameters[all_parameters[:, 1] == delta_val]
            results = Parallel(n_jobs=n_jobs)(
                delayed(_objective_one_sample)(
                    alternative,
                    param[1],
                    delta_dim,
                    param[0],
                    mean_dat,
                    n,
                    num_iter,
                    quantile,
                    param[2],
                    s_dat,
                    skew_data,
                    random_state,
                    1,
                )
                for param in parameters
            )
            results = pd.DataFrame(results, columns=["rep", "delta", "h", "score"])
            results["score"] = 1 - results["score"]
            results_mean = (
                results.groupby(["h", "delta"]).agg({"score": "mean"}).reset_index()
            )
            results_mean.columns = ["h", "delta", "power"]
            results_mean = results_mean.sort_values(by=["delta", "h"])
            all_results[delta_val] = results_mean
            min_h_power_gt_05 = results_mean[results_mean["power"] >= 0.5]
            if not min_h_power_gt_05.empty:
                min_h = results_mean[results_mean["power"] >= 0.50].iloc[0]["h"]
                break

    elif k > k_threshold:
        if x.shape[1] != y.shape[1]:
            raise ValueError("'x' and 'y' must have the same number of columns")

        n = x.shape[0]
        m = y.shape[0]
        pooled = np.concatenate((x, y), axis=0)
        d = pooled.shape[1]

        if isinstance(delta_dim, (int, float)):
            delta_dim = np.ones(d)
        else:
            if not isinstance(delta_dim, (int, float)) or len(delta_dim) != d:
                raise ValueError(
                    "delta_dim must be 1 or a numeric Array of \
                        length equal to the number of columns of pooled."
                )

        mean_dat = np.mean(pooled, axis=0)
        s_dat = np.diag(
            np.cov(pooled, rowvar=False).reshape(pooled.shape[1], pooled.shape[1])
        )
        skew_data = skew(pooled)
        all_parameters = np.array(np.meshgrid(h_values, delta, rep_values)).T.reshape(
            -1, 3
        )

        all_results = {}
        for delta_val in delta:
            parameters = all_parameters[all_parameters[:, 1] == delta_val]
            results = Parallel(n_jobs=n_jobs)(
                delayed(_objective_two_sample)(
                    alternative,
                    b,
                    param[1],
                    delta_dim,
                    param[0],
                    m,
                    mean_dat,
                    method,
                    n,
                    num_iter,
                    pooled,
                    quantile,
                    param[2],
                    s_dat,
                    skew_data,
                    d,
                    random_state,
                    1,
                )
                for param in parameters
            )
            results = pd.DataFrame(results, columns=["rep", "delta", "h", "score"])
            results["score"] = 1 - results["score"]
            results_mean = (
                results.groupby(["h", "delta"]).agg({"score": "mean"}).reset_index()
            )
            results_mean.columns = ["h", "delta", "power"]
            results_mean = results_mean.sort_values(by=["delta", "h"])
            all_results[delta_val] = results_mean
            min_h_power_gt_05 = results_mean[results_mean["power"] >= 0.5]
            if not min_h_power_gt_05.empty:
                min_h = results_mean[results_mean["power"] >= 0.50].iloc[0]["h"]
                break
    else:
        n, d = x.shape
        if isinstance(delta_dim, (int, float)):
            delta_dim = np.ones(d)
        else:
            if not isinstance(delta_dim, (int, float)) or len(delta_dim) != d:
                raise ValueError(
                    "delta_dim must be 1 or a numeric Array of \
                        length equal to the number of columns of pooled."
                )

        mean_dat = np.mean(x, axis=0)
        s_dat = np.diag(np.cov(x, rowvar=False).reshape(x.shape[1], x.shape[1]))
        skew_data = skew(x)

        all_parameters = np.array(np.meshgrid(h_values, delta, rep_values)).T.reshape(
            -1, 3
        )

        all_results = {}
        for delta_val in delta:
            parameters = all_parameters[all_parameters[:, 1] == delta_val]
            results = Parallel(n_jobs=n_jobs)(
                delayed(_objective_k_sample)(
                    alternative,
                    num_iter,
                    b,
                    param[1],
                    delta_dim,
                    param[0],
                    k,
                    mean_dat,
                    method,
                    n,
                    quantile,
                    param[2],
                    s_dat,
                    skew_data,
                    random_state,
                    1,
                )
                for param in parameters
            )

            results_df = pd.DataFrame(results, columns=["rep", "delta", "h", "score"])
            results_df["score"] = 1 - results_df["score"]
            results_mean = (
                results_df.groupby(["h", "delta"]).agg({"score": "mean"}).reset_index()
            )
            results_mean.columns = ["h", "delta", "power"]
            results_mean = results_mean.sort_values(by=["delta", "h"])
            all_results[delta_val] = results_mean
            min_h_power_gt_05 = results_mean[results_mean["power"] >= 0.5]
            if not min_h_power_gt_05.empty:
                min_h = results_mean[results_mean["power"] >= 0.50].iloc[0]["h"]
                break

    all_results = pd.concat(all_results.values())
    if "min_h" not in locals():
        min_h = results_mean.loc[results_mean["power"].idxmax()]["h"]

    if power_plot:
        groups = all_results.groupby("delta")
        figure = plt.figure(figsize=(8, 4))
        for delta, group in groups:
            plt.plot(
                group["h"],
                group["power"],
                marker="o",
                linestyle="-",
                label=f"delta={round(delta, 3)}",
            )
        plt.xlabel("h")
        plt.ylabel("Power")
        plt.title("h vs Power for different delta")
        plt.legend()
        plt.close()
        return (min_h, all_results, figure)

    return (min_h, all_results)
