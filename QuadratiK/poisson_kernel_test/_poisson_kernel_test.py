"""
Poisson kernel-based quadratic distance test of Uniformity on the Sphere
"""

import importlib
import numpy as np
import pandas as pd
from scipy.stats import chi2
from tabulate import tabulate

from ._utils import dof, stat_poisson_unif
from ._cv_functions import poisson_cv


time_decorator = importlib.import_module(
    'QuadratiK.tools._utils').class_method_call_timing
stats = importlib.import_module('QuadratiK.tools').stats


class PoissonKernelTest():
    """
    Class for Poisson kernel-based quadratic distance 
    test of Uniformity on the Sphere

    Parameters
    ----------
        rho : float
            The value of concentration parameter used for the 
            Poisson kernel function.

        num_iter : int, optional
            Number of iterations for critical value estimation of U-statistic.

        quantile : float, optional
            The quantile to use for critical value estimation

        random_state : int, None, optional. 
            Seed for random number generation. Defaults to None

        n_jobs : int, optional.
            n_jobs specifies the maximum number of concurrently running workers. 
            If 1 is given, no joblib parallelism is used at all, which is useful for debugging.
            For more information on joblib n_jobs refer 
            to - https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html. 
            Defaults to 8.


    Attributes
    ----------
        test_type\\_ : str
            The type of test performed on the data

        execution_time : float
            Time taken for the test method to execute

        u_statistic_h0\\_ : boolean
            A logical value indicating whether or not the null hypothesis 
            is rejected according to Un

        u_statistic_un\\_ : float
            The value of the U-statistic.

        u_statistic_cv\\_ : float
            The empirical critical value for Un

        v_statistic_h0\\_ : boolean
            A logical value indicating whether or not the null hypothesis is 
            rejected according to Vn.

        v_statistic_vn\\_ : float
            The value of the V-statistic.

        v_statistic_cv\\_ : float
            The critical value for Vn computed following the asymptotic distribution.

    References
    -----------
        Ding Y., Markatou M., Saraceno G. (2023). “Poisson Kernel-Based Tests for
        Uniformity on the d-Dimensional Sphere.” Statistica Sinica. doi: doi:10.5705/ss.202022.0347

    Examples
    ---------
    >>> from QuadratiK.tools import sample_hypersphere
    >>> from QuadratiK.poisson_kernel_test import PoissonKernelTest
    >>> np.random.seed(42)
    >>> X = sample_hypersphere(100,3, random_state=42)
    >>> unif_test = PoissonKernelTest(rho = 0.7, random_state=42).test(X)
    >>> print("Execution time: {:.3f} seconds".format(unif_test.execution_time))
    >>> print("U Statistic Results")
    >>> print("H0 is rejected : {}".format(unif_test.u_statistic_h0_))
    >>> print("Un Statistic : {}".format(unif_test.u_statistic_un_))
    >>> print("Critical Value : {}".format(unif_test.u_statistic_cv_))
    >>> print("V Statistic Results")
    >>> print("H0 is rejected : {}".format(unif_test.v_statistic_h0_))
    >>> print("Vn Statistic : {}".format(unif_test.v_statistic_vn_))
    >>> print("Critical Value : {}".format(unif_test.v_statistic_cv_))
    ... Execution time: 0.181 seconds
    ... U Statistic Results
    ... H0 is rejected : False
    ... Un Statistic : 1.6156682048968174
    ... Critical Value : 0.06155875299050079
    ... V Statistic Results
    ... H0 is rejected : False
    ... Vn Statistic : 22.83255917641962
    ... Critical Value : 23.229486935225513
    """

    def __init__(self, rho, num_iter=300, quantile=0.95, random_state=None, n_jobs=8):
        self.rho = rho
        self.num_iter = num_iter
        self.quantile = quantile
        self.random_state = random_state
        self.n_jobs = n_jobs

    @time_decorator
    def test(self, x):
        """
        Performs the Poisson kernel-based quadratic distance Goodness-of-fit tests for
        Uniformity for spherical data using the Poisson kernel with concentration parameter :math:`rho`

        Parameters
        ----------
            x : numpy.ndarray, pandas.DataFrame
                a numeric d-dim matrix of data points on the Sphere :math:`S^{(d-1)}`.

        Returns
        -------
            self : object
                Fitted estimator
        """
        self.x = x
        if isinstance(x, np.ndarray):
            if self.x.ndim == 1:
                self.x = self.x.reshape(-1, 1)
        elif isinstance(self.x, pd.DataFrame):
            self.x = self.x.to_numpy()
        elif not isinstance(self.x, (np.ndarray, pd.DataFrame)):
            raise TypeError(
                "x must be a numpy array or a pandas dataframe")

        if (self.quantile <= 0) or (self.quantile > 1):
            raise ValueError(
                "Quantile indicates the level used for the critical \
                    value computation. It must be in (0,1].")

        if not (isinstance(self.rho, (int, float)) and (0 < self.rho <= 1)):
            raise ValueError(
                "rho indicates the concentration parameter \
                    of the Poisson kernel, it must be in (0,1).")

        if not isinstance(self.random_state, (int, type(None))):
            raise ValueError("Please specify a integer or None random_state")

        method = "Poisson Kernel-based quadratic \n distance test of Uniformity on the Sphere"

        n, d = self.x.shape
        pk = stat_poisson_unif(self.x, self.rho)
        var_un = (2 / (n * (n - 1))) * ((1 + np.power(self.rho, 2))
                                        / np.power(1 - np.power(self.rho, 2), (d - 1)) - 1)
        dof_val = dof(d, self.rho)
        qu_q = chi2.ppf(0.95, dof_val['DOF'])
        cv_vn = dof_val['Coefficient'] * qu_q
        cv_un = poisson_cv(d=d, size=n, rho=self.rho,
                           num_iter=self.num_iter, quantile=self.quantile, random_state=self.random_state, n_jobs=self.n_jobs)

        self.test_type_ = method
        self.u_statistic_h0_ = pk[0] > cv_un
        self.u_statistic_un_ = pk[0]/np.sqrt(var_un)
        self.u_statistic_cv_ = cv_un

        self.v_statistic_h0_ = pk[1] > cv_vn
        self.v_statistic_vn_ = pk[1]
        self.v_statistic_cv_ = cv_vn

        return self

    def stats(self):
        """
        Function to generate descriptive statistics.

        Returns
        -------
            summary_stats_df : pandas.DataFrame
                Dataframe of descriptive statistics
        """
        summary_stats_df = stats(self.x)
        return summary_stats_df.round(4)

    def summary(self, print_fmt="simple_grid"):
        """
        Summary function generates a table for 
        the poisson kernel test results and the summary statistics.

        Parameters
        ----------
            print_fmt : str, optional. 
                Used for printing the output in the desired format. 
                Supports all available options in tabulate, 
                see here: https://pypi.org/project/tabulate/. 
                Defaults to "simple_grid".

        Returns
        --------
            summary : str
                A string formatted in the desired output 
                format with the kernel test results and summary statistics.
        """
        res = pd.DataFrame()
        res[''] = [self.test_type_, self.u_statistic_un_, self.u_statistic_cv_,
                   self.u_statistic_h0_, self.v_statistic_vn_,
                   self.v_statistic_cv_, self.v_statistic_h0_]
        res = res.set_axis(["Test Type", "U Statistic Un", "U Statistic Critical Value",
                           "U Statistic Reject H0", "V Statistic Vn",
                            "V Statistic Critical Value", "V Statistic Reject H0"])

        summary_stats_df = self.stats().round(4)

        if print_fmt == "html":
            summary_string = (
                "Time taken for execution: {:.3f} seconds".format(
                    self.execution_time)
                + "<br>Test Results <br>"
                + tabulate(res, tablefmt=print_fmt)
                + "<br>Summary Statistics <br>"
                + tabulate(
                    summary_stats_df,
                    tablefmt=print_fmt,
                    headers=summary_stats_df.columns
                )
                + "<br>"
            )
        else:
            summary_string = (
                "Time taken for execution: {:.3f} seconds".format(
                    self.execution_time)
                + "\nTest Results \n"
                + tabulate(res, tablefmt=print_fmt)
                + "\nSummary Statistics \n"
                + tabulate(
                    summary_stats_df,
                    tablefmt=print_fmt,
                    headers=summary_stats_df.columns
                )
            )

        return summary_string
