"""
Kernel-based quadratic distance Goodness-of-Fit tests
"""

import importlib
import numpy as np
import pandas as pd
from tabulate import tabulate
from scipy.stats import chi2

from ._utils import (
    stat_normality_test,
    stat_two_sample,
    stat_ksample,
    dof_normality_test,
    variance_normality_test,
)
from ._cv_functions import cv_twosample, cv_normality, cv_ksample
from ._h_selection import select_h

time_decorator = importlib.import_module(
    "QuadratiK.tools._utils"
).class_method_call_timing
stats = importlib.import_module("QuadratiK.tools").stats


class KernelTest:
    """
    Class for performing the kernel-based quadratic distance goodness-of-fit tests using
    the Gaussian kernel with tuning parameter h. Depending on the input `y` the function performs
    the test of multivariate normality, the non-parametric two-sample tests or the k-sample tests.

    Parameters
    ----------
        h : float, optional
            Bandwidth for the kernel function.

        method : str, optional
            The method used for critical value estimation ("subsampling", "bootstrap",
            or "permutation").

        num_iter : int, optional
            The number of iterations to use for critical value estimation. Defaults to 150.

        b : float, optional
            The size of the subsamples used in the subsampling algorithm. Defaults to 0.9 i.e.
            `0.9N` samples are used, where `N` represents the total sample size.

        quantile : float, optional
            The quantile to use for critical value estimation. Defaults to 0.95.

        mu_hat : numpy.ndarray, optional
            Mean vector for the reference distribution. Defaults to None.

        sigma_hat : numpy.ndarray, optional
            Covariance matrix of the reference distribution. Defaults to None.

        alternative : str, optional
            String indicating the type of alternative to be used for calculating "h"
            by the tuning parameter selection algorithm when h is not provided.
            Defaults to 'None'

        k_threshold : int, optional
            Maximum number of groups allowed. Defaults to 10. Change in case of more than 10 groups.

        random_state : int, None, optional.
            Seed for random number generation. Defaults to None.

        n_jobs : int, optional.
            n_jobs specifies the maximum number of concurrently
            running workers. If 1 is given, no joblib parallelism
            is used at all, which is useful for debugging. For more
            information on joblib n_jobs refer to -
            https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html.
            Defaults to 8.

    Attributes
    ----------
        For Normality Test:
            test_type\\_ : str
                The type of test performed on the data.

            execution_time : float
                Time taken for the test method to execute.

            un_h0_rejected\\_ : boolean
                Whether the null hypothesis using Un is rejected (True) or not (False).

            vn_h0_rejected\\_ : boolean
                Whether the null hypothesis using Vn is rejected (True) or not (False).

            un_test_statistic\\_ : float
                Un Test statistic of the perfomed test type.

            vn_test_statistic\\_ : float
                Vn Test statistic of the perfomed test type.

            un_cv\\_ : float
                Critical value for Un.

            un_cv\\_ : float
                Critical value for Vn.

        For Two-Sample and K-Sample Test:
            test_type\\_ : str
                The type of test performed on the data.

            execution_time : float
                Time taken for the test method to execute.

            dn_h0_rejected\\_ : boolean
                Whether the null hypothesis using Un is rejected (True) or not (False).

            dn_test_statistic\\_ : float
                Un Test statistic of the perfomed test type.

            dn_cv\\_ : float
                Critical value for Un.

            trace_h0_rejected\\_ : boolean
                Whether the null hypothesis using trace statistic is rejected (True) or not (False).

            trace_test_statistic\\_ : float
                Trace Test statistic of the perfomed test type.

            trace_cv\\_ : float
                Critical value for trace statistic.

            cv_method\\_ : str
                Critical value method used for performing the test.

    References
    -----------
        Markatou M., Saraceno G., Chen Y (2023). “Two- and k-Sample Tests Based on Quadratic Distances.
        ”Manuscript, (Department of Biostatistics, University at Buffalo).

        Lindsay BG, Markatou M. & Ray S. (2014) Kernels, Degrees of Freedom, and
        Power Properties of Quadratic Distance Goodness-of-Fit Tests, Journal of the American Statistical
        Association, 109:505, 395-410, DOI: 10.1080/01621459.2013.836972.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(78990)
    >>> from QuadratiK.kernel_test import KernelTest
    >>> # data generation
    >>> data_norm = np.random.multivariate_normal(mean = np.zeros(4), cov = np.eye(4),size = 500)
    >>> # performing the normality test
    >>> normality_test = KernelTest(h=0.4, num_iter=150, method= "subsampling", random_state=42).test(data_norm)
    >>> print(normality_test)
    ... KernelTest(
        Test Type=Kernel-based quadratic distance Normality test,
        Execution Time=3.198080062866211 seconds,
        U-Statistic=-0.1145936604628874,
        U-Statistic Critical Value=1.1593122985543514,
        U-Statistic Null Hypothesis Rejected=False,
        U-Statistic Variance=1.108021332522181e-08,
        V-Statistic=0.977955027161687,
        V-Statistic Critical Value=42.460022848761945,
        V-Statistic Null Hypothesis Rejected=False,
        Selected tuning parameter h=0.4
        )

    >>> import numpy as np
    >>> np.random.seed(0)
    >>> from scipy.stats import skewnorm
    >>> from QuadratiK.kernel_test import KernelTest
    >>> # data generation
    >>> X_2 = np.random.multivariate_normal(mean = np.zeros(4), cov = np.eye(4), size=200)
    >>> Y_2 = skewnorm.rvs(size=(200, 4),loc=np.zeros(4), scale=np.ones(4),a=np.repeat(0.5,4), random_state=20)
    >>> # performing the two sample test
    >>> two_sample_test = KernelTest(h = 2,num_iter = 150, random_state=42).test(X_2,Y_2)
    >>> print(two_sample_test)
    ... KernelTest(
        Test Type=Kernel-based quadratic distance two-sample test,
        Execution Time=0.36570000648498535 seconds,
        Dn-Statistic=5.061212999055004,
        Dn-Statistic Critical Value=0.4901155246432661,
        Dn-Statistic Null Hypothesis Rejected=True,
        Dn-Statistic Variance=3.037711857184588e-10,
        Trace-Statistic=15.751718163734266,
        Trace-Statistic Critical Value=1.525782865913332,
        Trace-Statistic Null Hypothesis Rejected=True,
        Trace-Statistic Variance=7.879780877050946e-12,
        Selected tuning parameter h=2,
        Critical Value Method=subsampling
        )
    """

    def __init__(
        self,
        h=None,
        method="subsampling",
        num_iter=150,
        b=0.9,
        quantile=0.95,
        mu_hat=None,
        sigma_hat=None,
        centering_type="nonparam",
        alternative=None,
        k_threshold=10,
        random_state=None,
        n_jobs=8,
    ):
        self.h = h
        self.method = method
        self.num_iter = num_iter
        self.b = b
        self.quantile = quantile
        self.mu_hat = mu_hat
        self.sigma_hat = sigma_hat
        self.centering_type = centering_type
        self.alternative = alternative
        self.k_threshold = k_threshold
        self.n_jobs = n_jobs
        self.random_state = random_state

    def __repr__(self):
        if self.vn_test_statistic_ is not None:
            return (
                f"{self.__class__.__name__}(\n"
                f"  Test Type={self.test_type_},\n"
                f"  Execution Time={self.execution_time} seconds,\n"
                f"  U-Statistic={self.un_test_statistic_},\n"
                f"  U-Statistic Critical Value={self.un_cv_},\n"
                f"  U-Statistic Null Hypothesis Rejected={self.un_h0_rejected_},\n"
                f"  U-Statistic Variance={self.var_un_},\n"
                f"  V-Statistic={self.vn_test_statistic_},\n"
                f"  V-Statistic Critical Value={self.vn_cv_},\n"
                f"  V-Statistic Null Hypothesis Rejected={self.vn_h0_rejected_},\n"
                f"  Selected tuning parameter h={self.h}\n"
                f")"
            )
        else:
            return (
                f"{self.__class__.__name__}(\n"
                f"  Test Type={self.test_type_},\n"
                f"  Execution Time={self.execution_time} seconds,\n"
                f"  Dn-Statistic={self.dn_test_statistic_},\n"
                f"  Dn-Statistic Critical Value={self.dn_cv_},\n"
                f"  Dn-Statistic Null Hypothesis Rejected={self.dn_h0_rejected_},\n"
                f"  Dn-Statistic Variance={self.var_dn_},\n"
                f"  Trace-Statistic={self.trace_test_statistic_},\n"
                f"  Trace-Statistic Critical Value={self.trace_cv_},\n"
                f"  Trace-Statistic Null Hypothesis Rejected={self.trace_h0_rejected_},\n"
                f"  Trace-Statistic Variance={self.var_trace_},\n"
                f"  Selected tuning parameter h={self.h},\n"
                f"  Critical Value Method={self.cv_method_}\n"
                f")"
            )

    @time_decorator
    def test(self, x, y=None):
        """
        Function to perform the kernel-based quadratic distance tests using
        the Gaussian kernel with bandwidth parameter h.
        Depending on the shape of the `y`, the function performs the tests of
        multivariate normality, the non-parametric two-sample tests or the k-sample tests.

        Parameters
        ----------
            x : numpy.ndarray or pandas.DataFrame.
                A numeric array of data values.
            y : numpy.ndarray or pandas.DataFrame, optional
                A numeric array data values (for two-sample test) and a 1D array of class labels
                (for k-sample test). Defaults to None.

        Returns
        -------
            self : object
                Fitted estimator.
        """

        self.x = x
        self.y = y

        if isinstance(self.x, (np.ndarray, pd.Series)):
            if self.x.ndim == 1:
                self.x = np.array(self.x).reshape(-1, 1)
        elif isinstance(self.x, pd.DataFrame):
            self.x = self.x.to_numpy()
        else:
            raise TypeError("x must be a numpy array or a pandas dataframe")

        if self.y is not None:
            if isinstance(self.y, (np.ndarray, pd.Series)):
                if self.y.ndim == 1:
                    self.y = np.array(self.y).reshape(-1, 1)
            elif isinstance(y, pd.DataFrame):
                self.y = self.y.to_numpy()
            else:
                raise TypeError("y must be a numpy array or a pandas dataframe")

        valid_methods = ["bootstrap", "permutation", "subsampling"]
        if self.method not in valid_methods:
            raise ValueError(
                "method must be one of 'bootstrap', 'permutation', or 'subsampling'"
            )

        if self.b is not None:
            if not 0 < self.b <= 1:
                raise ValueError(
                    "b indicates the proportion used for the \
                        subsamples in the subsampling algorithm. It must be in (0, 1]."
                )

        valid_centering_type = ["param", "nonparam"]
        if self.centering_type not in valid_centering_type:
            raise ValueError("centering must be chosen between 'param' and 'nonparam'")

        if self.h is None:
            if self.alternative is None:
                raise ValueError(
                    "You have not specified a value of h. "
                    "Please specify a alternative from 'location', 'scale' or 'skewness' "
                    "for the tuning parameter selection algorithm to run"
                )

        if not isinstance(self.random_state, (int, type(None))):
            raise ValueError("Please specify a integer or None random_state")

        size_x, k = self.x.shape

        if self.y is None:
            if self.h is None:
                self.h = select_h(
                    self.x,
                    y=None,
                    alternative=self.alternative,
                    num_iter=self.num_iter,
                    quantile=self.quantile,
                    n_jobs=self.n_jobs,
                )[0]

            if self.mu_hat is None:
                self.mu_hat = np.zeros(k).reshape(1, -1)
            else:
                self.x = self.x - self.mu_hat
                self.mu_hat = np.zeros(k)

            if self.sigma_hat is None:
                self.sigma_hat = np.eye(k)

            statistic = stat_normality_test(self.x, self.h, self.mu_hat, self.sigma_hat)

            sigma_h = (self.h**2) * np.eye(k)

            var_un = variance_normality_test(sigma_h, self.sigma_hat, size_x)
            cv_un = cv_normality(
                size_x,
                self.h,
                self.mu_hat,
                self.sigma_hat,
                self.num_iter,
                self.quantile,
                self.random_state,
                self.n_jobs,
            ) / np.sqrt(var_un)

            dof, coeff = dof_normality_test(sigma_h, self.sigma_hat)
            qu_q = chi2.ppf(self.quantile, dof)
            cv_vn = coeff * qu_q

            un_h0 = (statistic[0] / np.sqrt(var_un)) > cv_un
            vn_h0 = statistic[1] > cv_vn

            self.test_type_ = "Kernel-based quadratic distance Normality test"
            self.un_h0_rejected_ = un_h0
            self.vn_h0_rejected_ = vn_h0
            self.un_test_statistic_ = statistic[0] / np.sqrt(var_un)
            self.vn_test_statistic_ = statistic[1]
            self.un_cv_ = cv_un
            self.vn_cv_ = cv_vn
            self.var_un_ = var_un
            self.cv_method_ = None
            return self

        else:
            k = len(np.unique(y))
            if k > self.k_threshold:
                if (self.y is not None) and (self.x.shape[1] != self.y.shape[1]):
                    raise ValueError(
                        "'x' and 'y' must have the same number of columns."
                    )

                size_y = self.y.shape[0]
                data_pool = np.vstack((self.x, self.y))

                if self.h is None:
                    self.h = select_h(
                        self.x,
                        self.y,
                        alternative=self.alternative,
                        quantile=self.quantile,
                        method=self.method,
                        num_iter=self.num_iter,
                        n_jobs=self.n_jobs,
                    )[0]

                if self.centering_type == "param":
                    # Compute the estimates of mean and covariance from the data
                    if self.mu_hat is None:
                        self.mu_hat = np.mean(data_pool, axis=0, keepdims=True)
                    if self.sigma_hat is None:
                        self.sigma_hat = np.cov(data_pool, rowvar=False)

                    statistic = stat_two_sample(
                        self.x, self.y, self.h, self.mu_hat, self.sigma_hat, "param"
                    )

                elif self.centering_type == "nonparam":
                    statistic = stat_two_sample(
                        self.x,
                        self.y,
                        self.h,
                        np.zeros(k),
                        np.eye(k),
                        "nonparam",
                    )

                cv = cv_twosample(
                    self.num_iter,
                    self.quantile,
                    data_pool,
                    size_x,
                    size_y,
                    self.h,
                    self.method,
                    self.b,
                    self.random_state,
                    self.n_jobs,
                )

                h0 = statistic[:2] / np.sqrt(statistic[2:]) > cv / np.sqrt(
                    statistic[2:]
                )

                self.test_type_ = "Kernel-based quadratic distance two-sample test"
                self.dn_h0_rejected_, self.trace_h0_rejected_ = h0
                self.dn_test_statistic_, self.trace_test_statistic_ = statistic[
                    :2
                ] / np.sqrt(statistic[2:])
                self.dn_cv_, self.trace_cv_ = cv / np.sqrt(statistic[2:])
                self.cv_method_ = self.method
                self.var_dn_, self.var_trace_ = statistic[2:]
                self.vn_test_statistic_ = None
                self.vn_cv_ = None
                return self

            else:
                if (self.y is not None) and (self.x.shape[0] != self.y.shape[0]):
                    raise ValueError("'x' and 'y' must have the same number of rows.")

                if self.h is None:
                    self.h = select_h(
                        self.x,
                        self.y,
                        alternative=self.alternative,
                        method=self.method,
                        num_iter=self.num_iter,
                        quantile=self.quantile,
                        n_jobs=self.n_jobs,
                    )[0]

                statistic = stat_ksample(self.x, self.y, self.h)
                cv = cv_ksample(
                    self.x,
                    self.y,
                    self.h,
                    self.num_iter,
                    self.b,
                    self.quantile,
                    self.method,
                    self.random_state,
                    self.n_jobs,
                )

                h0 = statistic[:2] / np.sqrt(statistic[2:]) > cv / np.sqrt(
                    statistic[2:]
                )

                self.test_type_ = "Kernel-based quadratic distance K-sample test"
                self.dn_h0_rejected_, self.trace_h0_rejected_ = h0
                self.dn_test_statistic_, self.trace_test_statistic_ = statistic[
                    :2
                ] / np.sqrt(statistic[2:])
                self.dn_cv_, self.trace_cv_ = cv / np.sqrt(statistic[2:])
                self.cv_method_ = self.method
                self.var_dn_, self.var_trace_ = statistic[2:]
                self.vn_test_statistic_ = None
                self.vn_cv_ = None
                return self

    def stats(self):
        """
        Function to generate descriptive statistics per variable (and per group if available).

        Returns
        -------
            summary_stats_df : pandas.DataFrame
                Dataframe of descriptive statistics.
        """
        summary_stats_df = stats(self.x, self.y)
        return summary_stats_df.round(4)

    def summary(self, print_fmt="simple_grid"):
        """
        Summary function generates a table for the kernel test results and the summary statistics.

        Parameters
        ----------
            print_fmt : str, optional.
                Used for printing the output in the desired format. Defaults to "simple_grid".
                Supports all available options in tabulate, see here: https://pypi.org/project/tabulate/.

        Returns
        --------
            summary : str
                A string formatted in the desired output
                format with the kernel test results and summary statistics.
        """

        if self.vn_test_statistic_ is None:

            index_labels = [
                "Test Statistic",
                "Critical Value",
                "H0 is rejected (1 = True, 0 = False)",
            ]
            test_summary = {
                "Dn": [self.dn_test_statistic_, self.dn_cv_, self.dn_h0_rejected_],
                "Trace": [
                    self.trace_test_statistic_,
                    self.trace_cv_,
                    self.trace_h0_rejected_,
                ],
            }
            res = pd.DataFrame(test_summary, index=index_labels)

        else:
            index_labels = [
                "Test Statistic",
                "Critical Value",
                "H0 is rejected (1 = True, 0 = False)",
            ]
            test_summary = {
                "U-Statistic": [
                    self.un_test_statistic_,
                    self.un_cv_,
                    self.un_h0_rejected_,
                ],
                "V-Statistic": [
                    self.vn_test_statistic_,
                    self.vn_cv_,
                    self.vn_h0_rejected_,
                ],
            }
            res = pd.DataFrame(test_summary, index=index_labels)

        summary_stats_df = self.stats()

        if print_fmt == "html":
            summary_string = (
                "Time taken for execution: {} seconds".format(self.execution_time)
                + "<br>Test Results <br>"
                + f"<br> {self.test_type_} <br>"
                + tabulate(res, tablefmt=print_fmt, headers=res.columns)
                + f"<br> CV method: {self.cv_method_} <br>"
                + f"<br> Selected tuning parameter h : {self.h} <br>"
                + "<br> Summary Statistics <br>"
                + tabulate(
                    summary_stats_df,
                    tablefmt=print_fmt,
                    headers=summary_stats_df.columns,
                )
                + "<br>"
            )
        else:
            summary_string = (
                "Time taken for execution: {:.3f} seconds".format(self.execution_time)
                + "\nTest Results \n"
                + f"{self.test_type_} \n"
                + tabulate(res, tablefmt=print_fmt, headers=res.columns)
                + f"\nCV method: {self.cv_method_} \n"
                + f"Selected tuning parameter h : {self.h} \n"
                + "\nSummary Statistics \n"
                + tabulate(
                    summary_stats_df,
                    tablefmt=print_fmt,
                    headers=summary_stats_df.columns,
                )
            )

        return summary_string
