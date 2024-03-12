"""
Kernel-based quadratic distance Goodness-of-Fit tests
"""

import importlib
import numpy as np
import pandas as pd
from tabulate import tabulate

from ._utils import stat_normality_test, stat_two_sample, stat_ksample
from ._cv_functions import cv_twosample, cv_normality, cv_ksample
from ._h_selection import select_h

time_decorator = importlib.import_module(
    'QuadratiK.tools._utils').class_method_call_timing
stats = importlib.import_module('QuadratiK.tools').stats


class KernelTest():
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
            The size of the subsamples used in the subsampling algorithm. Defaults to 0.9.

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
            Seed for random number generation. Defaults to None

        n_jobs : int, optional. 
            n_jobs specifies the maximum number of concurrently 
            running workers. If 1 is given, no joblib parallelism 
            is used at all, which is useful for debugging. For more 
            information on joblib n_jobs refer to - 
            https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html.
            Defaults to 8.

    Attributes
    ----------
        test_type\\_ : str
            The type of test performed on the data

        execution_time : float
            Time taken for the test method to execute

        h0_rejected\\_ : boolean
            Whether the null hypothesis is rejected (True) or not (False)

        test_statistic\\_ : float
            Test statistic of the perfomed test type

        cv\\_ : float
            Critical value

        cv_method\\_ : str
            Critical value method used for performing the test

    References
    -----------
        Markatou M., Saraceno G., Chen Y (2023). “Two- and k-Sample Tests Based on Quadratic Distances.
        ”Manuscript, (Department of Biostatistics, University at Buffalo)

        Lindsay BG, Markatou M. & Ray S. (2014) Kernels, Degrees of Freedom, and 
        Power Properties of Quadratic Distance Goodness-of-Fit Tests, Journal of the American Statistical
        Association, 109:505, 395-410, DOI: 10.1080/01621459.2013.836972

    Examples
    --------
    >>> # Example for normality test
    >>> import numpy as np
    >>> from QuadratiK.kernel_test import KernelTest
    >>> np.random.seed(42)
    >>> data = np.random.randn(100,5)
    >>> normality_test = KernelTest(h=0.4, centering_type="param",random_state=42).test(data)
    >>> print("Test : {}".format(normality_test.test_type_))
    >>> print("Execution time: {:.3f}".format(normality_test.execution_time))
    >>> print("H0 is Rejected : {}".format(normality_test.h0_rejected_))
    >>> print("Test Statistic : {}".format(normality_test.test_statistic_))
    >>> print("Critical Value (CV) : {}".format(normality_test.cv_))
    >>> print("CV Method : {}".format(normality_test.cv_method_))
    >>> print("Selected tuning parameter : {}".format(normality_test.h))
    ... Test : Kernel-based quadratic distance Normality test
    ... Execution time: 0.096
    ... H0 is Rejected : False
    ... Test Statistic : -8.588873037044384e-05
    ... Critical Value (CV) : 0.0004464111809800183
    ... CV Method : Empirical
    ... Selected tuning parameter : 0.4
    
    >>> # Example for two sample test
    >>> import numpy as np
    >>> from QuadratiK.kernel_test import KernelTest
    >>> np.random.seed(42)
    >>> X = np.random.randn(100,5)
    >>> np.random.seed(42)
    >>> Y = np.random.randn(100,5)
    >>> two_sample_test = KernelTest(h=0.4, centering_type="param").test(X,Y)
    >>> print("Test : {}".format(two_sample_test.test_type_))
    >>> print("Execution time: {:.3f}".format(two_sample_test.execution_time))
    >>> print("H0 is Rejected : {}".format(two_sample_test.h0_rejected_))
    >>> print("Test Statistic : {}".format(two_sample_test.test_statistic_))
    >>> print("Critical Value (CV) : {}".format(two_sample_test.cv_))
    >>> print("CV Method : {}".format(two_sample_test.cv_method_))
    >>> print("Selected tuning parameter : {}".format(two_sample_test.h))
    ... Test : Kernel-based quadratic distance two-sample test
    ... Execution time: 0.092
    ... H0 is Rejected : False
    ... Test Statistic : -0.019707895277270022
    ... Critical Value (CV) : 0.003842482597612725
    ... CV Method : subsampling
    ... Selected tuning parameter : 0.4
    """

    def __init__(self, h=None, method="subsampling", num_iter=150,
                 b=0.9, quantile=0.95, mu_hat=None, sigma_hat=None, centering_type="nonparam",
                 alternative=None, k_threshold=10, random_state=None, n_jobs=8):
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
                Fitted estimator
        """

        self.x = x
        self.y = y

        if isinstance(self.x, (np.ndarray, pd.Series)):
            if self.x.ndim == 1:
                self.x = np.array(self.x).reshape(-1, 1)
        elif isinstance(self.x, pd.DataFrame):
            self.x = self.x.to_numpy()
        else:
            raise TypeError(
                "x must be a numpy array or a pandas dataframe")

        if self.y is not None:
            if isinstance(self.y, np.ndarray):
                if self.y.ndim == 1:
                    self.y = self.y.reshape(-1, 1)
            elif isinstance(y, pd.DataFrame):
                self.y = self.y.to_numpy()
            else:
                raise TypeError(
                    "y must be a numpy array or a pandas dataframe")

        valid_methods = ["bootstrap", "permutation", "subsampling"]
        if self.method not in valid_methods:
            raise ValueError(
                "method must be one of 'bootstrap', 'permutation', or 'subsampling'")

        if self.b is not None:
            if not 0 < self.b <= 1:
                raise ValueError(
                    "b indicates the proportion used for the \
                        subsamples in the subsampling algorithm. It must be in (0, 1].")

        valid_centering_type = ["param", "nonparam"]
        if self.centering_type not in valid_centering_type:
            raise ValueError(
                "centering must be chosen between 'param' and 'nonparam'")

        if self.h is None:
            if self.alternative is None:
                raise ValueError(
                    "You have not specified a value of h. "
                    "Please specify a alternative from 'location', 'scale' or 'skewness' "
                    "for the tuning parameter selection algorithm to run")
        
        if not isinstance(self.random_state,(int,type(None))):
            raise ValueError("Please specify a integer or None random_state") 

        size_x, k = self.x.shape

        if self.y is None:
            if self.h is None:
                self.h = select_h(
                    self.x, y=None, alternative=self.alternative, num_iter=self.num_iter,
                    quantile=self.quantile, n_jobs=self.n_jobs)[0]

            # Compute the estimates of mean and covariance from the data
            if self.mu_hat is None:
                self.mu_hat = np.mean(self.x, axis=0, keepdims=True)
            if self.sigma_hat is None:
                self.sigma_hat = np.cov(self.x, rowvar=False)
                if k == 1:
                    self.sigma_hat = np.array([[np.take(self.sigma_hat, 0)]])

            statistic = stat_normality_test(
                self.x, self.h, self.mu_hat, self.sigma_hat, self.centering_type)
            cv = cv_normality(size_x, self.h, self.mu_hat, self.sigma_hat,
                              self.num_iter, self.quantile, self.centering_type, self.random_state, self.n_jobs)
            h0 = statistic > cv

            self.test_type_ = "Kernel-based quadratic distance Normality test"
            self.h0_rejected_ = h0
            self.test_statistic_ = statistic
            self.cv_ = cv
            self.cv_method_ = "Empirical"
            return self

        else:
            k = len(np.unique(y))
            if k > self.k_threshold:
                if (self.y is not None) and (self.x.shape[1] != self.y.shape[1]):
                    raise ValueError(
                        "'x' and 'y' must have the same number of columns.")

                size_y = self.y.shape[0]
                data_pool = np.vstack((self.x, self.y))

                if self.h is None:
                    self.h = select_h(self.x, self.y, alternative=self.alternative,
                                      quantile=self.quantile, method=self.method,
                                      num_iter=self.num_iter, n_jobs=self.n_jobs)[0]

                if self.centering_type == "param":
                    # Compute the estimates of mean and covariance from the data
                    if self.mu_hat is None:
                        self.mu_hat = np.mean(data_pool, axis=0, keepdims=True)
                    if self.sigma_hat is None:
                        self.sigma_hat = np.cov(data_pool, rowvar=False)

                    statistic = stat_two_sample(
                        self.x, self.y, self.h, self.mu_hat, self.sigma_hat, "param")

                elif self.centering_type == "nonparam":
                    statistic = stat_two_sample(self.x, self.y, self.h, np.array(
                        [[0]]), np.array([[1]]), "nonparam")

                cv = cv_twosample(self.num_iter, self.quantile, data_pool, size_x,
                                  size_y, self.h, self.method, self.b, self.random_state, self.n_jobs)
                h0 = statistic > cv

                self.test_type_ = "Kernel-based quadratic distance two-sample test"
                self.h0_rejected_ = h0
                self.test_statistic_ = statistic
                self.cv_ = cv
                self.cv_method_ = self.method
                return self

            else:
                if (self.y is not None) and (self.x.shape[0] != self.y.shape[0]):
                    raise ValueError(
                        "'x' and 'y' must have the same number of rows.")

                if self.h is None:
                    self.h = select_h(self.x, y=None, alternative=self.alternative,
                                      method=self.method, num_iter=self.num_iter,
                                      quantile=self.quantile, n_jobs=self.n_jobs)[0]

                statistic = stat_ksample(self.x, self.y, self.h)
                cv = cv_ksample(self.x, self.y, self.h, self.num_iter, self.b,
                                self.quantile, self.method, self.random_state, self.n_jobs)
                h0 = statistic[0] > cv[0]

                self.test_type_ = "Kernel-based quadratic distance K-sample test"
                self.h0_rejected_ = h0
                self.test_statistic_ = statistic
                self.cv_ = cv
                self.cv_method_ = self.method
                return self

    def stats(self):
        """
        Function to generate descriptive statistics per variable (and per group if available).

        Returns
        -------
            summary_stats_df : pandas.DataFrame
                Dataframe of descriptive statistics
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
                Supports all available options in tabulate, see here: https://pypi.org/project/tabulate/ 

        Returns
        --------
            summary : str
                A string formatted in the desired output 
                format with the kernel test results and summary statistics.
        """
        res = pd.DataFrame()
        res[''] = [self.test_type_, self.test_statistic_,
                   self.cv_, self.h0_rejected_]
        res = res.set_axis(["Test Type", "Test Statistic",
                           "Critical Value", "Reject H0"])

        summary_stats_df = self.stats()

        if print_fmt == "html":
            summary_string = (
                "Time taken for execution: {} seconds".format(
                    self.execution_time)
                + "<br>Test Results <br>"
                + tabulate(res, tablefmt=print_fmt)
                + "<br>Summary Statistics <br>"
                + tabulate(
                    summary_stats_df,
                    tablefmt=print_fmt,
                    headers=summary_stats_df.columns,
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
                    headers=summary_stats_df.columns,
                )
            )

        return summary_string
