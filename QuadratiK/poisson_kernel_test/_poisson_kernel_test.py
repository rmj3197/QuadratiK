"""
Poisson kernel-based quadratic distance test of Uniformity on the Sphere
"""

import importlib

import numpy as np
import pandas as pd
from scipy.stats import chi2
from tabulate import tabulate

from ._cv_functions import poisson_cv
from ._utils import dof, stat_poisson_unif

time_decorator = importlib.import_module(
    "QuadratiK.tools._utils"
).class_method_call_timing
stats = importlib.import_module("QuadratiK.tools").stats


class PoissonKernelTest:
    r"""
    Class for Poisson kernel-based quadratic distance tests
    of Uniformity on the Sphere. More details can be found in the
    :ref:`User Guide <pktest>`.

    Parameters
    ----------
    rho : float
        The value of concentration parameter used for the
        Poisson kernel function.

    num_iter : int, optional
        Number of iterations for critical value estimation of Tn-statistic.

    quantile : float, optional
        The quantile to use for critical value estimation.

    random_state : int, None, optional.
        Seed for random number generation. Defaults to None.

    n_jobs : int, optional.
        n_jobs specifies the maximum number of concurrently running workers.
        If 1 is given, no joblib parallelism is used at all, which is useful for debugging.
        For more information on joblib n_jobs refer
        to - https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html.
        Defaults to 8.

    Attributes
    -----------
    test_type\_ : str
        The type of test performed on the data.

    execution_time : float
        Time taken for the test method to execute.

    u_statistic_h0_ : boolean
        A logical value indicating whether or not the null hypothesis
        is rejected according to Tn.

    u_statistic_tn_ : float
        The value of the Tn-statistic.

    u_statistic_cv_ : float
        The empirical critical value for Tn.

    v_statistic_h0_ : boolean
        A logical value indicating whether or not the null hypothesis is
        rejected according to Sn.

    v_statistic_sn_ : float
        The value of the Sn-statistic.

    v_statistic_cv_ : float
        The critical value for Sn computed following the asymptotic distribution.


    Note
    -----
    A Tn-statistic is a type of U-statistic that is used to estimate a population parameter.
    It is based on the idea of averaging over all possible distinct combinations of a fixed size from a sample.
    A Sn-statistic is a type of V-statistic that considers all possible tuples of a certain size, not just distinct combinations and can be used in contexts
    where unbiasedness is not required.

    References
    -----------
    Ding Y., Markatou M., Saraceno G. (2023). “Poisson Kernel-Based Tests for
    Uniformity on the d-Dimensional Sphere.” Statistica Sinica. doi: doi:10.5705/ss.202022.0347.

    Examples
    ---------
    .. jupyter-execute::

        import numpy as np
        np.random.seed(0)
        from QuadratiK.poisson_kernel_test import PoissonKernelTest
        # data generation
        z = np.random.normal(size=(200, 3))
        data_unif = z / np.sqrt(np.sum(z**2, axis=1, keepdims=True))
        #performing the uniformity test
        unif_test = PoissonKernelTest(rho = 0.7, random_state=42).test(data_unif)
        print(unif_test)
    """

    __slots__ = (
        "execution_time",
        "n_jobs",
        "num_iter",
        "quantile",
        "random_state",
        "rho",
        "test_type_",
        "u_statistic_cv_",
        "u_statistic_h0_",
        "u_statistic_tn_",
        "v_statistic_cv_",
        "v_statistic_h0_",
        "v_statistic_sn_",
        "x",
    )

    def __init__(
        self,
        rho: float,
        num_iter: int = 300,
        quantile: float = 0.95,
        random_state: int | None = None,
        n_jobs: int = 8,
    ) -> None:
        self.rho = rho
        self.num_iter = num_iter
        self.quantile = quantile
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.x = None
        self.test_type_ = None
        self.execution_time = None
        self.u_statistic_tn_ = None
        self.u_statistic_cv_ = None
        self.u_statistic_h0_ = None
        self.v_statistic_sn_ = None
        self.v_statistic_cv_ = None
        self.v_statistic_h0_ = None

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  Test Type={self.test_type_},\n"
            f"  Execution Time={self.execution_time} seconds,\n"
            f"  Tn-Statistic={self.u_statistic_tn_},\n"
            f"  Tn-Statistic Critical Value={self.u_statistic_cv_},\n"
            f"  Tn-Statistic Null Hypothesis Rejected={self.u_statistic_h0_},\n"
            f"  Sn-Statistic={self.v_statistic_sn_},\n"
            f"  Sn-Statistic Critical Value={self.v_statistic_cv_},\n"
            f"  Sn-Statistic Null Hypothesis Rejected={self.v_statistic_h0_},\n"
            f"  Selected concentration parameter rho={self.rho},\n"
            f")"
        )

    @time_decorator
    def test(self, x: np.ndarray | pd.DataFrame) -> "PoissonKernelTest":
        """
        Performs the Poisson kernel-based quadratic distance Goodness-of-fit tests for
        Uniformity for spherical data using the Poisson kernel with concentration parameter :math:`rho`.

        Parameters
        ----------
        x : numpy.ndarray, pandas.DataFrame
            a numeric d-dim matrix of data points on the Sphere :math:`S^{(d-1)}`.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self.x = x
        if isinstance(x, np.ndarray):
            if self.x.ndim == 1:
                self.x = self.x.reshape(-1, 1)
        elif isinstance(self.x, pd.DataFrame):
            self.x = self.x.to_numpy()
        elif not isinstance(self.x, (np.ndarray, pd.DataFrame)):
            raise TypeError("x must be a numpy array or a pandas dataframe")

        if (self.quantile <= 0) or (self.quantile > 1):
            raise ValueError("Quantile indicates the level used for the critical \
                    value computation. It must be in (0,1].")

        if not (isinstance(self.rho, (int, float)) and (0 < self.rho <= 1)):
            raise ValueError("rho indicates the concentration parameter \
                    of the Poisson kernel, it must be in (0,1).")

        if not isinstance(self.random_state, (int, type(None))):
            raise ValueError("Please specify a integer or None random_state")

        method = (
            "Poisson Kernel-based quadratic distance test of Uniformity on the Sphere"
        )

        n, d = self.x.shape
        pk = stat_poisson_unif(self.x, self.rho)
        var_un = (2 / (n * (n - 1))) * (
            (1 + np.power(self.rho, 2)) / np.power(1 - np.power(self.rho, 2), (d - 1))
            - 1
        )
        dof_val = dof(d, self.rho)
        qu_q = chi2.ppf(0.95, dof_val["DOF"])
        cv_vn = dof_val["Coefficient"] * qu_q
        cv_un = poisson_cv(
            d=d,
            size=n,
            rho=self.rho,
            num_iter=self.num_iter,
            quantile=self.quantile,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        ) / np.sqrt(var_un)

        self.test_type_ = method
        self.u_statistic_h0_ = (pk[0] / np.sqrt(var_un)) > cv_un
        self.u_statistic_tn_ = pk[0] / np.sqrt(var_un)
        self.u_statistic_cv_ = cv_un

        self.v_statistic_h0_ = pk[1] > cv_vn
        self.v_statistic_sn_ = pk[1]
        self.v_statistic_cv_ = cv_vn

        return self

    def stats(self) -> pd.DataFrame:
        """
        Function to generate descriptive statistics.

        Returns
        -------
        summary_stats_df : pandas.DataFrame
            Dataframe of descriptive statistics.
        """
        summary_stats_df = stats(self.x)
        return summary_stats_df.round(4)

    def summary(self, print_fmt: str = "simple_grid") -> str:
        """
        Summary function generates a table for
        the Poisson kernel test results and the summary statistics.

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
        index_labels = [
            "Test Statistic",
            "Critical Value",
            "H0 is rejected (1 = True, 0 = False)",
        ]
        test_summary = {
            "Tn-Statistic": [
                self.u_statistic_tn_,
                self.u_statistic_cv_,
                self.u_statistic_h0_,
            ],
            "Sn-Statistic": [
                self.v_statistic_sn_,
                self.v_statistic_cv_,
                self.v_statistic_h0_,
            ],
        }
        res = pd.DataFrame(test_summary, index=index_labels)

        summary_stats_df = self.stats().round(4)

        if print_fmt == "html":
            summary_string = (
                f"Time taken for execution: {self.execution_time} seconds"
                + "<br>Test Results <br>"
                + f"<br>{self.test_type_} <br>"
                + tabulate(res, tablefmt=print_fmt, headers=res.columns)
                + f"<br>Concentration parameter rho: {self.rho}<br>"
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
                f"Time taken for execution: {self.execution_time:.3f} seconds"
                + "\nTest Results \n"
                + f"{self.test_type_} \n"
                + tabulate(res, tablefmt=print_fmt, headers=res.columns)
                + f"\nConcentration parameter rho: {self.rho}\n"
                + "\nSummary Statistics \n"
                + tabulate(
                    summary_stats_df,
                    tablefmt=print_fmt,
                    headers=summary_stats_df.columns,
                )
            )

        return summary_string
