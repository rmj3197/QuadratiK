"""
The PKBD class provides methods for estimating the density and generating samples from the
Poisson-kernel based distribution (PKBD).
"""

import numpy as np
import pandas as pd
import scipy.special as sp
from scipy.optimize import root_scalar
from scipy.stats import vonmises_fisher
from sklearn.utils.validation import check_random_state

from ._utils import c_d_lambda


class PKBD:
    """
    Class for estimating density and generating samples of Poisson-kernel based distribution (PKBD).
    """

    def __init__(self):
        pass

    def dpkb(self, x, mu, rho, logdens=False):
        """
        Function for estimating the density function of the PKB distribution.

        Parameters
        ----------
            x : numpy.ndarray, pandas.DataFrame
                A matrix with a number of columns >= 2.
            mu : float
                Location parameter with the same length as the rows of x. Normalized to length one.
            rho : float
                Concentration parameter. :math:`\\rho \\in (0,1]`.
            logdens : bool, optional
                If True, densities d are given as :math:`\\log(d)`. Defaults to False.

        Returns
        -------
            density : numpy.ndarray
                An array with the evaluated density values.
        """
        if len(mu) < 2:
            raise ValueError("mu must have length >= 2")
        if isinstance(x, pd.DataFrame):
            x = x.to_numpy()

        if x.ndim == 1:
            raise ValueError("vectors must have length >= 2")
        else:
            p = x.shape[1]
            if p < 2:
                raise ValueError("vectors must have length >= 2")

        if len(mu) != p:
            raise ValueError("vectors and mu must have the same length")
        if (rho >= 1) or (rho < 0):
            raise ValueError("Input argument rho must be within [0,1)")
        if np.linalg.norm(mu, ord=1) == 0:
            raise ValueError("Input argument mu cannot be a vector of zeros")

        mu = mu / np.linalg.norm(mu)
        x = x / np.linalg.norm(x, axis=1, keepdims=True)

        logretval = (
            np.log(1 - rho**2)
            - np.log(2)
            - p / 2 * np.log(np.pi)
            + sp.gammaln(p / 2)
            - p / 2 * np.log(1 + rho**2 - 2 * rho * (x @ mu))
        )

        if logdens:
            density = logretval
        else:
            density = np.exp(logretval)

        return density

    def rpkb(self, n, mu, rho, method="rejvmf", random_state=None):
        """
        Function for generating a random sample from PKBD.
        The number of observation generated is determined by `n`.

        Parameters
        ----------
            n : int
                Sample size.

            mu : float
                Location parameter with the same length as the quantiles.

            rho : float
                Concentration parameter. :math:`\\rho \\in (0,1]`.

            method : str, optional
                String that indicates the method used for sampling observations.
                The available methods are :\n
                - 'rejvmf': acceptance-rejection algorithm using von Mises-Fisher envelops.
                    (Algorithm in Table 2 of Golzy and Markatou 2020);
                - 'rejacg': using angular central Gaussian envelops.
                    (Algorithm in Table 1 of Sablica et al. 2023);

                Defaults to 'rejvmf'.

            random_state : int, None, optional.
                Seed for random number generation. Defaults to None.

        Returns
        -------
            samples : numpy.ndarray
                Generated observations from a poisson kernel-based density.
                This function returns a list with the matrix of generated observations, the
                number of tries and the number of acceptance.

        References
        -----------
            Golzy M. & Markatou M. (2020) Poisson Kernel-Based
            Clustering on the Sphere: Convergence Properties, Identifiability,
            and a Method of Sampling, Journal of Computational and Graphical Statistics,
            29:4, 758-770, DOI: 10.1080/10618600.2020.1740713.

            Sablica L., Hornik K., Leydold J. "Efficient sampling from the PKBD
            distribution," Electronic Journal of Statistics, 17(2), 2180-2209, (2023).

        Examples
        --------
        >>> from QuadratiK.spherical_clustering import PKBD
        >>> pkbd_data = PKBD().rpkb(10,[0.5,0],0.5, "rejvmf", random_state= 42)
        >>> dens_val  = PKBD().dpkb(pkbd_data, [0.5,0.5],0.5)
        >>> print(dens_val)
        ... [0.46827108 0.05479605 0.21163936 0.06195099 0.39567698 0.40473724
        ...     0.26561508 0.36791766 0.09324676 0.46847274]
        """

        if (rho >= 1) or (rho < 0):
            raise ValueError("Input argument rho must be within [0, 1)")

        if isinstance(mu, list):
            mu = np.array(mu)

        if np.sum(abs(mu)) == 0:
            raise ValueError("Input argument mu cannot be a vector of zeros")

        if not isinstance(n, int) or n < 0:
            raise ValueError("n must be a positive integer")

        allowed_methods = ["rejvmf", "rejacg", "rejpsaw"]
        if method not in allowed_methods:
            raise ValueError("Unknown method")

        if not isinstance(random_state, (int, type(None))):
            raise ValueError("Please specify a integer or None random_state")

        mu = mu / np.linalg.norm(mu)
        p = len(mu)

        generator = check_random_state(random_state)

        if method == "rejvmf":
            kappa = p * rho / (1 + rho**2)
            m1 = (1 + rho) / ((1 - rho) ** (p - 1) * np.exp(kappa))
            retvals = np.zeros((n, p))
            num_accepted = 0
            num_tries = 0
            while num_accepted < n:
                num_tries = num_tries + 1

                if isinstance(random_state, int):
                    random_state = random_state + num_accepted

                yx = vonmises_fisher(mu, kappa).rvs(random_state=random_state).ravel()
                v = mu @ yx
                f1 = (1 - rho**2) / ((1 + rho**2 - 2 * rho * v) ** (p / 2))
                u = generator.uniform(0, 1)
                if u <= f1 / (m1 * np.exp(kappa * v)):
                    num_accepted += 1
                    retvals[num_accepted - 1, :] = yx

        elif method == "rejacg":
            lamda = 2 * rho / (1 + rho**2)
            beta_lambda = lamda / (2 - lamda)
            beta_star = root_scalar(
                c_d_lambda, args=(p, lamda), bracket=[beta_lambda, 1], xtol=0.001
            ).root
            b1 = beta_star / (1 - beta_star)
            b2 = -1 + 1 / np.sqrt(1 - beta_star)
            retvals = np.zeros((n, p))
            num_accepted = 0
            num_tries = 0
            while num_accepted < n:
                num_tries = num_tries + 1
                u = generator.uniform(0, 1)
                z = generator.normal(size=p)
                q = (np.dot(mu.T, z) + b2 * np.dot(mu.T, z)) / (
                    np.sqrt(np.dot(z.T, z) + b1 * (np.dot(mu.T, z) ** 2))
                )
                m1 = (
                    p / 2 * (-np.log(1 - lamda * q))
                    + np.log(1 - beta_star * q**2)
                    - np.log(2 / (1 + np.sqrt(1 - lamda**2 / beta_star)))
                )
                yx = (z + b2 * np.dot(mu.T, np.dot(z, mu))) / np.sqrt(
                    np.dot(z.T, z) + b1 * (np.dot(mu.T, z) ** 2)
                )

                if np.log(u) <= m1:
                    num_accepted = num_accepted + 1
                    retvals[num_accepted - 1, :] = yx

        elif method == "rejpsaw":
            raise NotImplementedError("Not implemented at the moment")

        return retvals
