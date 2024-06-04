import numpy as np
from sklearn.utils.validation import check_random_state


def dof(d, rho):
    """
    Compute the Degrees of Freedom (DOF) of the Poisson Kernel given the
    dimension d and concentration parameter rho

    Parameters
    --------------
    d : int
        The number of dimensions
    rho : float
        Concentration parameter

    Returns
    ---------
    dict
        A dictionary containing the degrees of freedom (DOF) and coefficient
        of the asymptotic distribution.
    """

    num_c = ((1 + rho**2) / ((1 - rho**2) ** (d - 1))) - 1
    den_c = ((1 + rho) / ((1 - rho) ** (d - 1))) - 1
    dof_val = (den_c**2) / num_c
    c = num_c / den_c
    result = {"DOF": dof_val, "Coefficient": c}
    return result


def stat_poisson_unif(x_mat, rho):
    """
    Compute the Poisson kernel-based test for Uniformity
    given a sample of observations on the Sphere.

    Parameters
    --------------
    x_mat : numpy.ndarray
        A matrix containing the observations of X on the Sphere.

    rho : float
        Concentration parameter of the Poisson kernel.

    Returns
    ---------
    (Un,Vn) : tuple
        Tuple containing value of the
        U-statistic and V-statistic.
    """
    n_x = x_mat.shape[0]
    pmat = compute_poisson_matrix(x_mat, rho)
    vn = np.sum(pmat) / n_x
    lower_tri_sum = np.sum(np.tril(pmat, -1))
    un = 2 * lower_tri_sum / (n_x * (n_x - 1))
    return (un, vn)


def poisson_cv_helper(size, d, rho, n_rep, random_state):
    """
    Generate a sample of observations on the Sphere and
    perform a Poisson kernel-based test for uniformity.

    Parameters
    ----------
    size : int
        The size of the sample to be generated on the Sphere.
    d : int
        The dimension of the observations.
    rho : float
        Concentration parameter of the Poisson kernel.

    n_rep : int
        The number of replication.

    random_state : int, None.
        Seed for random number generation.

    Returns
    -------
    Un : float
        The U-statistic for the generated random samples.
    """
    if random_state is None:
        generator = check_random_state(random_state)
    elif isinstance(random_state, int):
        generator = check_random_state(random_state + n_rep)

    dat = generator.randn(size, d)
    dat = dat / np.linalg.norm(dat, axis=1, keepdims=True)
    un = stat_poisson_unif(dat, rho)[0]
    return un


def compute_poisson_matrix(x_mat, rho):
    """
    Compute the Poisson kernel matrix between
    observations in a sample.

    Parameters
    --------------
    x_mat : numpy.ndarray
        A matrix containing the observations of X on the Sphere.

    rho : float
        Concentration parameter of the Poisson kernel.

    Returns
    ---------
    poisson kernel matrix : numpy.ndarray
        Matrix of Poisson kernel.
    """
    n_x = x_mat.shape[0]
    k = x_mat.shape[1]
    x_mat_star = np.dot(x_mat, x_mat.T)
    pmat_array = (1 - rho**2) / (1 + rho**2 - 2 * rho * x_mat_star) ** (
        k / 2.0
    ) - np.ones((n_x, n_x))
    return pmat_array
