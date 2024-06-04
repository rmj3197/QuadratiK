import numpy as np
from sklearn.utils.validation import check_random_state


def compute_kernel_matrix(x_mat, y_mat, cov_h):
    """
    Compute the Gaussian kernel matrix between two samples.

    Parameters
    ----------
    x_mat : numpy.ndarray
        A matrix containing the observations of X.
    y_mat : numpy.ndarray
        A matrix containing the observations of Y.
    cov_h : numpy.ndarray
        Covariance matrix of the Gaussian kernel.

    Returns
    -------
    kernel matrix : numpy.ndarray
        The Gaussian kernel matrix between the observations of X and Y.
    """
    k = x_mat.shape[1]
    cov_h_inv = np.linalg.inv(cov_h)
    cov_h_det = np.linalg.det(cov_h) ** (-0.5)
    pi_const = (2 * np.pi) ** (-0.5 * k)
    diag_cov_h_inv = cov_h_inv.diagonal().reshape(-1, 1)
    cov_h_inv_sqrt = np.sqrt(diag_cov_h_inv).T
    x_mat_star = np.multiply(x_mat, cov_h_inv_sqrt)
    y_mat_star = np.multiply(y_mat, cov_h_inv_sqrt)
    x_mat_rowsum = np.sum(x_mat_star**2, axis=1, keepdims=True)
    y_mat_rowsum = np.sum(y_mat_star**2, axis=1, keepdims=True)
    qd = x_mat_rowsum - 2 * np.dot(x_mat_star, y_mat_star.T) + y_mat_rowsum.T
    kmat_zz = pi_const * cov_h_det * (np.exp(-0.5 * qd))
    return kmat_zz


def nonparam_centering(kmat_zz, n_z):
    """
    Compute the non-parametric centered kernel.

    Parameters
    ----------
    kmat_zz : numpy.ndarray
        A matrix containing the values of the
        kernel functions for each pair of X and Y.

    n_z : int
        The number of total observations, n + m.

    Returns
    -------
    centered kernel matrix : numpy.ndarray
        Matrix of centered kernel.
    """
    kmat = np.copy(kmat_zz)
    np.fill_diagonal(kmat, 0)

    k_center = (
        kmat_zz
        - np.sum(kmat, axis=1, keepdims=True) / (n_z - 1)
        - np.sum(kmat, axis=0, keepdims=True) / (n_z - 1)
        + (np.sum(kmat)) / (n_z * (n_z - 1))
    )
    return k_center


def param_centering(kmat_zz, z_mat, cov_h, mu_hat, sigma_hat):
    """
    Compute the Gaussian kernel centered with respect to a Normal distribution
    with mean vector mu and covariance sigma.

    Parameters
    ----------
    kmat_zz : numpy.ndarray
        A matrix containing the values of the kernel functions for each pair of X.

    z_mat : numpy.ndarray
        Matrix of observations used for computing the kernel matrix.

    cov_h : numpy.ndarray
        Covariance matrix of the Normal kernel.

    mu_hat : numpy.ndarray
        Mean of centering of the Normal distribution.

    sigma_hat : numpy.ndarray
        Covariance of centering Normal distribution.

    Returns
    --------
    centered kernel matrix : numpy.ndarray
        Matrix of centered kernel.

    """
    n_z = z_mat.shape[0]
    k_center = (
        kmat_zz
        - compute_kernel_matrix(z_mat, mu_hat, cov_h + sigma_hat)
        - compute_kernel_matrix(mu_hat, z_mat, cov_h + sigma_hat)
        + compute_kernel_matrix(mu_hat, mu_hat, cov_h + 2 * sigma_hat)[0][0]
        * np.ones((n_z, n_z))
    )
    return k_center


def dof_normality_test(sigma_h, v):
    """
    Compute the Degrees of Freedom (DOF) of the normal Kernel centered with
    respect to the standard normal distribution, given the dimension d and the
    bandwidth parameter h.

    Parameters
    -----------
        sigma_h : np.ndarray
            ovariance matrix of the gaussian kernel.

        v : np.ndarray
            Covariance matrix of the tested distribution G

    Returns
    --------
        DOF and Coefficient : tuple
            DOF and the coefficient of the asymptotic distribution.
    """
    num_dof = np.linalg.det(sigma_h) ** (-1 / 2) - np.linalg.det(sigma_h + 2 * v) ** (
        -1 / 2
    )
    den_dof = (
        np.linalg.det(sigma_h) ** (-1 / 2) * np.linalg.det(sigma_h + 4 * v) ** (-1 / 2)
        - 2
        * np.linalg.det(sigma_h + v) ** (-1 / 2)
        * np.linalg.det(sigma_h + 3 * v) ** (-1 / 2)
        + np.linalg.det(sigma_h + 2 * v) ** (-1 / 2)
    )

    dof = num_dof**2 / den_dof
    const = den_dof / num_dof

    return (dof, const)


def variance_normality_test(sigma_h, v, n):
    """
    Compute the exact variance of kernel test for normality under the null
    hypothesis that G=N(0,I).

    Parameters
    ----------
        sigma_h: np.ndarray
            Covariance matrix of the gaussian kernel.

        v : np.ndarray
            Covariance matrix of the tested distribution G.

        n : int
            Sample size.

    Returns
    --------
        variance: float
            value of computed variance.
    """
    d = sigma_h.shape[0]
    var = (
        2
        / (n * (n - 1))
        * (1 / ((2 * np.pi) ** d))
        * (
            np.linalg.det(sigma_h) ** (-1 / 2)
            * np.linalg.det(sigma_h + 4 * v) ** (-1 / 2)
            - 2
            * np.linalg.det(sigma_h + v) ** (-1 / 2)
            * np.linalg.det(sigma_h + 3 * v) ** (-1 / 2)
            + np.linalg.det(sigma_h + 2 * v) ** (-1)
        )
    )
    return var


def variance_two_sample_test(k_cen, n, m):
    """
    Exact variance of two-sample test. Compute the exact variance of kernel test for the
    two-sample problem under the null hypothesis that F=G.

    Parameters
    ----------
    k_cen : numpy.ndarray
        Matrix with centered kernel values.

    n : int
        Number of samples of type/kind 1.

    n : int
        Number of samples of type/kind 2.

    Returns
    -------
    Computed variances : tuple
        Tuple of variances for Dn and trace statistics.
    """

    np.fill_diagonal(k_cen, 0)

    K_xx = k_cen[:n, :n]
    K_yy = k_cen[n : n + m, n : n + m]
    K_xy = k_cen[:n, n : n + m]

    n_factor = 1.0 / (n * (n - 1))
    m_factor = 1.0 / (m * (m - 1))
    cross_factor = 1.0 / (n * m)

    # Variance estimate calculation - Dn
    est_var_D = (
        2 * n_factor**2 * (K_xx**2).sum()
        + 8 * cross_factor**2 * (K_xy**2).sum()
        + 2 * m_factor**2 * (K_yy**2).sum()
    )

    delta1 = np.sum(K_xx[:n, :n].T @ K_xy[:n, :m])

    delta2 = np.sum(K_yy[:m, :m].T @ K_xy[:n, :m].T)

    est_var_D -= 8 * n_factor * cross_factor * delta1
    est_var_D -= 8 * m_factor * cross_factor * delta2

    # Variance estimate calculation - trace
    est_var_Tr = (
        2 * (n_factor**2) * (K_xx**2).sum() + 2 * (m_factor**2) * (K_yy**2).sum()
    )

    return (est_var_D, est_var_Tr)


def variance_k_sample_test(k_cen, sizes, cum_size):
    """
    Compute the exact variance of kernel test for the k-sample problem under
    the null hypothesis that F1=...=Fk.

    Parameters
    -----------
        k_cen : np.ndarray
            matrix with centered kernel values.

        sizes : list, np.ndarray
            vector indicating size of samples.

        cum_size : list, np.ndarray
            vector indicating sample's cumulative sizes.

    Returns:
        variance: float
            value of computed variance.
    """
    k = len(sizes)

    np.fill_diagonal(k_cen, 0)

    ni_factors = 1 / (sizes * (sizes - 1))

    C1, C2, C3 = 0, 0, 0

    for l in range(k):
        ni_factor = ni_factors[l]
        k_ll = k_cen[
            cum_size[l] : cum_size[l] + sizes[l],
            cum_size[l] : cum_size[l] + sizes[l],
        ]

        C1 += 2 * (ni_factor**2) * (k_ll**2).sum()

        for r in range(l + 1, k):
            n_lr_factor = 1 / (sizes[l] ** 2)
            k_lr = k_cen[
                cum_size[l] : cum_size[l] + sizes[l],
                cum_size[r] : cum_size[r] + sizes[r],
            ]
            k_rr = k_cen[
                cum_size[r] : cum_size[r] + sizes[r],
                cum_size[r] : cum_size[r] + sizes[r],
            ]

            C2 += 8 * (n_lr_factor**2) * (k_lr**2).sum()
            C3 -= 8 * n_lr_factor * ni_factor * (k_ll.T @ k_lr).sum()
            C3 -= 8 * n_lr_factor * ni_factor * (k_rr.T @ k_lr.T).sum()

    est_var_D = (k - 1) ** 2 * C1 + C2 + C3
    est_var_Tr = C1

    return (est_var_D, est_var_Tr)


def stat_two_sample(x_mat, y_mat, h, mu_hat, sigma_hat, centering_type="nonparam"):
    """
    Compute kernel-based quadratic distance two-sample
    test with Normal kernel.

    Parameters
    ----------
    x_mat : numpy.ndarray
        A matrix containing observations from the first sample.

    y_mat : numpy.ndarray
        A matrix containing observations from the second sample.

    h : float
        The bandwidth parameter for the kernel function.

    mu_hat : numpy.ndarray
        Mean vector for the reference distribution.

    sigma_hat : numpy.ndarray
        Covariance matrix of the reference distribution.

    centering_type : str
        String indicating the method used for centering
        the normal kernel.

    Returns
    -------
    test statistics and associated variances : np.ndarray
        Test statistics and variances for the two-sample test.
    """
    n_x = x_mat.shape[0]
    k = x_mat.shape[1]
    n_y = y_mat.shape[0]
    n_z = n_x + n_y
    z_mat = np.vstack((x_mat, y_mat))
    cov_h = (h**2) * np.identity(k)
    kmat_zz = compute_kernel_matrix(z_mat, z_mat, cov_h)
    if centering_type == "nonparam":
        k_center = nonparam_centering(kmat_zz, n_z)
    elif centering_type == "param":
        k_center = param_centering(kmat_zz, z_mat, cov_h, mu_hat, sigma_hat)
    np.fill_diagonal(k_center, 0)
    test_non_par = (
        (np.sum(k_center[:n_x, :n_x]) / (n_x * (n_x - 1)))
        - 2 * (np.sum(k_center[:n_x, n_x : n_x + n_y]) / (n_x * n_y))
        + (np.sum(k_center[n_x : n_x + n_y, n_x : n_x + n_y]) / (n_y * (n_y - 1)))
    )

    test_trace = np.sum(k_center[:n_x, :n_x]) / (n_x * (n_x - 1)) + np.sum(
        k_center[n_x : n_x + n_y, n_x : n_x + n_y]
    ) / (n_y * (n_y - 1))

    var1, var2 = variance_two_sample_test(k_center, n_x, n_y)

    return np.array([test_non_par, test_trace, var1, var2])


def stat_normality_test(x_mat, h, mu_hat, sigma_hat):
    """
    Compute kernel-based quadratic distance test for
    Normality

    Parameters
    ----------
    x_mat : numpy.ndarray
        A matrix containing observations from the first sample.
    h : float
        The bandwidth parameter for the kernel function.
    mu_hat : numpy.ndarray
        Mean vector for the reference distribution.
    sigma_hat : numpy.ndarray
        Covariance matrix of the reference distribution.
    centering_type : str, optional. Defaults to param.
        String indicating the method used for centering the normal kernel.

    Returns
    -------
    test statistics : np.ndarray
        Test statistics for the normality test.
    """
    n_x = x_mat.shape[0]
    k = x_mat.shape[1]
    cov_h = (h**2) * np.identity(k)
    kmat_zz = compute_kernel_matrix(x_mat, x_mat, cov_h)

    k_center = param_centering(kmat_zz, x_mat, cov_h, mu_hat, sigma_hat)

    # Compute the normality test V-statistic
    Vn = np.sum(k_center) / (n_x)

    # Compute the normality test U-statistic
    Un = (np.sum(k_center) - np.sum(np.diagonal(k_center))) / (n_x * (n_x - 1))

    return np.array([Un, Vn])


def stat_ksample(x, y, h):
    """
    Compute the kernel-based quadratic distance k-sample tests
    with the Normal kernel and bandwidth parameter h.

    Parameters
    ----------
    x : numpy.ndarray
        A matrix containing observations from the pooled
        sample, from the k samples.

    y : numpy.ndarray
        A vector containing observations' memberships to the k samples.

    h : float
        The bandwidth parameter for the kernel function.

    Returns
    -------
    test statistics and associated variances : numpy.ndarray
        Test statistics and variances for the k-sample test.
    """
    sizes = np.unique(y, return_counts=True)[1]
    n, d = x.shape
    k = len(np.unique(y))
    cov_h = (h**2) * np.diag(np.ones(d))
    kern_matrix = compute_kernel_matrix(x, x, cov_h)
    k_center = nonparam_centering(kern_matrix, n)
    trace_k = 0
    cum_size = np.insert(np.cumsum(sizes), 0, 0)
    tn = 0

    for l in range(k):
        k_ll = k_center[
            cum_size[l] : cum_size[l] + sizes[l], cum_size[l] : cum_size[l] + sizes[l]
        ]
        np.fill_diagonal(k_ll, 0)

        if sizes[l] > 1:
            trace_k += k_ll.sum() / (sizes[l] * (sizes[l] - 1))

        for r in range(l + 1, k):
            k_lr = k_center[
                cum_size[l] : cum_size[l] + sizes[l],
                cum_size[r] : cum_size[r] + sizes[r],
            ]
            if (sizes[l] > 0) and (sizes[r] > 0):
                tn -= 2 * k_lr.sum() / (sizes[l] * sizes[r])

    var1, var2 = variance_k_sample_test(k_center, sizes, cum_size)
    stat1 = (k - 1) * trace_k + tn
    stat2 = trace_k
    return np.array([stat1, stat2, var1, var2])


def normal_cv_helper(size, h, mu_hat, sigma_hat, n_rep, random_state):
    """
    Generate a sample from a multivariate normal distribution and perform a
    kernel-based quadratic distance test for normality.

    Parameters
    ----------
    size : int
        The size of the sample to be generated from the
        multivariate normal distribution.

    h : float
        The bandwidth parameter for the kernel function
        in the quadratic distance test.

    mu_hat : numpy.ndarray
        Mean vector for the reference distribution.

    sigma_hat : numpy.ndarray
        Covariance matrix of the reference distribution.

    n_rep : int
        The number of replication.

    random_state : int, None.
        Seed for random number generation.

    Returns
    -------
    test statistic : numpy.ndarray
        An array containing the test statistics from the kernel-based quadratic
        distance test for normality on the generated sample.
    """
    if random_state is None:
        generator = check_random_state(random_state)
    elif isinstance(random_state, (int, np.int_)):
        generator = check_random_state(random_state + n_rep)

    dat = generator.multivariate_normal(mu_hat.ravel(), sigma_hat, size)
    return stat_normality_test(dat, h, mu_hat, sigma_hat)[0]


def bootstrap_helper_twosample(size_x, size_y, h, data_pool, n_rep, random_state):
    """
    Helper function for CV estimation using bootstrap for
    two sample test.

    Parameters
    ----------
    size_x : int
        The size of the bootstrap sample for the first group.

    size_y : int
        The size of the bootstrap sample for the second group.

    h : float
        The bandwidth parameter for the kernel function
        in the two-sample test.

    data_pool : numpy.ndarray
        Pooled data from both groups.

    n_rep : int
        The number of replication.

    random_state : int, None.
        Seed for random number generation.

    Returns
    -------
    test statistic : numpy.ndarray
        The result of the two-sample test using bootstrap
        resampling for the randomly generated data.
    """
    if random_state is None:
        generator = check_random_state(random_state)
    elif isinstance(random_state, (int, np.int_)):
        generator = check_random_state(random_state + n_rep)

    ind_x = generator.choice(np.arange(0, size_x + size_y, 1), size_x, replace=True)
    ind_y = generator.choice(np.arange(0, size_x + size_y, 1), size_y, replace=True)
    data_x_star = data_pool[ind_x]
    data_y_star = data_pool[ind_y]
    result = stat_two_sample(
        data_x_star,
        data_y_star,
        h,
        np.repeat(0, data_pool.shape[1]),
        np.array([[1]]),
        "nonparam",
    )[:2]
    return result


def permutation_helper_twosample(size_x, size_y, h, data_pool, n_rep, random_state):
    """
    Helper function for CV estimation using permutation for
    two sample test.

    Parameters
    ----------
    size_x : int
        The size of the bootstrap sample for the first group.

    size_y : int
        The size of the bootstrap sample for the second group.

    h : float
        The bandwidth parameter for the kernel function
        in the two-sample test.

    data_pool : numpy.ndarray
        Pooled data from both groups.

    n_rep : int
        The number of replication.

    random_state : int, None.
        Seed for random number generation.

    Returns
    -------
    test statistic : numpy.ndarray
        The result of the two-sample test using permutation
        resampling for the randomly generated data.
    """
    if random_state is None:
        generator = check_random_state(random_state)
    elif isinstance(random_state, (int, np.int_)):
        generator = check_random_state(random_state + n_rep)

    ind_x = generator.choice(np.arange(0, size_x + size_y, 1), size_x, replace=False)
    ind_y = np.setdiff1d(np.arange(0, size_x + size_y, 1), ind_x)
    data_x_star = data_pool[ind_x]
    data_y_star = data_pool[ind_y]
    result = stat_two_sample(
        data_x_star,
        data_y_star,
        h,
        np.repeat(0, data_pool.shape[1]),
        np.array([[1]]),
        "nonparam",
    )[:2]
    return result


def subsampling_helper_twosample(size_x, size_y, b, h, data_pool, n_rep, random_state):
    """
    Helper function for CV estimation using subsampling for
    two sample test.

    Parameters
    ----------
    size_x : int
        The size of the bootstrap sample for the first group.

    size_y : int
        The size of the bootstrap sample for the second group.

    b : float
        The size of the subsamples used in the subsampling algorithm.

    h : float
        The bandwidth parameter for the kernel function
        in the two-sample test.

    data_pool : numpy.ndarray
        Pooled data from both groups.

    n_rep : int
        The number of replication.

    random_state : int, None.
        Seed for random number generation.

    Returns
    -------
    test statistic : numpy.ndarray
        The result of the two-sample test using subsampling
        resampling for the randomly generated data.
    """
    if random_state is None:
        generator = check_random_state(random_state)
    elif isinstance(random_state, (int, np.int_)):
        generator = check_random_state(random_state + n_rep)

    ind_x = generator.choice(np.arange(size_x), size=round(size_x * b), replace=False)
    ind_y = generator.choice(
        np.arange(size_x, size_x + size_y), size=round(size_y * b), replace=False
    )
    newind = np.concatenate((ind_x, ind_y))
    newsample = generator.choice(newind, size=len(newind), replace=False)
    data_x_star = data_pool[newsample[: round(size_x * b)], :]
    data_y_star = data_pool[newsample[round(size_x * b) :], :]
    result = stat_two_sample(
        data_x_star,
        data_y_star,
        h,
        np.repeat(0, data_pool.shape[1]),
        np.array([[1]]),
        "nonparam",
    )[:2]
    return result


def bootstrap_helper_ksample(x, y, k, h, sizes, cum_size, n_rep, random_state):
    """
    Helper function for CV estimation using bootstrap for
    K-sample test.

    Parameters
    ----------
    x : numpy.ndarray
        The matrix containing observations from the pooled sample,
        from the k samples.

    y : numpy.ndarray
        The vector containing observations' memberships to the k samples.

    k : int
        The number of groups.

    h : float
        The bandwidth parameter for the kernel function in the k-sample test.

    sizes : numpy.ndarray
        Array containing the sizes of each group in the pooled sample.

    cum_size : numpy.ndarray
        Array containing the cumulative sizes of each group in the pooled sample.

    n_rep : int
        The number of replication.

    random_state : int, None.
        Seed for random number generation.

    Returns
    -------
    test statistic : numpy.ndarray
        The result of the k-sample test using bootstrap resampling for
        the randomly generated data.
    """
    if random_state is None:
        generator = check_random_state(random_state)
    elif isinstance(random_state, (int, np.int_)):
        generator = check_random_state(random_state + n_rep)

    ind_k = np.concatenate(
        [
            generator.choice(
                np.arange(0, sizes[k]) + cum_size[k], sizes[k], replace=True
            )
            for k in range(k)
        ]
    )
    ind_k = ind_k.astype(int)
    newsample = generator.choice(ind_k, len(ind_k), replace=False)
    y_ind = y
    data_k = x[newsample,]
    res_num_iter = stat_ksample(data_k, y_ind, h)[:2]
    return res_num_iter


def subsampling_helper_ksample(x, y, k, h, sizes, b, cum_size, n_rep, random_state):
    """
    Helper function for CV estimation using subsampling for
    K-sample test.

    Parameters
    ----------
    x : numpy.ndarray
        The matrix containing observations from the pooled sample,
        from the k samples.

    y : numpy.ndarray
        The vector containing observations' memberships to the k samples.

    k : int
        The number of samples/groups.

    h : float
        The bandwidth parameter for the kernel function in the k-sample test.

    sizes : numpy.ndarray
        Array containing the sizes of each group in the pooled sample.

    b : float
        The subsampling proportion (percentage of each group's size to use
        in each subsample).

    cum_size : numpy.ndarray
        Array containing the cumulative sizes of each group in the pooled sample.

    n_rep : int
        The number of replication.

    random_state : int, None.
        Seed for random number generation.

    Returns
    -------
    test statistic : numpy.ndarray
        The result of the k-sample test using subsampling using the randomly
        generated data.
    """
    if random_state is None:
        generator = check_random_state(random_state)
    elif isinstance(random_state, (int, np.int_)):
        generator = check_random_state(random_state + n_rep)

    ind_k = np.concatenate(
        [
            generator.choice(
                np.arange(0, sizes[k]) + cum_size[k],
                int(np.round(sizes[k] * b)),
                replace=False,
            )
            for k in range(k)
        ]
    )
    ind_k = ind_k.astype(int)
    y_ind = y[ind_k]
    newsample = generator.choice(ind_k, len(ind_k), replace=False)
    data_k = x[newsample,]
    res_num_iter = stat_ksample(data_k, y_ind, h)[:2]
    return res_num_iter


def permutation_helper_ksample(x, y, n, h, n_rep, random_state):
    """
    Helper function for CV estimation using permutation for
    K-sample test.

    Parameters
    --------------
    x : numpy.ndarray
        The matrix containing observations from the pooled sample,
        from the k samples.

    y : numpy.ndarray
        The vector containing observations' memberships to the k samples.

    n : int
        The total number of observations in the pooled sample.

    h : float
        The bandwidth parameter for the kernel function in the k-sample test.

    n_rep : int
        The number of replication.

    random_state : int, None.
        Seed for random number generation.

    Returns
    ---------
    test statistic : numpy.ndarray
        The result of the k-sample test using permutation using the randomly
        generated data.
    """
    if random_state is None:
        generator = check_random_state(random_state)
    elif isinstance(random_state, (int, np.int_)):
        generator = check_random_state(random_state + n_rep)

    ind_k = generator.choice(np.arange(0, n), n, replace=False)
    ind_k = ind_k.astype(int)
    y_ind = y
    data_k = x[ind_k,]
    res_num_iter = stat_ksample(data_k, y_ind, h)[:2]
    return res_num_iter
