"""
Poisson Kernel based Clustering
"""
import importlib
import numpy as np
import pandas as pd
import scipy.special as sp
from scipy.optimize import root_scalar
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import check_random_state
from sklearn.metrics import (precision_score,
                             recall_score,
                             adjusted_rand_score,
                             silhouette_score,
                             confusion_matrix)
from sklearn.preprocessing import LabelEncoder

from ._utils import (root_func,
                     calculate_wcss_euclidean,
                     calculate_wcss_cosine)

stats = importlib.import_module('QuadratiK.tools').stats


class PKBC():
    """
    Poisson kernel-based clustering on the sphere. 
    The class performs the Poisson kernel-based clustering algorithm 
    on the sphere based on the Poisson kernel-based densities. It estimates
    the parameter of a mixture of Poisson kernel-based densities. The obtained
    estimates are used for assigning final memberships, identifying the data points.

    Parameters
    ----------
        num_clust : int
            Number of clusters.

        max_iter : int
            Maximum number of iterations before a run is terminated.

        stopping_rule : str, optional
            String describing the stopping rule to be used within each run. 
            Currently must be either 'max', 'membership', or 'loglik'.

        init_method : str, optional
            String describing the initialization method to be used. 
            Currently must be 'sampleData'.

        num_init : int, optional
            Number of initializations.

        tol : float.
            Constant defining threshold by which log 
            likelihood must change to continue iterations, if applicable.
            Defaults to 1e-7.

        random_state : int, None, optional. 
            Seed for random number generation. Defaults to None

        n_jobs : int
            Used only for computing the WCSS efficiently.
            n_jobs specifies the maximum number of concurrently running workers. 
            If 1 is given, no joblib parallelism is used at all, which is useful for debugging.
            For more information on joblib n_jobs refer to - 
            https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html.
            Defaults to 4. 

    Attributes
    ----------
        alpha\\_ : numpy.ndarray of shape (n_clusters,)
            Estimated mixing proportions

        labels\\_ : numpy.ndarray of shape (n_samples,)
            Final cluster membership assigned by the algorithm to each observation

        log_lik_vec : numpy.ndarray of shape (num_init, )
            Array of log-likelihood values for each initialization

        loklik\\_ : float
            Maximum value of the log-likelihood function

        mu\\_ : numpy.ndarray of shape (n_clusters, n_features)
            Estimated centroids

        num_iter_per_run : numpy.ndarray of shape (num_init, )
            Number of E-M iterations per run

        post_probs\\_ : numpy.ndarray of shape (n_samples, n_features)
            Posterior probabilities of each observation for the indicated clusters

        rho\\_ : numpy.ndarray of shape (n_clusters,)
            Estimated concentration parameters rho

        euclidean\\_wcss\\_ : float
            Values of within-cluster sum of squares computed with 
            Euclidean distance.
        
        cosine\\_wcss\\_ : float
            Values of within-cluster sum of squares computed with 
            cosine similarity.
        
    References
    ----------
        Golzy M. & Markatou M. (2020) Poisson Kernel-Based 
        Clustering on the Sphere: Convergence Properties, Identifiability, 
        and a Method of Sampling, Journal of Computational and Graphical Statistics, 
        29:4, 758-770, DOI: 10.1080/10618600.2020.1740713.

    Examples
    ---------
    >>> from QuadratiK.datasets import load_wireless_data
    >>> from QuadratiK.spherical_clustering import PKBC
    >>> from sklearn.preprocessing import LabelEncoder
    >>> X, y = load_wireless_data(return_X_y=True)
    >>> le = LabelEncoder()
    >>> le.fit(y)
    >>> y = le.transform(y)
    >>> cluster_fit = PKBC(num_clust=4, random_state=42).fit(X)
    >>> ari, macro_precision, macro_recall, avg_silhouette_Score = cluster_fit.validation(y)
    >>> print("Estimated mixing proportions :", cluster_fit.alpha_)
    >>> print("Estimated concentration parameters: ", cluster_fit.rho_)
    >>> print("Adjusted Rand Index:", ari)
    >>> print("Macro Precision:", macro_precision)
    >>> print("Macro Recall:", macro_recall)
    >>> print("Average Silhouette Score:", avg_silhouette_Score)
    ... Estimated mixing proportions : [0.23590339 0.24977919 0.25777522 0.25654219]
    ... Estimated concentration parameters:  [0.97773265 0.98348976 0.98226901 0.98572597]
    ... Adjusted Rand Index: 0.9403086353805835
    ... Macro Precision: 0.9771870612442508
    ... Macro Recall: 0.9769999999999999
    ... Average Silhouette Score: 0.3803089203572107
    """

    def __init__(self, num_clust, max_iter=300, stopping_rule='loglik',
                 init_method='sampledata', num_init=10, tol=1e-7, random_state=None, n_jobs=4):
        self.num_clust = num_clust
        self.max_iter = max_iter
        self.stopping_rule = stopping_rule
        self.init_method = init_method
        self.num_init = num_init
        self.tol = tol
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, dat):
        """
        Performs Poisson Kernel-based Clustering.

        Parameters
        ----------
            dat : numpy.ndarray, pandas.DataFrame
                A numeric array of data values.

        Returns
        -------
            self : object
                Fitted estimator
        """
        self.dat = dat
        if isinstance(self.dat, pd.DataFrame):
            self.dat = self.dat.to_numpy()
        if self.num_clust < 1:
            raise Exception("Input parameter num_clust must be greater than 1")
        if self.max_iter < 1:
            raise Exception("Input parameter maxIter must be greater than 0")
        if self.stopping_rule not in ['max', 'membership', 'loglik']:
            raise Exception(
                "Unrecognized value {} in input parameter.".format(self.stopping_rule))
        if self.init_method not in ['sampledata']:
            raise Exception(
                "Unrecognized value {} in input parameter.".format(self.init_method))
        if self.num_init < 1:
            raise Exception("Input parameter numInit must be greater than 0")
        if not isinstance(self.random_state, (int, type(None))):
            raise ValueError("Please specify a integer or None random_state")

        generator = check_random_state(self.random_state)

        # set options for stopping rule
        check_membership = self.stopping_rule == 'membership'
        check_loglik = self.stopping_rule == 'loglik'

        # Normalize the data
        self.dat = self.dat / np.linalg.norm(self.dat, axis=1, keepdims=True)

        num_data, num_var = self.dat.shape
        self.log_lik_vec = np.repeat(float('-inf'), self.num_init)
        self.num_iter_per_run = np.repeat(-99, self.num_init)
        alpha_best = np.repeat(-99, self.num_clust)
        rho_best = np.repeat(-99, self.num_clust)
        mu_best = np.zeros((self.num_clust, num_var))
        norm_prob_mat_best = np.full((num_data, self.num_clust), -99)

        if self.init_method == 'sampledata':
            unique_data = np.unique(self.dat, axis=0)
            num_unique_obs = unique_data.shape[0]
            if num_unique_obs < self.num_clust:
                raise ValueError("Only {} 'unique observations.', \
                    When init_method = {}, \
                        must have more than num_clust unique observations.".format(
                    num_unique_obs, self.init_method))
        log_w_d = (num_var/2)*(np.log(2) + np.log(np.pi)) - \
            sp.gammaln(num_var/2)

        for init in range(self.num_init):
            alpha_current = np.full(self.num_clust, 1/self.num_clust)
            rho_current = np.full(self.num_clust, 0.5)

            if self.init_method == 'sampledata':
                mu_current = unique_data[generator.choice(
                    num_unique_obs, size=self.num_clust, replace=False), :]

            current_iter = 1
            memb_current = np.empty(num_data)
            log_lik_current = float('-inf')

            while current_iter <= self.max_iter:
                v_mat = np.dot(self.dat, mu_current.T)
                alpha_mat_current = np.tile(alpha_current, (num_data, 1))
                rho_mat_current = np.tile(rho_current, (num_data, 1))
                log_prob_mat_denom = np.log(
                    1+rho_mat_current**2 - 2*np.asarray(rho_mat_current)*np.asarray(v_mat))

                log_prob_mat = np.log(1-(rho_mat_current)**2) - \
                    log_w_d - (num_var/2)*log_prob_mat_denom

                prob_sum = np.tile(
                    np.dot(np.exp(log_prob_mat),
                           alpha_current).reshape(num_data, 1),
                    (1, self.num_clust))

                log_norm_prob_mat_current = np.log(
                    alpha_mat_current) + log_prob_mat - np.log(prob_sum)

                log_weight_mat = log_norm_prob_mat_current - log_prob_mat_denom
                alpha_current = np.sum(
                    np.exp(log_norm_prob_mat_current), axis=0)/num_data
                mu_num_sum_mat = np.dot(np.exp(log_weight_mat).T, self.dat)
                mu_denom = np.linalg.norm(
                    mu_num_sum_mat, axis=1, keepdims=True)
                mu_current = mu_num_sum_mat / mu_denom
                for h in range(self.num_clust):
                    sum_h_weight_mat = np.sum(np.exp(log_weight_mat[:, h]))
                    alpha_current_h = alpha_current[h]
                    mu_denom_h = mu_denom[h]
                    rho_current[h] = root_scalar(root_func, args=(
                        num_data, alpha_current_h, num_var, mu_denom_h,
                        sum_h_weight_mat), bracket=[0, 1], xtol=0.001).root
                if current_iter >= self.max_iter:
                    break

                if check_membership:
                    memb_previous = memb_current
                    memb_current = np.argmax(
                        np.exp(log_norm_prob_mat_current), axis=1)
                    if all(memb_previous == memb_current):
                        break

                if check_loglik:
                    log_lik_previous = log_lik_current
                    log_lik_current = np.sum(
                        np.log(np.dot(np.exp(log_prob_mat), alpha_current)))
                    if np.abs(log_lik_previous - log_lik_current) < self.tol:
                        break
                current_iter = current_iter + 1
            log_lik_current = np.sum(
                np.log(np.dot(np.exp(log_prob_mat), alpha_current)))

            if log_lik_current > max(self.log_lik_vec):
                alpha_best = alpha_current
                rho_best = rho_current
                mu_best = mu_current
                norm_prob_mat_best = np.exp(log_norm_prob_mat_current)
            self.log_lik_vec[init] = log_lik_current
            self.num_iter_per_run[init] = current_iter - 1

        memb_best = np.argmax(norm_prob_mat_best, axis=1)

        # euclidean wcss calculation
        cluster_euclidean_wcss = Parallel(n_jobs=self.n_jobs)(
            delayed(calculate_wcss_euclidean)(k, memb_best, self.dat, mu_best)
            for k in range(self.num_clust)
        )
        self.euclidean_wcss_ = np.sum(cluster_euclidean_wcss)

        # cosine similarity wcss calculation
        cluster_cosine_wcss = Parallel(n_jobs=self.n_jobs)(
            delayed(calculate_wcss_cosine)(k, memb_best, self.dat, mu_best)
            for k in range(self.num_clust)
        )

        self.cosine_wcss_ = np.sum(cluster_cosine_wcss)

        self.post_probs_ = norm_prob_mat_best
        self.loglik_ = max(self.log_lik_vec)
        self.mu_ = mu_best
        self.rho_ = rho_best
        self.alpha_ = alpha_best
        self.labels_ = memb_best
        return self

    def validation(self, y_true=None):
        """
        Computes validation metrics such as ARI, Macro Precision 
        and Macro Recall when true labels are provided.

        Parameters
        -----------
            y_true : numpy.ndarray. 
                Array of true memberships to clusters,
                Defaults to None.

        Returns
        -------
            validation metrics : tuple
                The tuple consists of the following:\n
                - Adjusted Rand Index : float (returned only when y_true is provided)
                    Adjusted Rand Index computed between the true and predicted cluster memberships.
                - Macro Precision : float (returned only when y_true is provided)
                    Macro Precision computed between the true and predicted cluster memberships.
                - Macro Recall : float (returned only when y_true is provided)
                    Macro Recall computed between the true and predicted cluster memberships.
                - Average Silhouette Score : float
                    Mean Silhouette Coefficient of all samples.
                    
        References
        ----------
            Rousseeuw, P.J. (1987) Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. 
            Journal of Computational and Applied Mathematics, 20, 53–65.
        
        Notes
        -----
            We have taken a naive approach to map the predicted cluster labels 
            to the true class labels (if provided). This might not work in cases where `num_clust` is large.
            Please use `sklearn.metrics` for computing metrics in such cases, and provide the correctly
            matched labels.

        See also
        --------
            `sklearn.metrics` : Scikit-learn metrics functionality support a wide range of metrics.
        """

        if isinstance(y_true, pd.DataFrame):
            y_true = y_true.values.flatten()

        avg_silhouette_score = silhouette_score(self.dat, self.labels_)

        if y_true is not None:
            le = LabelEncoder()
            le.fit(y_true)
            y_true = le.transform(y_true)

            cm = confusion_matrix(y_true, self.labels_)
            cm_argmax = cm.argmax(axis=0)
            y_pred_ = np.array([cm_argmax[i] for i in self.labels_])

            ari = adjusted_rand_score(y_true, self.labels_)
            macro_precision = precision_score(
                y_true, y_pred_, average="macro")
            macro_recall = recall_score(y_true, y_pred_, average="macro")
            validation_metrics = (ari, macro_precision,
                                  macro_recall, avg_silhouette_score)

        else:
            validation_metrics = (avg_silhouette_score)
        return validation_metrics

    def stats(self):
        """
        Function to generate descriptive statistics per variable (and per group if available).

        Returns
        -------
            summary_stats_df : pandas.DataFrame
                Dataframe of descriptive statistics

        """

        summary_stats = stats(self.dat, self.labels_)
        summary_stats_df = pd.concat(
            summary_stats.values(), keys=summary_stats.keys(), axis=0)
        return summary_stats_df
