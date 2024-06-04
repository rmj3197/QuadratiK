"""
Poisson Kernel based Clustering
"""

import importlib
import numpy as np
import pandas as pd
from collections import Counter
from tabulate import tabulate
import scipy.special as sp
from scipy.optimize import root_scalar
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import check_random_state
from sklearn.metrics import (
    precision_score,
    recall_score,
    adjusted_rand_score,
    silhouette_score,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ._utils import root_func, calculate_wcss_euclidean, calculate_wcss_cosine

stats = importlib.import_module("QuadratiK.tools").stats
extract3d = importlib.import_module("QuadratiK.tools._utils")._extract_3d


class PKBC:
    """
    Poisson kernel-based clustering on the sphere.
    The class performs the Poisson kernel-based clustering algorithm
    on the sphere based on the Poisson kernel-based densities. It estimates
    the parameter of a mixture of Poisson kernel-based densities. The obtained
    estimates are used for assigning final memberships, identifying the data points.

    Parameters
    ----------
        num_clust : int, list, np.ndarray, range
            Number of clusters.

        max_iter : int, optional
            Maximum number of iterations before a run is terminated. Defaults to 300.

        stopping_rule : str, optional
            String describing the stopping rule to be used within each run.
            Currently must be either 'max', 'membership', or 'loglik'. Defaults to `loglik`.

        init_method : str, optional
            String describing the initialization method to be used.
            Currently must be 'sampledata'.

        num_init : int, optional
            Number of initializations. Defaults to 10.

        tol : float.
            Constant defining threshold by which log
            likelihood must change to continue iterations, if applicable.
            Defaults to 1e-7.

        random_state : int, None, optional.
            Determines random number generation for centroid initialization. Defaults to None.

        n_jobs : int
            Used only for computing the WCSS efficiently.
            n_jobs specifies the maximum number of concurrently running workers.
            If 1 is given, no joblib parallelism is used at all, which is useful for debugging.
            For more information on joblib n_jobs refer to -
            https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html.
            Defaults to 4.

    Attributes
    ----------
        alpha\\_ : dict
            Estimated mixing proportions.
            A dictionary containing key-value pairs, where each key is an element from the `num_clust` vector,
            and each value is a numpy.ndarray of shape (n_clusters,).


        labels\\_ : dict
            Final cluster membership assigned by the algorithm to each observation.
            A dictionary containing key-value pairs, where each key is an element from the `num_clust` vector,
            and each value is a numpy.ndarray of shape (n_samples,).

        log_lik_vecs\\_ : dict
            Array of log-likelihood values for each initialization.
            A dictionary containing key-value pairs, where each key is an element from the `num_clust` vector,
            and each value is a numpy.ndarray of shape (num_init, ).


        loglik\\_ : dict
            Maximum value of the log-likelihood function.
            A dictionary containing key-value pairs, where each key is an element from the `num_clust` vector,
            and each value is a float.


        mu\\_ : dict
            Estimated centroids.
            A dictionary containing key-value pairs, where each key is an element from the `num_clust` vector,
            and each value is a numpy.ndarray of shape (n_clusters, n_features).

        num_iter_per_runs\\_ : dict
            Number of E-M iterations per run.
            A dictionary containing key-value pairs, where each key is an element from the `num_clust` vector,
            and each value is a numpy.ndarray of shape (num_init, ).

        post_probs\\_ : dict
            Posterior probabilities of each observation for the indicated clusters.
            A dictionary containing key-value pairs, where each key is an element from the `num_clust` vector,
            and each value is a numpy.ndarray of shape (n_samples, num_clust).

        rho\\_ : dict
            Estimated concentration parameters rho.
            A dictionary containing key-value pairs, where each key is an element from the `num_clust` vector,
            and each value is a numpy.ndarray of shape (n_clusters,).

        euclidean\\_wcss\\_ : dict
            Values of within-cluster sum of squares computed with Euclidean distance.
            A dictionary containing key-value pairs, where each key is an element from the `num_clust` vector,
            and each value is a float.

        cosine\\_wcss\\_ : dict
            Values of within-cluster sum of squares computed with cosine similarity.
            A dictionary containing key-value pairs, where each key is an element from the `num_clust` vector,
            and each value is a float.

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
    >>> cluster_fit = PKBC(num_clust=4, random_state=42).fit(X)
    """

    def __init__(
        self,
        num_clust,
        max_iter=300,
        stopping_rule="loglik",
        init_method="sampledata",
        num_init=10,
        tol=1e-7,
        random_state=None,
        n_jobs=4,
    ):
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

        if not isinstance(self.num_clust, (int, range, list, np.ndarray)):
            raise Exception(
                "Please input only integer, list and np.ndarray for num_clust"
            )

        if isinstance(self.num_clust, int):
            if self.num_clust < 1:
                raise Exception("Input parameter num_clust must be greater than 1")
            else:
                self.num_clust = [self.num_clust]

        elif isinstance(self.num_clust, range):
            self.num_clust = list(self.num_clust)

        elif isinstance(self.num_clust, (list, np.ndarray)):
            if not all(isinstance(x, int) and x > 1 for x in self.num_clust):
                raise Exception(
                    "Input parameter num_clust must a list or np.ndarray where each element be greater than 1"
                )

        if self.max_iter < 1:
            raise Exception("Input parameter maxIter must be greater than 0")
        if self.stopping_rule not in ["max", "membership", "loglik"]:
            raise Exception(
                "Unrecognized value {} in input parameter.".format(self.stopping_rule)
            )
        if self.init_method not in ["sampledata"]:
            raise Exception(
                "Unrecognized value {} in input parameter.".format(self.init_method)
            )
        if self.num_init < 1:
            raise Exception("Input parameter numInit must be greater than 0")
        if not isinstance(self.random_state, (int, type(None))):
            raise ValueError("Please specify a integer or None random_state")

        generator = check_random_state(self.random_state)

        # set options for stopping rule
        check_membership = self.stopping_rule == "membership"
        check_loglik = self.stopping_rule == "loglik"

        # Normalize the data
        self.dat_copy = self.dat.copy()
        self.dat = self.dat / np.linalg.norm(self.dat, axis=1, keepdims=True)

        self.euclidean_wcss_ = {}
        self.cosine_wcss_ = {}
        self.post_probs_ = {}
        self.loglik_ = {}
        self.mu_ = {}
        self.rho_ = {}
        self.alpha_ = {}
        self.labels_ = {}
        self.log_lik_vecs_ = {}
        self.num_iter_per_runs_ = {}

        for k in self.num_clust:
            num_data, num_var = self.dat.shape
            self.log_lik_vec = np.repeat(float("-inf"), self.num_init)
            self.num_iter_per_run = np.repeat(-99, self.num_init)
            alpha_best = np.repeat(-99, k)
            rho_best = np.repeat(-99, k)
            mu_best = np.zeros((k, num_var))
            norm_prob_mat_best = np.full((num_data, k), -99)

            if self.init_method == "sampledata":
                unique_data = np.unique(self.dat, axis=0)
                num_unique_obs = unique_data.shape[0]
                if num_unique_obs < k:
                    raise ValueError(
                        "Only {} 'unique observations.', \
                        When init_method = {}, \
                            must have more than num_clust unique observations.".format(
                            num_unique_obs, self.init_method
                        )
                    )
            log_w_d = (num_var / 2) * (np.log(2) + np.log(np.pi)) - sp.gammaln(
                num_var / 2
            )

            for init in range(self.num_init):
                alpha_current = np.full(k, 1 / k)
                rho_current = np.full(k, 0.5)

                if self.init_method == "sampledata":
                    mu_current = unique_data[
                        generator.choice(num_unique_obs, size=k, replace=False),
                        :,
                    ]

                current_iter = 1
                memb_current = np.empty(num_data)
                log_lik_current = float("-inf")

                while current_iter <= self.max_iter:
                    v_mat = np.dot(self.dat, mu_current.T)
                    alpha_mat_current = np.tile(alpha_current, (num_data, 1))
                    rho_mat_current = np.tile(rho_current, (num_data, 1))
                    log_prob_mat_denom = np.log(
                        1
                        + rho_mat_current**2
                        - 2 * np.asarray(rho_mat_current) * np.asarray(v_mat)
                    )

                    log_prob_mat = (
                        np.log(1 - (rho_mat_current) ** 2)
                        - log_w_d
                        - (num_var / 2) * log_prob_mat_denom
                    )

                    prob_sum = np.tile(
                        np.dot(np.exp(log_prob_mat), alpha_current).reshape(
                            num_data, 1
                        ),
                        (1, k),
                    )

                    log_norm_prob_mat_current = (
                        np.log(alpha_mat_current) + log_prob_mat - np.log(prob_sum)
                    )

                    log_weight_mat = log_norm_prob_mat_current - log_prob_mat_denom
                    alpha_current = (
                        np.sum(np.exp(log_norm_prob_mat_current), axis=0) / num_data
                    )
                    mu_num_sum_mat = np.dot(np.exp(log_weight_mat).T, self.dat)
                    mu_denom = np.linalg.norm(mu_num_sum_mat, axis=1, keepdims=True)
                    mu_current = mu_num_sum_mat / mu_denom
                    for h in range(k):
                        sum_h_weight_mat = np.sum(np.exp(log_weight_mat[:, h]))
                        alpha_current_h = alpha_current[h]
                        mu_denom_h = mu_denom[h]
                        rho_current[h] = root_scalar(
                            root_func,
                            args=(
                                num_data,
                                alpha_current_h,
                                num_var,
                                mu_denom_h,
                                sum_h_weight_mat,
                            ),
                            bracket=[0, 1],
                            xtol=0.001,
                        ).root
                    if current_iter >= self.max_iter:
                        break

                    if check_membership:
                        memb_previous = memb_current
                        memb_current = np.argmax(
                            np.exp(log_norm_prob_mat_current), axis=1
                        )
                        if all(memb_previous == memb_current):
                            break

                    if check_loglik:
                        log_lik_previous = log_lik_current
                        log_lik_current = np.sum(
                            np.log(np.dot(np.exp(log_prob_mat), alpha_current))
                        )
                        if np.abs(log_lik_previous - log_lik_current) < self.tol:
                            break
                    current_iter = current_iter + 1
                log_lik_current = np.sum(
                    np.log(np.dot(np.exp(log_prob_mat), alpha_current))
                )

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
                for k in range(k)
            )
            self.euclidean_wcss_[k] = np.sum(cluster_euclidean_wcss)

            # cosine similarity wcss calculation
            cluster_cosine_wcss = Parallel(n_jobs=self.n_jobs)(
                delayed(calculate_wcss_cosine)(k, memb_best, self.dat, mu_best)
                for k in range(k)
            )

            self.cosine_wcss_[k] = np.sum(cluster_cosine_wcss)

            self.post_probs_[k] = norm_prob_mat_best
            self.loglik_[k] = max(self.log_lik_vec)
            self.mu_[k] = mu_best
            self.rho_[k] = rho_best
            self.alpha_[k] = alpha_best
            self.labels_[k] = memb_best
            self.log_lik_vecs_[k] = self.log_lik_vec
            self.num_iter_per_runs_[k] = self.num_iter_per_run
        return self

    def validation(self, y_true=None):
        """
        Computes validation metrics such as ARI, Macro Precision and Macro Recall when true labels are provided.

        Parameters
        -----------
            y_true : numpy.ndarray.
                Array of true memberships to clusters,
                Defaults to None.

        Returns
        --------
            validation metrics : tuple
                Return a tuple of a dictionary and elbow plots.
                The dictionary contains the following for different number of clusters:

                - **Adjusted Rand Index** : float (returned only when y_true is provided)
                    Adjusted Rand Index computed between the true and predicted cluster memberships.

                - **Macro Precision** : float (returned only when y_true is provided)
                    Macro Precision computed between the true and predicted cluster memberships.

                - **Macro Recall** : float (returned only when y_true is provided)
                    Macro Recall computed between the true and predicted cluster memberships.

                - **Average Silhouette Score** : float
                    Mean Silhouette Coefficient of all samples.

        References
        -----------
            Rousseeuw, P.J. (1987) Silhouettes: A graphical aid to the interpretation and validation of cluster analysis.
            Journal of Computational and Applied Mathematics, 20, 53–65.

        Notes
        ------
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

        validation_metrics = {}

        for k in self.num_clust:

            avg_silhouette_score = silhouette_score(self.dat, self.labels_[k])

            if y_true is not None:
                le = LabelEncoder()
                le.fit(y_true)
                y_true = le.transform(y_true)

                cm = confusion_matrix(y_true, self.labels_[k])
                cm_argmax = cm.argmax(axis=0)
                y_pred_ = np.array([cm_argmax[i] for i in self.labels_[k]])

                ari = adjusted_rand_score(y_true, self.labels_[k])
                macro_precision = precision_score(y_true, y_pred_, average="macro")
                macro_recall = recall_score(y_true, y_pred_, average="macro")
                validation_metrics[k] = [
                    ari,
                    macro_precision,
                    macro_recall,
                    avg_silhouette_score,
                ]
                validation_metrics_idx = {
                    "Metrics": [
                        "ARI",
                        "Macro Precision",
                        "Macro Recall",
                        "Average Silhouette Score",
                    ]
                }
            else:
                validation_metrics[k] = avg_silhouette_score
                validation_metrics_idx = {
                    "Metrics": [
                        "Average Silhouette Score",
                    ]
                }

        validation_metrics_df = pd.DataFrame(
            {**validation_metrics_idx, **validation_metrics}
        ).set_index("Metrics")

        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        axs[0].plot(self.euclidean_wcss_.keys(), self.euclidean_wcss_.values(), "--o")
        axs[0].set_xlabel("Number of Clusters")
        axs[0].set_ylabel("Within Cluster Sum of Squares (WCSS)")
        axs[0].set_title("Elbow Plot (Euclidean)")
        axs[1].plot(self.cosine_wcss_.keys(), self.cosine_wcss_.values(), "--o")
        axs[1].set_xlabel("Number of Clusters")
        axs[1].set_ylabel("Within Cluster Sum of Squares (WCSS)")
        axs[1].set_title("Elbow Plot (Cosine)")
        plt.tight_layout()
        plt.close()
        return (validation_metrics_df, fig)

    def stats_clusters(self, num_clust):
        """
        Function to generate descriptive statistics per variable (and per group if available).

        Parameters
        -----------
        num_clust : int
            Number of clusters for which the summary statistics should be shown.

        Returns
        -------
            summary_stats_df : pandas.DataFrame
                Dataframe of descriptive statistics.

        """
        summary_stats = stats(self.dat_copy, self.labels_[num_clust])
        return summary_stats

    def predict(self, X, num_clust):
        """
        Predict the cluster membership for each sample in X.

        Parameters
        -----------
            X : numpy.ndarray, pandas.DataFrame
                New data to predict membership.

            num_clust : int
                Number of clusters to be used for prediction.

        Returns
        --------
            (Cluster Probabilities, Membership) : tuple
                The first element of the tuple is the cluster probabilities of the input samples.
                The second element of the tuple is the predicted cluster membership of the new data.
        """
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        num_data, num_var = X.shape
        if self.dat.shape[1] != X.shape[1]:
            raise ValueError(
                f"X has {num_var} features, but PKBC is expecting {self.dat.shape[1]} features as input. Please provide same number of features as the fitted data."
            )
        log_w_d = (num_var / 2) * (np.log(2) + np.log(np.pi)) - sp.gammaln(num_var / 2)
        v_mat = np.dot(X, self.mu_[num_clust].T)
        alpha_mat_current = np.tile(self.alpha_[num_clust], (num_data, 1))
        rho_mat_current = np.tile(self.rho_[num_clust], (num_data, 1))
        log_prob_mat_denom = np.log(
            1 + rho_mat_current**2 - 2 * np.asarray(rho_mat_current) * np.asarray(v_mat)
        )
        log_prob_mat = (
            np.log(1 - (rho_mat_current) ** 2)
            - log_w_d
            - (num_var / 2) * log_prob_mat_denom
        )
        prob_sum = np.tile(
            np.dot(np.exp(log_prob_mat), self.alpha_[num_clust]).reshape(num_data, 1),
            (1, num_clust),
        )
        log_norm_prob_mat_current = (
            np.log(alpha_mat_current) + log_prob_mat - np.log(prob_sum)
        )
        log_weight_mat = log_norm_prob_mat_current - log_prob_mat_denom
        alpha_current = np.sum(np.exp(log_norm_prob_mat_current), axis=0) / num_data
        mu_num_sum_mat = np.dot(np.exp(log_weight_mat).T, X)
        mu_denom = np.linalg.norm(mu_num_sum_mat, axis=1, keepdims=True)
        for h in range(num_clust):
            sum_h_weight_mat = np.sum(np.exp(log_weight_mat[:, h]))
            alpha_current_h = alpha_current[h]
            mu_denom_h = mu_denom[h]
            self.rho_[num_clust][h] = root_scalar(
                root_func,
                args=(
                    num_data,
                    alpha_current_h,
                    num_var,
                    mu_denom_h,
                    sum_h_weight_mat,
                ),
                bracket=[0, 1],
                xtol=0.001,
            ).root
        norm_prob_mat_best = np.exp(log_norm_prob_mat_current)
        memb_best = np.argmax(norm_prob_mat_best, axis=1)

        return (norm_prob_mat_best, memb_best)

    def plot(self, num_clust, y_true=None):
        """
        The method plot creates a 2D or 3D scatter plot with a circle or sphere
        as the surface and data points plotted on it.

        Parameters
        ----------
            num_clust : int
                Specifies the number of clusters to visualize.

            y_true : numpy.ndarray, list, pandas.series, optional
                - If `y_true` is None, then only clusters colored according to the predicted labels.
                - If `y_true` is provided, clusters are colored according to the predicted and true labels in different subplots.

        Returns
        -------
            Returns a 2D matplotlib figure object or 3D plotly figure object with data points plotted on it.
        """
        if self.dat.shape[1] < 2:
            raise Exception(
                "Plot is not implemented when dimensionality is less than 2."
            )

        if y_true is not None:
            if not isinstance(y_true, (list, np.ndarray, pd.DataFrame)):
                raise TypeError("The y_true must be a list, np.ndarray or pd.DataFrame")

            if isinstance(y_true, pd.DataFrame):
                y_true = y_true.to_numpy().flatten()
            elif isinstance(y_true, pd.Series):
                y_true = y_true.values
            elif isinstance(y_true, np.ndarray):
                if y_true.ndim == 1:
                    pass
                elif y_true.ndim == 2:
                    y_true = y_true.flatten()

        if num_clust not in self.num_clust:
            raise ValueError(
                "Please input correct number of clusters for which you want to see visualization"
            )

        if y_true is not None:
            if self.dat.shape[1] == 2:
                theta = np.linspace(0, 2 * np.pi, 100)
                unit_circle_x = np.cos(theta)
                unit_circle_y = np.sin(theta)
                fig, axes = plt.subplots(1, 2, figsize=(14, 6))
                axes[0].scatter(
                    self.dat[:, 0],
                    self.dat[:, 1],
                    c=self.labels_[num_clust],
                    cmap="viridis",
                    edgecolors="k",
                )
                axes[0].plot(
                    unit_circle_x, unit_circle_y, linestyle="dashed", color="red"
                )
                axes[0].set_xlabel("Feature 1")
                axes[0].set_ylabel("Feature 2")
                axes[0].set_title("Plot with Predicted Labels")
                axes[1].scatter(
                    self.dat[:, 0], self.dat[:, 1], c=y_true, edgecolors="k"
                )
                axes[1].plot(
                    unit_circle_x, unit_circle_y, linestyle="dashed", color="red"
                )
                axes[1].set_xlabel("Feature 1")
                axes[1].set_ylabel("Feature 2")
                axes[1].set_title("Plot with True Labels")
                plt.tight_layout()
                plt.close()
                return fig
            else:
                xx, yy, zz = extract3d(self.dat)
                r = 1
                pi = np.pi
                cos = np.cos
                sin = np.sin
                phi, theta = np.mgrid[0.0:pi:100j, 0.0 : 2.0 * pi : 100j]
                x1 = r * sin(phi) * cos(theta)
                y1 = r * sin(phi) * sin(theta)
                z1 = r * cos(phi)

                fig = make_subplots(
                    rows=1,
                    cols=2,
                    specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]],
                    subplot_titles=(
                        "Colored by Predicted Class",
                        "Colored by True Class",
                    ),
                )
                fig.add_trace(
                    go.Surface(
                        x=x1,
                        y=y1,
                        z=z1,
                        colorscale=[[0, "#DCDCDC"], [1, "#DCDCDC"]],
                        opacity=0.5,
                        showscale=False,
                    ),
                    row=1,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter3d(
                        x=xx,
                        y=yy,
                        z=zz,
                        mode="markers",
                        marker=dict(
                            size=5,
                            color=self.labels_[num_clust],
                            colorscale="turbo",
                            showscale=False,
                        ),
                    ),
                    row=1,
                    col=1,
                )

                fig.add_trace(
                    go.Surface(
                        x=x1,
                        y=y1,
                        z=z1,
                        colorscale=[[0, "#DCDCDC"], [1, "#DCDCDC"]],
                        opacity=0.5,
                        showscale=False,
                    ),
                    row=1,
                    col=2,
                )
                fig.add_trace(
                    go.Scatter3d(
                        x=xx,
                        y=yy,
                        z=zz,
                        mode="markers",
                        marker=dict(
                            size=5,
                            color=y_true,
                            colorscale="cividis",
                            showscale=False,
                        ),
                    ),
                    row=1,
                    col=2,
                )
                fig.update_layout(
                    title="",
                    scene=dict(
                        xaxis=dict(range=[-1, 1]),
                        yaxis=dict(range=[-1, 1]),
                        zaxis=dict(range=[-1, 1]),
                        aspectmode="data",
                    ),
                    showlegend=False,
                )
                return fig
        else:
            if self.dat.shape[1] == 2:
                theta = np.linspace(0, 2 * np.pi, 100)
                unit_circle_x = np.cos(theta)
                unit_circle_y = np.sin(theta)
                fig, ax = plt.subplots(1, 1, figsize=(8, 8))
                ax.scatter(
                    self.dat[:, 0],
                    self.dat[:, 1],
                    cmap="viridis",
                    edgecolors="k",
                )
                ax.plot(unit_circle_x, unit_circle_y, linestyle="dashed", color="red")
                ax.set_xlabel("Feature 1")
                ax.set_ylabel("Feature 2")
                ax.set_title("Plot with Predicted Labels")
                plt.tight_layout()
                plt.close()
                return fig
            else:
                xx, yy, zz = extract3d(self.dat)
                r = 1
                pi = np.pi
                cos = np.cos
                sin = np.sin
                phi, theta = np.mgrid[0.0:pi:100j, 0.0 : 2.0 * pi : 100j]
                x1 = r * sin(phi) * cos(theta)
                y1 = r * sin(phi) * sin(theta)
                z1 = r * cos(phi)

                fig = make_subplots(
                    rows=1,
                    cols=1,
                    specs=[[{"type": "scatter3d"}]],
                    subplot_titles=(
                        "Colored by Predicted Class",
                        "Colored by True Class",
                    ),
                )
                fig.add_trace(
                    go.Surface(
                        x=x1,
                        y=y1,
                        z=z1,
                        colorscale=[[0, "#DCDCDC"], [1, "#DCDCDC"]],
                        opacity=0.5,
                        showscale=False,
                    ),
                    row=1,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter3d(
                        x=xx,
                        y=yy,
                        z=zz,
                        mode="markers",
                        marker=dict(
                            size=5,
                            color=self.labels_[num_clust],
                            colorscale="turbo",
                            showscale=False,
                        ),
                    ),
                    row=1,
                    col=1,
                )
                fig.update_layout(
                    title="",
                    scene=dict(
                        xaxis=dict(range=[-1, 1]),
                        yaxis=dict(range=[-1, 1]),
                        zaxis=dict(range=[-1, 1]),
                        aspectmode="data",
                    ),
                    showlegend=False,
                )
                return fig

    def summary(self, print_fmt="simple"):
        """
        Summary function generates a table for the PKBC clustering.

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
                format with the Loglikelihood, Euclidean WCSS, Cosine WCSS, Number of data
                points in each cluster, and mixing proportion for the different number of clusters.
        """
        df = pd.DataFrame(
            columns=[
                "Loglikelihood",
                "Euclidean WCSS",
                "Cosine WCSS",
                "Num Data Point/Cluster",
                "Mixing Proportions (alpha)",
            ]
        )
        df["Loglikelihood"] = pd.Series(
            {k: np.round(v, 2) for k, v in self.loglik_.items()}
        )
        df["Euclidean WCSS"] = pd.Series(
            {k: np.round(v, 2) for k, v in self.euclidean_wcss_.items()}
        )
        df["Cosine WCSS"] = pd.Series(
            {k: np.round(v, 2) for k, v in self.cosine_wcss_.items()}
        )
        df["Num Data Point/Cluster"] = pd.Series(
            {k: dict(Counter(v)) for k, v in self.labels_.items()}
        )
        df["Mixing Proportions (alpha)"] = pd.Series(
            {k: np.round(v, 2) for k, v in self.alpha_.items()}
        )
        return tabulate(df, headers=df.columns, tablefmt=print_fmt)
