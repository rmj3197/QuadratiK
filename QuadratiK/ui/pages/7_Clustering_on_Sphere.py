"""
Contains the PKBC functionality of the UI
"""

import importlib
import copy
from io import BytesIO
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.parallel import Parallel, delayed


kt = importlib.import_module("QuadratiK.kernel_test").KernelTest
qq_plot = importlib.import_module("QuadratiK.tools").qq_plot
stats = importlib.import_module("QuadratiK.tools").stats

pkbc = importlib.import_module("QuadratiK.spherical_clustering").PKBC
sphere3d = importlib.import_module("QuadratiK.tools").sphere3d
circle2d = importlib.import_module("QuadratiK.tools").plot_clusters_2d
utils = importlib.import_module("QuadratiK.tools")._utils


st.title("Poisson Kernel based Clustering")
st.write(
    "Performs the Poisson kernel-based clustering algorithm on the Sphere based on \
    the Poisson kernel-based densities"
)

with st.expander("Click to view code"):
    code_python = """
    # In case you do not have the true labels, do not read y.
    X,y = Read the data and the cluster label files. 

    from QuadratiK.spherical_clustering import PKBC
    cluster_fit = PKBC(num_clust = Input the number of clusters).fit(X)
    """
    st.code(code_python, language="python")

    code_R = """
    library(QuadratiK)
    data <- Read Data here
    res_pk <- pkbc(as.matrix(data),2:10)
    labels <- Read labels here
    res_validation <- validation(res_pk, true_label = labels)
    """
    st.code(code_R, language="r")

head = st.checkbox("**Select, if the header is present in the data file.**", value=True)
delim = st.text_input("**Enter the delimiter**", ",")
data = st.file_uploader(
    "**Please Upload the data file**",
    accept_multiple_files=False,
    type=[".csv", ".txt"],
)

if data is not None:
    st.success(data.name + " Uploaded Successfully!")
    if head:
        data = pd.read_csv(data, sep=delim)
    else:
        data = pd.read_csv(data, sep=delim, header=None)
    data.columns = range(data.shape[1])
    y_true_available = st.checkbox(
        "**Select the checkbox if true labels for the data points are available**",
        value=False,
    )

    col_number = None
    if y_true_available:
        try:
            col_number = int(
                st.number_input(
                    "**Enter the column in the datafile that contains the label (start from 0)**",
                    value=None,
                    step=1,
                )
            )
        except:
            st.error("Enter a valid column number")

    else:
        pass
    num_clusters = int(
        st.number_input("**Enter the number of clusters**", value=2, step=1)
    )

if data is not None:
    if y_true_available:
        if col_number is not None:
            try:
                x = copy.copy(data.drop(columns=[col_number]))
                x = np.array(x, dtype=np.float64)
                y = data[col_number]
                le = LabelEncoder()
                le.fit(y)
                y = le.transform(y)
            except:
                st.error("Please check the input data file")
    else:
        x = copy.copy(data)

    try:
        with st.spinner("getting results ready..."):
            cluster_fit = pkbc(num_clust=num_clusters).fit(x)
            y_pred = cluster_fit.labels_

            validation_metrics, elbow_plot = cluster_fit.validation(y)

            st.dataframe(validation_metrics)

            st.write("**Results:**")
            clustering_results_json = {}
            clustering_results_json["post_probs"] = cluster_fit.post_probs_
            clustering_results_json["loglik"] = cluster_fit.loglik_
            clustering_results_json["mu"] = cluster_fit.mu_
            clustering_results_json["alpha"] = cluster_fit.alpha_
            clustering_results_json["rho"] = cluster_fit.rho_
            clustering_results_json["final_membership"] = cluster_fit.labels_
            clustering_results_json["euclidean_wcss"] = cluster_fit.euclidean_wcss_
            clustering_results_json["cosine_wcss"] = cluster_fit.euclidean_wcss_
            clustering_results_json["log_lik_vec"] = cluster_fit.log_lik_vec
            clustering_results_json["num_iter_per_run"] = cluster_fit.num_iter_per_run
            clustering_results_json["num_clust"] = cluster_fit.num_clust
            clustering_results_json["max_iter"] = cluster_fit.max_iter
            clustering_results_json["stopping_rule"] = cluster_fit.stopping_rule
            clustering_results_json["init_method"] = cluster_fit.init_method
            clustering_results_json["num_init"] = cluster_fit.num_init

            st.json(clustering_results_json, expanded=False)
            st.download_button(
                "Click to Download the clustering results",
                str(clustering_results_json),
                "Clustering Results.txt",
                "txt",
                key="download-cluster-res",
            )

            st.success("Done!")
    except:
        st.error("Please check the input data file")

st.header("K-Sample Test for the identified clusters", divider="grey")

num_iter = st.number_input(
    "Enter a value of number of iterations to be used for critical value estimation",
    value=500,
)
h = st.number_input("Enter a value of tuning parameter h", value=1)
b = st.number_input(
    "Enter a value for the proportion of subsampling samples to be used", value=0.9
)

if data is not None:
    num_iter = int(num_iter)
    h_val = float(h)
    b = float(b)

    try:
        x_copy = x / np.linalg.norm(x, axis=1, keepdims=True)
        y_pred = cluster_fit.labels_[num_clusters]

        with st.spinner("getting results ready..."):
            k_samp_test = kt(
                h=h_val, num_iter=num_iter, b=b, centering_type="nonparam"
            ).test(x=x_copy, y=y_pred)

            st.write("**Results:**")

            res = pd.DataFrame()
            res["Value"] = [
                k_samp_test.test_type_,
                k_samp_test.dn_test_statistic_,
                k_samp_test.dn_cv_,
                k_samp_test.dn_h0_rejected_,
                k_samp_test.var_dn_,
                k_samp_test.trace_test_statistic_,
                k_samp_test.trace_cv_,
                k_samp_test.trace_h0_rejected_,
                k_samp_test.var_trace_,
            ]
            res = res.set_axis(
                [
                    "Test Type",
                    "Dn Test Statistic",
                    "Dn Critical Value",
                    "Dn Reject H0",
                    "Var Dn",
                    "Trace Test Statistic",
                    "Trace Critical Value",
                    "Trace Reject H0",
                    "Var Trace",
                ]
            )

            st.dataframe(res, width=800)
            csv_res = res.to_csv().encode()
            st.download_button(
                "Click to Download the test results",
                csv_res,
                "K_Sample_Test_results.csv",
                "text/csv",
                key="download-test-res",
            )

            summary_stats_df = k_samp_test.stats()
            st.dataframe(summary_stats_df)

            csv_stats = summary_stats_df.to_csv(index=True).encode()
            st.download_button(
                "Click to Download the summary statistics",
                csv_stats,
                "Statistics.csv",
                "text/csv",
                key="download-summary",
            )
    except:
        st.error("Check data file and selected parameters.")

st.header("Visualizations", divider="grey")

st.subheader("Elbow Plot")

with st.expander("Click to view code"):
    elbow_code = """
    import matplotlib.pyplot as plt
    wcss_list = []
    for clus in range(2,10):
        cluster_fit = PKBC(num_clust=clus).fit(X)
        wcss_list.append(cluster_fit.euclidean_wcss_)
        
    plt.plot(list(range(2,10)),wcss_list, "--o")
    plt.xlabel("Number of Cluster")
    plt.ylabel("Within Cluster Sum of Squares (WCSS)")
    plt.title("Elbow Plot")
    """
    st.code(elbow_code, language="python")


def get_wcss_euclid(x, k):
    cluster_fitting = pkbc(num_clust=k).fit(x)
    return cluster_fitting.euclidean_wcss_[k]


def get_wcss_cosine(x, k):
    cluster_fitting = pkbc(num_clust=k).fit(x)
    return cluster_fitting.cosine_wcss_[k]


if data is not None:
    elbow_clusters = int(
        st.number_input(
            "Enter the total number of clusters for which elbow plot to be shown",
            value=10,
            step=1,
        )
    )
    with st.spinner("Generating the elbow plot ..."):
        try:
            wcss_list_euclid = Parallel(n_jobs=4)(
                delayed(get_wcss_euclid)(x, k) for k in range(2, elbow_clusters)
            )

            wcss_list_cosine = Parallel(n_jobs=4)(
                delayed(get_wcss_cosine)(x, k) for k in range(2, elbow_clusters)
            )
            fig, axs = plt.subplots(1, 2, figsize=(15, 6))
            axs[0].plot(list(range(2, elbow_clusters)), wcss_list_euclid, "--o")
            axs[0].set_xlabel("Number of Cluster")
            axs[0].set_ylabel("Euclidean Within Cluster Sum of Squares (WCSS)")
            axs[0].set_title("Elbow Plot WCSS Euclidean")

            axs[1].plot(list(range(2, elbow_clusters)), wcss_list_cosine, "--o")
            axs[1].set_xlabel("Number of Cluster")
            axs[1].set_ylabel(
                "Euclidean Within Cluster Sum of Squares (Cosine Similarity)"
            )
            axs[1].set_title("Elbow Plot WCSS Cosine Similarity")
            st.pyplot(fig)
        except:
            st.error("Please ensure that the data file is loaded")


st.subheader("Data on Sphere")
with st.expander("Click to view code"):
    viz_code = """
    from QuadratiK.tools import sphere3d
    sphere3d(X,y)

    # or in case the input data is 2d

    from QuadratiK.tools import plot_clusters_2d
    plot_clusters_2d(X,y)
    """
    st.code(viz_code, language="python")

if data is not None:
    try:
        if x.shape[1] > 2:
            with st.spinner("Plotting the data points on sphere ..."):
                r = 1
                pi = np.pi
                cos = np.cos
                sin = np.sin
                phi, theta = np.mgrid[0.0:pi:100j, 0.0 : 2.0 * pi : 100j]
                x1 = r * sin(phi) * cos(theta)
                y1 = r * sin(phi) * sin(theta)
                z1 = r * cos(phi)
                xx, yy, zz = utils._extract_3d(x)

                fig = make_subplots(
                    rows=1,
                    cols=2,
                    specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]],
                    subplot_titles=(
                        "Colored by Predicted Class",
                        "Colored by True Class",
                    ),
                )
                fig.append_trace(
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
                fig.append_trace(
                    go.Scatter3d(
                        x=xx,
                        y=yy,
                        z=zz,
                        mode="markers",
                        marker=dict(
                            size=5, color=y_pred, colorscale="turbo", showscale=False
                        ),
                    ),
                    row=1,
                    col=1,
                )

                if y_true_available:
                    fig.append_trace(
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
                    fig.append_trace(
                        go.Scatter3d(
                            x=xx,
                            y=yy,
                            z=zz,
                            mode="markers",
                            marker=dict(
                                size=5, color=y, colorscale="cividis", showscale=False
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
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        else:
            fig1 = plt.figure(figsize=(8, 6))
            fig1 = circle2d(x, y_pred)
            plt.title("Colored by Predicted Class")
            st.pyplot(fig1)

            if y_true_available:
                fig2 = plt.figure(figsize=(8, 6))
                fig2 = circle2d(x, y)
                plt.title("Colored by True Class")
                st.pyplot(fig2)
    except:
        st.error("Please ensure that the data file is loaded")

st.markdown(
    r"""
    <style>
        .reportview-container {
            margin-top: -2em;
        }
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
    </style>
""",
    unsafe_allow_html=True,
)
