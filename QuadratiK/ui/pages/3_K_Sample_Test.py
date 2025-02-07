"""
Contains the K-Sample test functionality of the UI
"""

import importlib

import pandas as pd
import streamlit as st

kt = importlib.import_module("QuadratiK.kernel_test").KernelTest
qq_plot = importlib.import_module("QuadratiK.tools").qq_plot
stats = importlib.import_module("QuadratiK.tools").stats


@st.cache_data(ttl=30, show_spinner=False)
def run_ksample_test(h_val, num_iter, b, X, y):
    return kt(h=h_val, num_iter=num_iter, b=b, centering_type="nonparam").test(x=X, y=y)


st.title("K Sample Test")
st.write("Performs the Nonparametric K-Sample Test")

with st.expander("Click to view example code in Python and R"):
    code_python = """
    import numpy as np

    np.random.seed(0)
    from QuadratiK.kernel_test import KernelTest

    size = 200
    eps = 1
    x1 = np.random.multivariate_normal(
        mean=[0, np.sqrt(3) * eps / 3], cov=np.eye(2), size=size
    )
    x2 = np.random.multivariate_normal(
        mean=[-eps / 2, -np.sqrt(3) * eps / 6], cov=np.eye(2), size=size
    )
    x3 = np.random.multivariate_normal(
        mean=[eps / 2, -np.sqrt(3) * eps / 6], cov=np.eye(2), size=size
    )
    # Merge the three samples into a single dataset
    X_k = np.concatenate([x1, x2, x3])
    # The memberships are needed for k-sample test
    y_k = np.repeat(np.array([1, 2, 3]), size).reshape(-1, 1)

    # performing the k-sample test
    k_sample_test = KernelTest(h=1.5, method="subsampling", random_state=42).test(X_k, y_k)

    # printing the summary for the k-sample test
    print(k_sample_test.summary())
    """
    st.code(code_python, language="python")

    code_R = """
    library(mvtnorm)
    library(QuadratiK)
    library(ggplot2)
    sizes <- rep(50,3)
    eps <- 1
    set.seed(2468)
    x1 <- rmvnorm(sizes[1], mean = c(0,sqrt(3)*eps/3))
    x2 <- rmvnorm(sizes[2], mean = c(-eps/2,-sqrt(3)*eps/6))
    x3 <- rmvnorm(sizes[3], mean = c(eps/2,-sqrt(3)*eps/6))
    x <- rbind(x1, x2, x3)
    y <- as.factor(rep(c(1, 2, 3), times = sizes))
    k_test <- kb.test(x = x, y = y, h = 2)
    show(k_test)
    """
    st.code(code_R, language="r")

st.subheader("Input Instructions", divider="grey")
st.write("1. Upload the data file in .txt or .csv format for both the X and Y datasets.")
st.write(
    "2. The file may contain a header (see image below for reference). If headers are present, check the box. The checkbox is selected by default. Please ensure that both X and Y either contain headers or neither contain headers."
)
st.write("3. Specify the separator or delimiter used in both the X and Y datasets; the default is a comma (,).")
st.write("4. Once the data is uploaded, specify the column in the data file that contains the labels.")
st.write("5. Furthermore please specify the values of bandwidth parameter, proportion of subsampling samples to be used, and number of iterations for critical value estimation. Default values are provided.")

st.image(
    str(
        importlib.resources.files("QuadratiK.ui").joinpath(
            "pages/assets/ksample_test_format.png"
        )
    ),
    caption="Sample data format for k-sample test",
    use_container_width=True,
)


delim = st.text_input("**Enter the delimiter**", ",")
header_exist = st.checkbox(
    "**Select, if the header is present in the data file.**", value=True
)

if header_exist is False:
    header = None
else:
    header = "infer"

data = st.file_uploader(
    "Please Upload the data file", accept_multiple_files=False, type=[".csv", ".txt"]
)

if data is not None:
    st.success(data.name + " Uploaded Successfully")
    try:
        data = pd.read_csv(data, sep=delim, header=header)
    except Exception as e:
        st.error(f"An error occurred: {e}")

    data = data.values
    label_col = data.shape[1] - 1

if data is not None:
    col_number = int(
        st.number_input(
            "Enter the column in the datafile that \
            contains the label (start from 0)",
            value=label_col,
            step=1,
        )
    )

if data is not None:
    num_iter = st.number_input(
        "Enter a value of number of iterations to be used for critical value estimation",
        value=500,
    )
    h = st.number_input("Enter a value of tuning parameter h", value=1)
    b = st.number_input(
        "Enter a value for the proportion of subsampling samples to be used", value=0.9
    )

    with st.spinner("getting results ready..."):
        num_iter = int(num_iter)
        h_val = float(h)
        b = float(b)

        X = data[:, :col_number]
        y = data[:, col_number]

        h = 0.5
        try:
            k_samp_test = run_ksample_test(h_val, num_iter, b, X, y)

            index_labels = [
                "Test Statistic",
                "Critical Value",
                "H0 is rejected (1 = True, 0 = False)",
            ]
            test_summary = {
                "Dn": [
                    k_samp_test.dn_test_statistic_,
                    k_samp_test.dn_cv_,
                    k_samp_test.dn_h0_rejected_,
                ],
                "Trace": [
                    k_samp_test.trace_test_statistic_,
                    k_samp_test.trace_cv_,
                    k_samp_test.trace_h0_rejected_,
                ],
            }
            res = pd.DataFrame(test_summary, index=index_labels)

            st.text(k_samp_test.test_type_)
            st.table(res)
            csv_res = res.to_csv().encode()
            st.download_button(
                "Click to download the test results",
                csv_res,
                "K_Sample_Test_results.csv",
                "text/csv",
                key="download-txt",
            )

            st.subheader("Summary Statistics", divider="grey")
            summary_stats_df = k_samp_test.stats()
            st.dataframe(summary_stats_df)
            # summary_stats_df = pd.concat(statistics.values(), keys=statistics.keys(), axis=0)
            csv_stats = summary_stats_df.to_csv(index=True).encode()
            st.download_button(
                "Click to Download the summary statistics",
                csv_stats,
                "Statistics.csv",
                "text/csv",
                key="download-csv",
            )
            st.success("Done!")
        except Exception as e:
            st.error(f"An error occurred: {e}")

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
