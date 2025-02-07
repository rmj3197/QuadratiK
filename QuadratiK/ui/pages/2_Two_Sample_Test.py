"""
Contains the Two Sample test functionality of the UI
"""

import importlib

import pandas as pd
import streamlit as st

kt = importlib.import_module("QuadratiK.kernel_test").KernelTest
qq_plot = importlib.import_module("QuadratiK.tools").qq_plot
pd.set_option("display.float_format", "{:.2g}".format)


@st.cache_data(ttl=30, show_spinner=False)
def run_twosample_test(h_val, num_iter, b, X, Y):
    return kt(h=h_val, num_iter=num_iter, b=b, centering_type="nonparam").test(x=X, y=Y)


st.title("Two Sample Test")

st.write("Performs the Nonparametric Two Sample Test")

with st.expander("Click to view example code in Python and R"):
    code_python = """
    import numpy as np

    np.random.seed(0)
    from scipy.stats import skewnorm

    from QuadratiK.kernel_test import KernelTest

    # data generation
    X_2 = np.random.multivariate_normal(mean=np.zeros(4), cov=np.eye(4), size=200)
    Y_2 = skewnorm.rvs(
    size=(200, 4),
    loc=np.zeros(4),
    scale=np.ones(4),
    a=np.repeat(0.5, 4),
    random_state=20,
    )
    # performing the two sample test
    two_sample_test = KernelTest(h=2, num_iter=150, random_state=42).test(X_2, Y_2)

    # printing the summary for the two sample test
    print(two_sample_test.summary())
    """
    st.code(code_python, language="python")

    code_R = """
    library(sn)         # For generating skew-normal distributed data
    library(mvtnorm)    # For generating multivariate normal data
    library(QuadratiK)  

    # Set parameters
    n <- 100           # Number of samples
    d <- 4             # Dimension of the data
    skewness_y <- 0.5  # Skewness parameter for Y distribution

    # Set seed for reproducibility
    set.seed(2468)

    # Generate multivariate normal data for X
    x_2 <- rmvnorm(n, mean = rep(0, d))  # Mean vector of zeros, identity covariance

    # Generate skew-normal data for Y
    y_2 <- rmsn(n = n, xi = 0, Omega = diag(d), alpha = rep(skewness_y, d))  

    # Perform a statistical test to compare the two datasets
    two_test <- kb.test(x = x_2, y = y_2, h = 2)

    # Output the test result
    two_test
    """
    st.code(code_R, language="r")

st.subheader("Input Instructions", divider="grey")
st.write("1. Upload the data file in .txt or .csv format for both the X and Y datasets.")
st.write(
    "2. The file may contain a header (see image below for reference). If headers are present, check the box. The checkbox is selected by default. Please ensure that both X and Y either contain headers or neither contain headers."
)
st.write("3. Specify the separator or delimiter used in both the X and Y datasets; the default is a comma (,).")
st.write("4. Once the data files are uploaded, specify the values of bandwidth parameter, proportion of subsampling samples to be used, and number of iterations for critical value estimation. Default values are provided.")

st.image(
    str(
        importlib.resources.files("QuadratiK.ui").joinpath(
            "pages/assets/two_sample_test_format.png"
        )
    ),
    caption="Sample data format for two sample test",
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

x_data = st.file_uploader(
    "Choose data file for X",
    accept_multiple_files=False,
    type=[".txt", ".csv"],
)

if x_data is not None:
    st.write(x_data.name + " Uploaded Successfully")
    try:
        X = pd.read_csv(x_data, sep=delim, header=header)
    except Exception as e:
        st.error(f"An error occurred: {e}")

y_data = st.file_uploader(
    "Choose data file for Y", accept_multiple_files=False, type=[".txt", ".csv"]
)

if y_data is not None:
    st.write(y_data.name + " Uploaded Successfully")
    try:
        Y = pd.read_csv(y_data, sep=delim, header=header)
    except Exception as e:
        st.error(f"An error occurred: {e}")

if (y_data) and (x_data) is not None:
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

        try:
            two_sample_test = run_twosample_test(h_val, num_iter, b, X, Y)
            index_labels = [
                "Test Statistic",
                "Critical Value",
                "H0 is rejected (1 = True, 0 = False)",
            ]
            test_summary = {
                "Dn": [
                    two_sample_test.dn_test_statistic_,
                    two_sample_test.dn_cv_,
                    two_sample_test.dn_h0_rejected_,
                ],
                "Trace": [
                    two_sample_test.trace_test_statistic_,
                    two_sample_test.trace_cv_,
                    two_sample_test.trace_h0_rejected_,
                ],
            }
            res = pd.DataFrame(test_summary, index=index_labels)

            st.text(two_sample_test.test_type_)
            st.table(res)
            csv_res = res.to_csv().encode()
            st.download_button(
                "Click to Download the test results",
                csv_res,
                "Two_Sample_Test_results.csv",
                "text/csv",
                key="download-txt",
            )

            st.subheader("Summary Statistics", divider="grey")
            summary_stats_df = two_sample_test.stats()
            st.table(summary_stats_df)
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


st.header("QQ Plots", divider="grey")
with st.expander("Click to view code"):
    viz_code = """
    from QuadratiK.tools import qq_plot
    qq_plot(X,Y)
    """
    st.code(viz_code, language="python")

if (y_data) and (x_data) is not None:
    try:
        st.pyplot(qq_plot(X, Y), use_container_width=True)
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
