"""
Contains the Normality test functionality of the UI
"""

import importlib

import pandas as pd
import streamlit as st

kt = importlib.import_module("QuadratiK.kernel_test").KernelTest
qq_plot = importlib.import_module("QuadratiK.tools").qq_plot


@st.cache_data(ttl=30, show_spinner=False)
def run_normality_test(h_val, num_iter, b, x):
    return kt(h=h_val, num_iter=num_iter, b=b, centering_type="param").test(x=x)


st.title("Normality Test")
st.write("Performs the Parametric Multivariate Normality Test.")

with st.expander("Click to view example code in Python and R"):
    code_python = """
    # Example of performing the normality test using QuadratiK in Python
    import numpy as np

    np.random.seed(78990)
    from QuadratiK.kernel_test import KernelTest

    # data generation
    data_norm = np.random.multivariate_normal(mean=np.zeros(4), cov=np.eye(4), size=500)

    # performing the normality test
    normality_test = KernelTest(
    h=0.4, num_iter=150, method="subsampling", random_state=42
    ).test(data_norm)

    # printing the summary for normality test
    print(normality_test.summary())
    """
    st.code(code_python, language="python")

    code_R = """
    # Example of performing the normality test using QuadratiK in R
    library(QuadratiK)
    
    # random data generation
    x <- matrix(rnorm(100,4), ncol = 2)
    
    # performing the normality test
    kb.test(x, mu_hat = c(4,4), Sigma_hat = diag(2), h = 0.4)
    """
    st.code(code_R, language="r")

st.subheader("Input Instructions", divider="grey")

st.write("1. Upload the data file in .txt or .csv format.")
st.write(
    "2. The file may contain a header (see image below for reference). If headers are present, check the box. The checkbox is selected by default."
)
st.write("3. Specify the separator or delimiter used; the default is a comma (,).")
st.write(
    "4. Once the data is uploaded, specify the values of bandwidth parameter, proportion of subsampling samples to be used, and number of iterations for critical value estimation. Default values are provided."
)

st.image(
    str(
        importlib.resources.files("QuadratiK.ui").joinpath(
            "pages/assets/normality_test_format.png"
        )
    ),
    caption="Sample data format for normality test",
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
    "Please Upload the data file", accept_multiple_files=False, type=[".txt", ".csv"]
)

if data is not None:
    st.success(data.name + " Uploaded Successfully!")
    try:
        x = pd.read_csv(data, sep=delim, header=header)
    except Exception as e:
        st.error(f"An error occurred: {e}")

    num_iter = st.number_input(
        "Enter a value of number of iterations to be used for critical value estimation",
        value=150,
    )
    h = st.number_input("Enter a value of tuning parameter h", value=0.5)
    b = st.number_input(
        "Enter a value for the proportion of subsampling samples to be used", value=0.9
    )

    with st.spinner("getting results ready..."):
        num_iter = int(num_iter)
        h_val = float(h)
        b = float(b)

        try:
            norm_test = run_normality_test(h_val, num_iter, b, x)
            index_labels = [
                "Test Statistic",
                "Critical Value",
                "H0 is rejected (1 = True, 0 = False)",
            ]
            test_summary = {
                "U-Statistic": [
                    norm_test.un_test_statistic_,
                    norm_test.un_cv_,
                    norm_test.un_h0_rejected_,
                ],
                "V-Statistic": [
                    norm_test.vn_test_statistic_,
                    norm_test.vn_cv_,
                    norm_test.vn_h0_rejected_,
                ],
            }
            res = pd.DataFrame(test_summary, index=index_labels)

            st.text(norm_test.test_type_)
            st.table(res)
            csv_res = res.to_csv().encode()
            st.download_button(
                "Click to Download the test results",
                csv_res,
                "Normality_Test_results.csv",
                "text/csv",
                key="download-txt",
            )

            st.subheader("Summary Statistics", divider="grey")
            summary_stats_df = norm_test.stats()
            st.dataframe(summary_stats_df)
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
    qq_plot(X)
    """
    st.code(viz_code, language="python")

if data is not None:
    try:
        st.pyplot(qq_plot(x), use_container_width=True)
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
