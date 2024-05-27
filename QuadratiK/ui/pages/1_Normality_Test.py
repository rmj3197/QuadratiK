"""
Contains the Normality test functionality of the UI
"""

import importlib
import streamlit as st
import pandas as pd

kt = importlib.import_module("QuadratiK.kernel_test").KernelTest
qq_plot = importlib.import_module("QuadratiK.tools").qq_plot


@st.cache_data(ttl=30, show_spinner=False)
def run_normality_test(h_val, num_iter, b, x):
    return kt(h=h_val, num_iter=num_iter, b=b, centering_type="nonparam").test(x=x)


st.title("Normality Test")
st.write("Performs the Nonparametric Multivariate Normality Test.")

with st.expander("Click to view code"):
    code_python = """
    from QuadratiK.kernel_test import KernelTest
    X = Read your data file here
    normality_test = KernelTest(h = 0.5, centering_type="nonparam").test(X)
    normality_test.summary()
    """
    st.code(code_python, language="python")

    code_R = """
    library(QuadratiK)
    norm_test <- kb.test(x=dat_norm, h=h, centering="Nonparam")
    summary(norm_test)
    """
    st.code(code_R, language="r")

delim = st.text_input("**Enter the delimiter**", ",")
header_exist = st.checkbox(
    "**Select, if the header is present in the data file.**", value=True
)

if header_exist == False:
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
    except:
        st.error(
            "Unable to read the data file. Please make sure that the delimiter is correct."
        )

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

        except:
            st.error("Please check user inputs and data file")


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
