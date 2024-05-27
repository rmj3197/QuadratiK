"""
Contains the Two Sample test functionality of the UI
"""

import importlib
import numpy as np
import streamlit as st
import pandas as pd

kt = importlib.import_module("QuadratiK.kernel_test").KernelTest
qq_plot = importlib.import_module("QuadratiK.tools").qq_plot
pd.set_option("display.float_format", "{:.2g}".format)


@st.cache_data(ttl=30, show_spinner=False)
def run_twosample_test(h_val, num_iter, b, X, Y):
    return kt(h=h_val, num_iter=num_iter, b=b, centering_type="nonparam").test(x=X, y=Y)


st.title("Two Sample Test")

st.write("Performs the Nonparametric Two Sample Test")

with st.expander("Click to view code"):
    code_python = """
    from QuadratiK.kernel_test import KernelTest
    X = Read your data file here
    Y = Read your data file here
    two_sample_test = KernelTest(h = 0.5).test(X,Y)
    two_sample_test.summary()
    """
    st.code(code_python, language="python")

    code_R = """
    library(QuadratiK)
    X = Read your data file here
    Y = Read your data file here
    two_test <- kb.test(x=X, y=Y, h=2)
    summary(two_test)
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

x_data = st.file_uploader(
    "Choose data file for X",
    accept_multiple_files=False,
    type=[".txt", ".csv"],
)

if x_data is not None:
    st.write(x_data.name + " Uploaded Successfully")
    try:
        X = pd.read_csv(x_data, sep=delim, header=header)
    except:
        st.error(
            "Unable to read the data file. Please make sure that the delimiter is correct."
        )

y_data = st.file_uploader(
    "Choose data file for Y", accept_multiple_files=False, type=[".txt", ".csv"]
)

if y_data is not None:
    st.write(y_data.name + " Uploaded Successfully")
    try:
        Y = pd.read_csv(y_data, sep=delim, header=header)
    except:
        st.error(
            "Unable to read the data file. Please make sure that the delimiter is correct."
        )


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
        except:
            st.error("Please check user inputs and data file")


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
