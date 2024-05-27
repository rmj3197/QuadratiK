"""
Contains the K-Sample test functionality of the UI
"""

import importlib
import streamlit as st
import pandas as pd

kt = importlib.import_module("QuadratiK.kernel_test").KernelTest
qq_plot = importlib.import_module("QuadratiK.tools").qq_plot
stats = importlib.import_module("QuadratiK.tools").stats


@st.cache_data(ttl=30, show_spinner=False)
def run_ksample_test(h_val, num_iter, b, X, y):
    return kt(h=h_val, num_iter=num_iter, b=b, centering_type="nonparam").test(x=X, y=y)


st.title("K Sample Test")
st.write("Performs the Nonparametric K-Sample Test")

with st.expander("Click to view code"):

    code_python = """
    from QuadratiK.kernel_test import KernelTest
    X,y = Read your data file here
    k_sample_test = KernelTest(h = 0.5).kb_test(X,y)
    k_sample_test.summary()
    """
    st.code(code_python, language="python")

    code_R = """
    library(QuadratiK)
    X,y = Read your data file here
    k_test <- kb.test(x=X, y=y, h=1.5)
    summary(k_test)
    """
    st.code(code_R, language="r")

delim = st.text_input("**Enter the delimiter**", " ")
header_exist = st.checkbox(
    "**Select, if the header is present in the data file.**", value=True
)

if header_exist == False:
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
    except:
        st.error(
            "Unable to read the data file. Please make sure that the delimiter is correct."
        )

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
        except:
            st.error("Please check user inputs and data file")

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
