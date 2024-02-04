"""
Contains the Poisson Kerel Test test functionality of the UI
"""

import importlib
import streamlit as st
import pandas as pd


pkt = importlib.import_module(
    'QuadratiK.poisson_kernel_test').PoissonKernelTest
qq_plot = importlib.import_module('QuadratiK.tools').qq_plot

st.title('Poisson Kernel-based quadratic distance test of Uniformity on the Sphere')

st.write(r"Performs the kernel-based quadratic distance Goodness-of-fit tests for Uniformity for \
    spherical data using the Poisson kernel with concentration parameter rho ($\rho$)")

with st.expander("Click to view code"):
    code_python = '''
    from QuadratiK.poisson_kernel_test import PoissonKernelTest
    X = Read your data file here
    unif_test = PoissonKernelTest(rho = 0.7).test(X)
    unif_test.summary()
    '''
    st.code(code_python, language="python")

    code_R = '''
    library(QuadratiK)
    X = Read your data file here
    res_unif <- pk.test(x=X, rho=rho)
    summary(res_unif)
    '''
    st.code(code_R, language="r")

delim = st.text_input('**Enter the delimiter**', " ")
header_exist = st.checkbox(
    "**Select, if the header is present in the data file.**", value=True)

if header_exist == False:
    header = None
else:
    header = "infer"
    
data = st.file_uploader("Please Upload the data file",
                        accept_multiple_files=False, type=['.txt', '.csv'])

if data is not None:
    st.success(data.name + " Uploaded Successfully!")
    try:
        X = pd.read_csv(data, sep=delim, header=header)
    except:
        st.error(
            "Unable to read the data file. Please make sure that the delimiter is correct.")

    num_iter = int(st.number_input(
        'Enter a value of number of iterations to be used for critical value estimation', value=300))
    rho = float(st.number_input(r'Enter a value of rho ($\rho$)', value=0.7))

    with st.spinner('getting results ready...'):
        try:
            unif_test = pkt(rho=rho, num_iter=num_iter).test(X)
            res = pd.DataFrame()
            res['Value'] = [unif_test.test_type_, unif_test.u_statistic_un_, unif_test.u_statistic_cv_,
                            unif_test.u_statistic_h0_, unif_test.v_statistic_vn_, unif_test.v_statistic_cv_,
                            unif_test.v_statistic_h0_]
            res = res.set_axis(["Test Type", "Un Statistic",
                                "Un Critical Value", "Un H0 is rejected",
                                "Vn Statistic", "Vn Critical Value", "Vn H0 is rejected"])
            st.table(res)
            csv_res = res.to_csv().encode()
            st.download_button("Click to Download the test results", csv_res,
                               "Normality_Test_results.csv", "text/csv", key='download-txt')

            st.subheader('Summary Statistics', divider="grey")
            summary_stats_df = unif_test.stats()
            st.dataframe(summary_stats_df)
            csv_stats = summary_stats_df.to_csv(index=True).encode()
            st.download_button("Click to Download the summary statistics",
                               csv_stats, "Statistics.csv", "text/csv", key='download-csv')
            st.success('Done!')
        except:
            st.error("Please check user inputs and data file")


st.header('QQ Plots', divider="grey")

with st.expander("Click to view code"):
    viz_code = '''
    from QuadratiK.tools import qq_plot
    qq_plot(X, dist = "uniform")
    '''
    st.code(viz_code, language="python")

if data is not None:
    try:
        st.pyplot(qq_plot(X, dist="uniform"), use_container_width=True)
    except:
        st.error("Please ensure that the data file is loaded")


st.markdown(r"""
    <style>
        .reportview-container {
            margin-top: -2em;
        }
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
    </style>
""", unsafe_allow_html=True)
