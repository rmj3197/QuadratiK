"""
Contains the Poisson Kerel Test test functionality of the UI
"""

import importlib

import pandas as pd
import streamlit as st

pkt = importlib.import_module("QuadratiK.poisson_kernel_test").PoissonKernelTest
qq_plot = importlib.import_module("QuadratiK.tools").qq_plot


@st.cache_data(ttl=30, show_spinner=False)
def run_uniformity_test(rho, num_iter, x):
    return pkt(rho, num_iter).test(x)


st.title("Poisson Kernel-based quadratic distance test of Uniformity on the Sphere")

st.write(
    r"Performs the kernel-based quadratic distance Goodness-of-fit tests for Uniformity for \
    spherical data using the Poisson kernel with concentration parameter rho ($\rho$)"
)

with st.expander("Click to view example code in Python and R"):
    code_python = """
    import numpy as np

    np.random.seed(0)
    from QuadratiK.poisson_kernel_test import PoissonKernelTest

    # data generation
    z = np.random.normal(size=(200, 3))
    data_unif = z / np.sqrt(np.sum(z**2, axis=1, keepdims=True))

    # performing the uniformity test
    unif_test = PoissonKernelTest(rho=0.7, random_state=42).test(data_unif)

    # printing the summary for uniformity test
    print(unif_test.summary())
    """
    st.code(code_python, language="python")

    code_R = """
    # Load the QuadratiK library
    library(QuadratiK)
    
    # Set parameters for data generation
    n <- 200
    d <- 3
    
    # Generate random data on the sphere
    set.seed(2468)
    z <- matrix(rnorm(n * d), n, d)
    dat_sphere <- z/sqrt(rowSums(z^2))
    
    # Set the concentration parameter rho
    rho <- 0.7
    
    # Perform the uniformity test using the Poisson Kernel Test
    set.seed(2468)
    res_unif <- pk.test(x = dat_sphere, rho = rho)
    
    # Display the results of the uniformity test
    show(res_unif)
    """
    st.code(code_R, language="r")

st.subheader("Input Instructions", divider="grey")

st.write("1. Upload the data file in .txt or .csv format.")
st.write(
    "2. The file may contain a header (see image below for reference). If headers are present, check the box. The checkbox is selected by default."
)
st.write("3. Specify the separator or delimiter used; the default is a comma (,).")
st.write(
    r"4. Once the data is uploaded, specify the number of iterations for critical value estimation and concentration parameter ($\rho$). Default values are provided."
)

st.image(
    str(
        importlib.resources.files("QuadratiK.ui").joinpath(
            "pages/assets/uniformity_test_format.png"
        )
    ),
    caption="Sample data format for uniformity test",
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
        X = pd.read_csv(data, sep=delim, header=header)
    except Exception as e:
        st.error(f"An error occurred: {e}")

    num_iter = int(
        st.number_input(
            "Enter a value of number of iterations to be used for critical value estimation",
            value=300,
        )
    )
    rho = float(st.number_input(r"Enter a value of rho ($\rho$)", value=0.7))

    with st.spinner("getting results ready..."):
        try:
            unif_test = run_uniformity_test(rho, num_iter, X)
            index_labels = [
                "Test Statistic",
                "Critical Value",
                "H0 is rejected (1 = True, 0 = False)",
            ]
            test_summary = {
                "U-Statistic": [
                    unif_test.u_statistic_un_,
                    unif_test.u_statistic_cv_,
                    unif_test.u_statistic_h0_,
                ],
                "V-Statistic": [
                    unif_test.v_statistic_vn_,
                    unif_test.v_statistic_cv_,
                    unif_test.v_statistic_h0_,
                ],
            }
            res = pd.DataFrame(test_summary, index=index_labels)

            st.text(unif_test.test_type_)
            st.table(res)
            csv_res = res.to_csv().encode()
            st.download_button(
                "Click to Download the test results",
                csv_res,
                "Uniformity_Test_results.csv",
                "text/csv",
                key="download-txt",
            )

            st.subheader("Summary Statistics", divider="grey")
            summary_stats_df = unif_test.stats()
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
    qq_plot(X, dist = "uniform")
    """
    st.code(viz_code, language="python")

if data is not None:
    try:
        st.pyplot(qq_plot(X, dist="uniform"), use_container_width=True)
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
