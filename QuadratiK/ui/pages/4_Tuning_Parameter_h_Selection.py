"""
Contains the Tuning Parameter h selection
functionality of the UI
"""

import importlib

import pandas as pd
import streamlit as st

select_h = importlib.import_module("QuadratiK.kernel_test").select_h

st.title("Tuning Parameter h selection")
st.write(
    "Computes the kernel bandwidth of the Gaussian kernel for the One-Sample, Two-sample\
    and K-sample kernel-based quadratic distance (KBQD) tests."
)

with st.expander("Click to view example code in Python and R"):
    code_python = """
    import numpy as np
    np.random.seed(0)
    
    from scipy.stats import skewnorm
    
    from QuadratiK.kernel_test import select_h
    
    X_2 = np.random.multivariate_normal(mean=np.zeros(4), cov=np.eye(4), size=200)
    Y_2 = skewnorm.rvs(
    size=(200, 4),
    loc=np.zeros(4),
    scale=np.ones(4),
    a=np.repeat(0.5, 4),
    random_state=20,
    )

    # Perform the algorithm for selecting h
    h_selected, all_powers, plot = select_h(
        x=X_2, y=Y_2, alternative="location", power_plot=True
    )
    print(f"Selected h is: {h_selected}")
    """
    st.code(code_python, language="python")

    code_R = """
    library(QuadratiK)
    # Select the value of h using the mid-power algorithm
    # Create two random normal matrices with 100 elements each
    x <- matrix(rnorm(100), ncol = 2)
    y <- matrix(rnorm(100), ncol = 2)
    # Perform h selection for location alternative
    h_sel <- select_h(x, y, alternative = "location")
    """
    st.code(code_R, language="r")

st.subheader("Input Instructions", divider="grey")
st.write("1. Upload the data file in .txt or .csv format.")
st.write(
    "2. The file may contain a header (see image below for reference). If headers are present, check the box. The checkbox is selected by default."
)
st.write("3. Specify the separator or delimiter used; the default is a comma (,).")

st.write(
    """Once the data is uploaded, specify the column in the data file that contains the labels. Additionally, 
    - For One-Sample test: All rows should have the same label
    - For Two-Sample test: Use two distinct labels to identify the groups
    - For K-Sample test: Use K distinct labels to identify the K groups"""
)
st.write(
    "5. Furthermore please specify the values umber of iterations to be used for critical value estimation, proportion of subsampling samples to be used, and the alternative for computing the value of h. Default values are provided."
)

st.image(
    str(
        importlib.resources.files("QuadratiK.ui").joinpath(
            "pages/assets/hselect_format.png"
        )
    ),
    caption="Sample data format for tuning parameter selection.",
    width="stretch",
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

    data = data.to_numpy()
    label_col = data.shape[1] - 1

if data is not None:
    col_number = int(
        st.number_input(
            "Enter the column in the datafile that contains the label (start from 0)",
            value=label_col,
            step=1,
        )
    )

if data is not None:
    B = st.number_input(
        "Enter a value of number of iterations to be used for critical value estimation",
        value=50,
    )
    b = st.number_input(
        "Enter a value for the proportion of subsampling samples to be used", value=0.8
    )
    alternative = st.selectbox(
        "Choose the alternative to be used", ("location", "scale", "skewness")
    )

    with st.spinner("getting results ready..."):
        try:
            B = int(B)
            b = float(b)
            X = data[:, :col_number]
            y = data[:, col_number]
            h_selected, all_values, power_plot = select_h(
                x=X, y=y, num_iter=B, b=b, alternative=alternative, power_plot=True
            )

            st.markdown(r"The selected $h$ is : " + str(h_selected))

            csv_stats = all_values.to_csv(index=True).encode()

            st.download_button(
                "Click to Download the detailed results",
                csv_stats,
                "h_selection_res.csv",
                "text/csv",
                key="download-h-select",
            )

            st.pyplot(power_plot, width="stretch")
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
