"""
Contains the Tuning Parameter h selection
functionality of the UI
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import importlib
import matplotlib.pyplot as plt

select_h = importlib.import_module("QuadratiK.kernel_test").select_h

st.title("Tuning Parameter h selection")
st.write(
    "Computes the kernel bandwidth of the Gaussian kernel for the Two-sample\
    and K-sample kernel-based quadratic distance (KBQD) tests."
)

with st.expander("Click to view code"):
    code_python = """
    from QuadratiK.kernel_test import select_h
    h_selected, all_values = select_h(x = x, y = y,alternative = 'skewness')
    """
    st.code(code_python, language="python")

    code_R = """
    library(QuadratiK)
    h_k <- select_h(dat_x=dat_k, dat_y=y, alternative="skewness")
    h_k$h_sel
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
                x=X, y=y, alternative=alternative, power_plot=True
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

            st.pyplot(power_plot, use_container_width=True)
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
