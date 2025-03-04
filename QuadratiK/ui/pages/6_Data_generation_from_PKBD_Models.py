"""
Contains the data generation from PKBD functionality of the UI
"""

import importlib
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

pkbd = importlib.import_module("QuadratiK.spherical_clustering").PKBD

st.title("Data generation from PKBD Models")
st.write('Generates samples from the supported PKBD Models - "rejvmf" and "rejacg"')

with st.expander("Click to view example code in Python and R"):
    code_python = """
    # Import the PKBD class from the QuadratiK.spherical_clustering module
    from QuadratiK.spherical_clustering import PKBD
    
    # Create an instance of the PKBD class
    pkbd = PKBD()
    
    # Generate samples using the "rejvmf" method
    x_rejvmf = pkbd.rpkb(n = 1000, mu = [0, 1, 1], rho = 0.8, method = "rejvmf", random_state=42)
    
    # Generate samples using the "rejacg" method
    x_rejacg = pkbd.rpkb(n = 1000, mu = [0, 1, 1], rho = 0.8, method = "rejacg", random_state=42)
    """
    st.code(code_python, language="python")

    code_R = """
    # Define the location parameter vector
    mu <- c(0, 0, 1)
    # Define the number of dimensions
    d <- 3
    # Define the number of samples to generate
    n <- 1000
    # Define the concentration parameter
    rho <- 0.8

    # Set the seed for reproducibility
    set.seed(2468)
    # Generate observations using the rejection algorithm with von-Mises 
    # distribution envelopes
    dat1 <- rpkb(n = n, rho = rho, mu = mu, method = "rejvmf")
    # Generate observations using the rejection algorithm with angular central 
    # Gaussian distribution envelopes
    dat2 <- rpkb(n = n, rho = rho, mu = mu, method = "rejacg")
    """
    st.code(code_R, language="r")

st.subheader("Input Instructions", divider="grey")

st.write("1. Enter the total number of samples to be generated.")
st.write(r"2. Enter the value of the concentration parameter ($\rho$).")
st.write(r"3. Enter the location parameter ($\mu$) separated by space.")


n_samples = int(
    st.number_input("Enter the total number of samples to be generated", value=100)
)

rho = float(
    st.number_input(r"Enter the value of concentration parameter ($\rho$)", value=0.85)
)


def collect_numbers(x):
    return [float(i) for i in re.split("[^.0-9]", x) if i != ""]


mu_list = collect_numbers(
    st.text_input(
        r"Please enter the location parameter ($\mu$) separated by space",
        value="0 0 0.5",
    )
)


@st.cache_data
def generate_samples(n_samples, mu_list, rho, method):
    return pkbd().rpkb(n_samples, mu_list, rho, method)


data1 = generate_samples(n_samples, mu_list, rho, "rejvmf")
with st.expander('**Data generated using "rejvmf"**'):
    st.write(data1)
csv_res1 = pd.DataFrame(data1).to_csv().encode()
download_rejvmf = st.download_button(
    "Click to Download the generated samples using rejvmf",
    csv_res1,
    "rejvmf_generated_samples.csv",
    "text/csv",
    key="download-rejvmf",
)

data2 = generate_samples(n_samples, mu_list, rho, "rejacg")
with st.expander('**Data generated using "rejacg"**'):
    st.write(data2)
csv_res2 = pd.DataFrame(data2).to_csv().encode()
download_rejacg = st.download_button(
    "Click to Download the generated samples using rejacg",
    csv_res2,
    "rejvmf_generated_samples.csv",
    "text/csv",
    key="download-rejacg",
)

if st.button("Rerun"):
    st.cache_data.clear()

if len(mu_list) == 2:
    fig1 = plt.figure(figsize=(6, 6))
    plt.scatter(data1[:, 0], data1[:, 1], label="rejvmf")
    plt.scatter(data2[:, 0], data2[:, 1], label="rejacg")

    theta = np.linspace(0, 2 * np.pi, 100)
    unit_circle_x = np.cos(theta)
    unit_circle_y = np.sin(theta)

    plt.plot(unit_circle_x, unit_circle_y, linestyle="dashed", color="red")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    st.pyplot(fig1)

elif len(mu_list) == 3:
    r = 1
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0.0:pi:100j, 0.0 : 2.0 * pi : 100j]
    x1 = r * sin(phi) * cos(theta)
    y1 = r * sin(phi) * sin(theta)
    z1 = r * cos(phi)
    xx1, yy1, zz1 = data1[:, 0], data1[:, 1], data1[:, 2]
    xx2, yy2, zz2 = data2[:, 0], data2[:, 1], data2[:, 2]

    fig = make_subplots(rows=1, cols=1, specs=[[{"type": "scatter3d"}]])
    fig.append_trace(
        go.Surface(
            x=x1,
            y=y1,
            z=z1,
            colorscale=[[0, "#DCDCDC"], [1, "#DCDCDC"]],
            opacity=0.5,
            showscale=False,
        ),
        row=1,
        col=1,
    )
    fig.append_trace(
        go.Scatter3d(
            x=xx1,
            y=yy1,
            z=zz1,
            mode="markers",
            marker=dict(size=5, color="red", showscale=False),
            name="rejvmf",
        ),
        row=1,
        col=1,
    )
    fig.append_trace(
        go.Scatter3d(
            x=xx2,
            y=yy2,
            z=zz2,
            mode="markers",
            marker=dict(size=5, color="blue", showscale=False),
            name="rejacg",
        ),
        row=1,
        col=1,
    )

    fig.update_layout(
        title="",
        scene=dict(
            xaxis=dict(range=[-1, 1]),
            yaxis=dict(range=[-1, 1]),
            zaxis=dict(range=[-1, 1]),
            aspectmode="data",
        ),
    )
    fig.update_layout(showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

else:
    st.error(
        "Visualization cannot be generated as number of dimensions are greater than 3"
    )

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
