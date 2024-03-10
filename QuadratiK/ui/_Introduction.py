import streamlit as st

st.set_page_config(
    page_title="QuadratiK User Interface", initial_sidebar_state="expanded"
)

st.header("QuadratiK User Interface")

st.write(
    "***Authors: Giovanni Saraceno, Marianthi Markatou, Raktim Mukhopadhyay, Mojgan Golzy***"
)

st.sidebar.success("Select a function above")
st.markdown(
    r"""<div  style="text-align: justify;">In
this work, we  introduce    novel R  and
Python             packages         that
incorporate  innovative   data  analysis
methodologies.  The   presented packages
offer
a comprehensive set  of  goodness-of-fit
tests  and clustering  techniques  using
kernel-based
quadratic    distances.    Our  packages
implements one, two  and  k-sample tests
for     goodness     of    fit    (GOF),
providing an  efficient,  mathematically
sound   way,  to   assess   the  fit  of
probability distributions. 
Additionally, our  framework
supports     tests      for   uniformity
on  the d-dimensional    sphere,  taking
advantage of  Poisson kernel  densities,
thus    expanding    its   capabilities.
Particularly     noteworthy     is   the
incorporation of sampling algorithms for
generating  data   from    PKBD   models
and  of  a  unique clustering  algorithm
specifically  tailored    for  spherical
data.
This algorithm leverages   a mixture  of
Poisson-kernel-based  densities   on the
sphere, enabling effective clustering of spherical
data or  data  that has been spherically
transformed.
This facilitates  the    uncovering  of
underlying patterns and relationships in
the                                data.
In   summary,  our R and Python packages
serve as a  powerful    suite of  tools,
offering                     researchers
and  practitioners the   means to  delve
deeper  into  their  data,   draw robust
inference,            and        conduct
potentially  impactful     analyses  and
inference   across   a   wide  array  of
disciplines.</div>""",
    unsafe_allow_html=True,
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
