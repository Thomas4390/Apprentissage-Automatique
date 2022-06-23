import streamlit as st
from lin_reg_class import LinReg
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

st.title("""Linear Regression Application""")

add_slider_weights = st.sidebar.slider("Choose a weight", 0, 100)

st.subheader("What is a Linear Regression?")
st.markdown("*A simple linear regression model is defined by the following equation:*")
st.latex(r"""\forall i \in \{1, 2, ..., n\}, y_i = \beta_1 + \beta_2 x_i + \epsilon_i""")
st.markdown("""The quantities $\epsilon_i$ come from the fact that the points 
                are never perfectly aligned on a line.
                They are called errors (or noise) and are assumed to be random. 
                To be able to say relevant things
                about this model, one must nevertheless impose some assumptions about them. 
                Here are those that we will make in a first step: """)

st.latex(r"""\\
    (\mathcal{H_1}) : \ \mathbb{E}[\epsilon_i] = 0, \forall i \\ 
(\mathcal{H_2}) : \ Cov(\epsilon_i, \epsilon_j) = \delta_{ij} \sigma^2, \forall (i, j)
""")



