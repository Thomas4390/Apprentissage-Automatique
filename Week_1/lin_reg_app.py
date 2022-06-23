import streamlit as st
from lin_reg_class import LinReg
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh, cholesky
from scipy.stats import norm


st.title("""Linear Regression Application""")

slider_weights = st.sidebar.slider("Choose a weight : W", -5, 5, value=0)
slider_bias = st.sidebar.slider("Choose a bias : B", -5, 5, value=0)

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

mu = st.sidebar.number_input("Mean of random data (mu):", step=1, value=0)
sigma = st.sidebar.number_input("Standard deviation of random data (sigma):", step=0.1, value=1.0)
size = st.sidebar.number_input("Size of random data:", step=1, value=100)

def generating_random_data(mu=mu, sigma=sigma, size=size):
    epsilons = np.random.normal(mu, sigma, size)
    return epsilons

xs = np.linspace(0, 100, size)
ys = slider_weights * xs + slider_bias

epsilons = generating_random_data(mu, sigma, size)
ys_random = slider_weights * xs + slider_bias + epsilons

fig, ax = plt.subplots()
ax.plot(xs, ys)
ax.scatter(xs, ys_random)
ax.set_ylim(bottom=-10, top=10)
ax.set_xlim(left=0, right=100)

st.pyplot(fig)





