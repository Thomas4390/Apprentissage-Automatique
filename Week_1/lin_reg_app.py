import streamlit as st
from lin_reg_class import LinReg
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh, cholesky
from scipy.stats import norm



st.title("""Linear Regression Application""")

slider_weights = st.sidebar.slider("Choose a weight : W", -5, 5, value=1)
slider_bias = st.sidebar.slider("Choose a bias : B", -5, 5, value=1)

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
size = int(st.sidebar.number_input("Size of random data:", step=1, value=100))

xs = np.linspace(-10, 10, 2)
ys = slider_weights * xs + slider_bias

# Choice of cholesky or eigenvector method.
method = 'cholesky'
#method = 'eigenvectors'



# The desired covariance matrix.
r = np.array([
    [3.40, -2.75, -2.00],
    [-2.75,  5.50,  1.50],
    [-2.00,  1.50,  1.25]
])

random_state = int(st.sidebar.number_input("Choose random state: ", step=1, value=1))

# Generate samples from three independent normally distributed random
# variables (with mean 0 and std. dev. 1).
x = norm.rvs(size=(3, size), random_state=random_state)

# We need a matrix `c` for which `c*c^T = r`.  We can use, for example,
# the Cholesky decomposition, or the we can construct `c` from the
# eigenvectors and eigenvalues.

if method == 'cholesky':
    # Compute the Cholesky decomposition.
    c = cholesky(r, lower=True)
else:
    # Compute the eigenvalues and eigenvectors.
    evals, evecs = eigh(r)
    # Construct c, so c*c^T = r.
    c = np.dot(evecs, np.diag(np.sqrt(evals)))

# Convert the data to correlated random variables.
y = np.dot(c, x)
# Plot various projections of the samples.
xs_random = y[0]
ys_random = y[1]

X_train, X_test, y_train, y_test = xs_random[:70], xs_random[70:], ys_random[:70], ys_random[70:]



model = LinReg()
weights, bias, losses = model.fit(X_train, y_train)

fig, ax = plt.subplots()
ax.plot(xs, ys)
ax.scatter(xs_random, ys_random)
ax.set_ylim(bottom=np.min(xs_random)-1, top=np.max(xs_random)+1)
ax.set_xlim(left=np.min(ys_random)-1, right=np.max(xs_random)+1)

st.pyplot(fig)





