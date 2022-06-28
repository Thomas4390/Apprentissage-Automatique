from scipy import rand
import streamlit as st
from lin_reg_class import LinReg
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh, cholesky
from scipy.stats import norm
from itertools import combinations


st.title("""Linear Regression Application""")

slider_weights = st.sidebar.slider("Choose a weight : Beta 2", -5.0, 5.0, value=1.0, step=0.1)
slider_bias = st.sidebar.slider("Choose a bias : Beta 1", -5.0, 5.0, value=1.0, step=0.1) 

st.subheader("What is a Linear Regression?")
st.markdown("*A simple linear regression model is defined by the following equation:*")
st.latex(r"""\forall i \in \{1, 2, ..., n\}, y_i = \beta_1 + \beta_2 x_i + \epsilon_i""")

st.write(r"""The objective of a linear regression is to explain $y_i$ as a function of $x_i$ 
                ,or to study how $y_i$ varies as a function of $x_i$. 
                In our linear regression equation, $y_i$ is the dependent or explained variable, 
                and $x_i$ is the independent or explanatory variable. 
                Also, $\beta_1$ represents the constant or the intercept, and $\beta_2$
                is the coefficient of the slope in the relationship between $y_i$ and $x_i$.
                Finally, The error term $\epsilon_i$, come from the fact that the points 
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

def compute_cost(xs, ys, weights, bias):
    return np.mean((ys - (weights * xs + bias)) ** 2)

my_regression_cost = compute_cost(xs_random, ys_random, slider_weights, slider_bias)
st.write("Our regression model loss value:", round(my_regression_cost, 5))

test_percentage = st.sidebar.slider("Choose training size percentage:", 0.01, 0.99, value=0.5, step=0.05)

train_size = int(size*(1-test_percentage))
X_train, X_test, y_train, y_test = xs_random[:train_size], \
                                   xs_random[train_size:], \
                                   ys_random[:train_size], \
                                   ys_random[train_size:]


model = LinReg()
weights, bias, losses = model.fit(X_train, y_train)
ys_hat = model.predict(xs_random)

st.write("Gradient descent final loss value:", round(losses[-1], 5))


fig, ax = plt.subplots()
ax.plot(xs, ys)
ax.plot(xs_random, ys_hat)
ax.scatter(xs_random, ys_random)
ax.set_ylim(bottom=np.min(xs_random)-1, top=np.max(xs_random)+1)
ax.set_xlim(left=np.min(ys_random)-1, right=np.max(xs_random)+1)

st.pyplot(fig)





