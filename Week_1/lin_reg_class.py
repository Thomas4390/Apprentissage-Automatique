import numpy as np 

class LinReg:

    # default setup: lr=0.01 and epoch=800

    def __init__(self, lr=0.01, epochs=800):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None

    # Training function: fit
    def fit(self, X, y):

        # Shape of X. 
        # m --> number of training examples 
        # n --> number of features 
        m, n = X.shape

        # Initializing a column vector of zeros of size : (number of features, 1)
        # and bias as 0. 
        self.weights = np.zeros((n, 1))
        self.bias = 0

        # Reshaping y as (m, 1) in case your dataset initialized as 
        # (m, ) which can cause problems
        y = y.reshape(m, 1)

        # empty list to store losses so we can plot them later
        # against epochs

        losses = []

        for epoch in range(self.epochs):

            # Calculating prediction y_hat or h(x)
            y_hat = np.dot(X, self.weights) + self.bias

            # Calculating loss
            loss = np.mean((y_hat - y)**2)

            # Appending loss to losses
            losses.append(loss)

            # Calculating derivatives of parameters (weights and bias)
            dw = (1/m) * np.dot(X.T, (y_hat - y))
            db = (1/m) * np.sum((y_hat - y))

            # Updating the parameters: parameter := parameter - lr*derivative
            # of loss/cost w.r.t parameter

            self.weights -= self.lr * dw
            self.bias = self.lr * db

        # Returning the parameter so we can look at them later
        return self.weights, self.bias, losses

    # Predicting(calculating y_hat with our updated weights) for the 
    # testing/validation     
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


