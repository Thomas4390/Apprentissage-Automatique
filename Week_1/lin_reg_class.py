import numpy as np 

class Linreg:

    # default setup: lr=0.01 and epoch=800

    def __init__(self, lr=0.01, epochs=800):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.biais = None

    # Training function: fit
    def fit(self, X, y):

        # Shape of X. 
        # m --> number of training examples 
        # n --> number of features 
        m, n = X.shape

        # Initializing 
        self.weights = np.zeros((n, 1))
        self.biais = 0
