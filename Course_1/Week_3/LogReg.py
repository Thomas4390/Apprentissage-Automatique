import numpy as np

class LogReg:
    def __init__(self, X, y, lr=0.01, epochs=100):
        self.X = X
        self.y = y
        self.lr = lr
        self.epochs = epochs
        self.w = np.random.rand(X.shape[1]+1)

    def compute_cost(self):
        m = self.X.shape[0]

        ones = np.ones((m, 1))
        self.X = np.hstack((ones, self.X))

        z = np.array([np.dot(self.X[i], self.w) for i in range(m)])
        g = lambda z: 1 / (1 + np.exp(-z))
        g_z = g(z)

        cost = (-1 / m) * (np.dot(self.y.T, np.log(g_z)) + np.dot((1 - self.y).T, np.log(1 - g_z)))

        return cost
            

