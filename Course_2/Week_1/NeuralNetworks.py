import numpy as np
from typing import Callable, Tuple, List

class NeuralNet:
    
    def my_sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def standardize_mat(self, X: np.matrix, method: str = 'min-max') -> np.matrix:
        if method == 'min-max':
            return (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        elif method == 'mean-std':
            return (X - X.mean(axis=0)) / X.std(axis=0)

    def my_dense_layer(self, X: np.matrix, W: np.matrix, b: np.ndarray, g: str = 'sigmoid') -> np.matrix:
        A_in = self.standardize_mat(X, method='min-max')
        A_out = np.matmul(A_in, W) + b
        if g == 'sigmoid':
            A_out = self.my_sigmoid(A_out)
        elif g == 'relu':
            A_out = np.maximum(0, A_out)
        return A_out

    def my_sequence(self, X: np.matrix, W: np.matrix, b: np.ndarray, num_layers: int = 2, g: str = 'sigmoid') -> np.matrix:
        for i in range(num_layers):
            X = self.my_dense_layer(X, W, b, g='sigmoid')
        return X

model = NeuralNet()
X = np.matrix([[1, 2, 3], [4, 5, 6]])
W = np.matrix([[1, 2, 3], [4, 5, 6], [5, 6, 7]])
b = np.array([[1, 2, 3]])
print(model.my_sequence(X, W, b))