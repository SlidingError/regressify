import numpy as np
class LinearRegressionScratch:
    def __init__(self, learning_rate=0.01, iterations=100):
        self.lr = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = 0
        self.costs = []
    def fit(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)
        for _ in range(self.iterations):
            pred = (X @ self.weights.T) + self.bias
            error = pred - y
            dw = (2/m) * np.sum(X.T @ error)
            self.weights -= self.lr * dw
            db = (2/m) * np.sum(error)
            self.bias -= self.lr * db
            cost = (1/m) * np.sum(error ** 2)
            self.costs.append(cost)
    def predict(self, X):
        return (X @ self.weights) + self.bias