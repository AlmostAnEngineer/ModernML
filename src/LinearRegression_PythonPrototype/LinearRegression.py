import numpy as np


class MultipleLinearRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        self.mean = None
        self.std = None

    def normalize(self, X):
        if self.mean is None or self.std is None:
            raise ValueError("Model has not been trained. Cannot normalize data.")
        X_normalized = (X - self.mean) / (self.std + 1e-8)
        return X_normalized

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        X_normalized = self.normalize(X)
        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.num_iterations):
            y_predicted = np.dot(X_normalized, self.weights) + self.bias

            dw = (1 / num_samples) * np.dot(X_normalized.T, (y_predicted - y))
            db = (1 / num_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        X_normalized = self.normalize(X)
        y_predicted = np.dot(X_normalized, self.weights) + self.bias
        return y_predicted
