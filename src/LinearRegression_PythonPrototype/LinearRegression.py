import numpy as np


class MultipleLinearRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000, optimizer="gradient_descent", beta1=0.9, beta2=0.999,
                 epsilon=1e-8, t=0):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.optimizer = optimizer
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = t
        self.weights = None
        self.bias = None
        self.mean = None
        self.std = None
        self.m = None
        self.v = None

    def __normalize(self, x):
        if self.mean is None or self.std is None:
            raise ValueError("Model has not been trained. Cannot normalize data.")
        x_normalized = (x - self.mean) / (self.std + 1e-8)
        return x_normalized

    def get_weights(self):
        return self.weights, self.bias

    def fit(self, x, y):
        num_samples, num_features = x.shape
        self.mean = np.mean(x, axis=0)
        self.std = np.std(x, axis=0)
        x_normalized = self.__normalize(x)
        self.weights = np.zeros(num_features)
        self.bias = 0
        self.m = np.zeros(num_features)
        self.v = np.zeros(num_features)

        for _ in range(self.num_iterations):
            y_predicted = np.dot(x_normalized, self.weights) + self.bias

            if self.optimizer == 'gradient_descent':
                dw = (1 / num_samples) * np.dot(x_normalized.T, (y_predicted - y))
                db = (1 / num_samples) * np.sum(y_predicted - y)
            elif self.optimizer == 'sgd':
                random_idx = np.random.choice(num_samples)
                x_sample = x_normalized[random_idx, :].reshape(1, -1)
                y_sample = y[random_idx]
                y_predicted_sample = np.dot(x_sample, self.weights) + self.bias
                dw = np.dot(x_sample.T, (y_predicted_sample - y_sample))
                db = y_predicted_sample - y_sample
            elif self.optimizer == 'adam':
                dw = (1 / num_samples) * np.dot(x_normalized.T, (y_predicted - y))
                db = (1 / num_samples) * np.sum(y_predicted - y)
                self.t += 1
                self.m = self.beta1 * self.m + (1 - self.beta1) * dw
                self.v = self.beta2 * self.v + (1 - self.beta2) * (dw ** 2)
                m_hat = self.m / (1 - self.beta1 ** self.t)
                v_hat = self.v / (1 - self.beta2 ** self.t)
                dw = m_hat / (np.sqrt(v_hat) + self.epsilon)
            else:
                raise TypeError("Bad optimizer name")

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, x):
        x_normalized = self.__normalize(x)
        y_predicted = np.dot(x_normalized, self.weights) + self.bias
        return y_predicted

    @staticmethod
    def __r2score(y_real, y_pred):
        y_mean = np.mean(y_real)
        ssr = np.sum((y_real - y_pred)**2)
        sst = np.sum((y_real-y_mean)**2)
        r2 = 1 - (ssr/sst)
        return r2

    @staticmethod
    def mae(y_real, y_pred):
        return np.mean(np.abs(y_real - y_pred))

    @staticmethod
    def mse(y_real, y_pred):
        return np.mean((y_real - y_pred) ** 2)

    def score(self, x, y, metric_name="R2"):
        if metric_name not in ["R2", "MAE", "MSE"]:
            raise TypeError("bad metric name, available: R2, MAE, MSE")
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y shape mismatch")
        y_pred = self.predict(x)
        if metric_name == "R2":
            return self.__r2score(y, y_pred)
        elif metric_name == "MAE":
            return self.__r2score(y, y_pred)
        elif metric_name == "MSE":
            return self.__r2score(y, y_pred)
        else:
            raise TypeError("bad metric name, available: R2, MAE, MSE")
