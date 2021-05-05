#Impementing Linear Regression algorithms from scratch
#Linear Regression with Gradient Descent

import numpy as np

class LinearRegression:
    def __init__(self, lr = 0.01, no_iterations = 1000):
        self.lr = lr #Learning Rate
        self.no_iterations = no_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        #initialize parameters for gradient descent
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        #Implementing Gradient Descent
        for _ in range(self.no_iterations):
            y_predicted = np.dot(X, self.weights) + self.bias  # y = wx + b
            #Calculating derivative w.r.t weight and bias
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y)) #.T Tranforms the vector
            db = (1/n_samples) * np.sum(y_predicted - y) # *2 just scales the value so not required
            #Updating the weights and the bias using the updation rule
            self.weights -= self.lr * dw 
            self.bias -= self.lr * db

    def predict(self, X):
        y_predicted = np.dot(X, self.weights) + self.bias  # y = wx + b
        return y_predicted
