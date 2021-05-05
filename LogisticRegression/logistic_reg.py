#Impementing Logistic Regression algorithms from scratch
#Logistic Regression with Gradient Descent

import numpy as np 

class LogisticRegression:

    def __init__(self, lr=0.01, no_of_iter=1000):
        self.lr = lr
        self.no_of_iter = no_of_iter
        self.weights = None
        self.bias = None

    
    def fit(self, X, y):
        #initialize the parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0


        #Implementing Gradient Descent 
        for _ in range(self.no_of_iter):
            #Linear model is y = wx +b
            linear_model = np.dot(X, self.weights) + self.bias
            #Applying Sigmoid function
            y_predicted = self.sigmoid(linear_model)
            #Calculating the derivatives
            dw = (1 / n_samples) * (np.dot(X.T, (y_predicted - y)))
            db = (1 / n_samples) * (np.sum(y_predicted - y))
            #Updation rule
            self.weights -= self.lr * dw
            self.bias -= self.lr * db



    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        y_predicted_class = [1 if i > 0.5 else 0 for i in y_predicted]
        return y_predicted_class



    def sigmoid(self, x):
        return 1/ (1 + np.exp(-x))