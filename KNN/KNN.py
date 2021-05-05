#Impementing KNN algorithms from scratch
#Steps ---------->
#   1. Calculate the Euclidian Distance [sqrt((x1-x2)^2 +(y1-y2)^2 + ...)] of the input point from every point in the neighbourhood "K".
#   2. Sum all the distances by categories.
#   3. The point belongs to the category that occurs the maximum number of times in the sorted list.

import numpy as np
from collections import Counter

def euclidian_distance(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))


class KNN:
    def __init__(self, k = 4): # k will default to 4
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self.helper_predictions(x) for x in X]
        return np.array(predictions)

    def helper_predictions(self, x):
        #calculating Euclidian distance
        dis = [euclidian_distance(x, x_train) for x_train in self.X_train]

        #K nearest labels
        index_for_labels = np.argsort(dis)[:self.k]
        k_labels = [self.y_train[i] for i in index_for_labels]

        #Category of the test sample is the category having the maximum number of labels in the k_labels list
        most_common = Counter(k_labels).most_common(1)
        return most_common[0][0]


    