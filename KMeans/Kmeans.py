#Implementing KMeans algorithm from scratch

import numpy as np 
from sklearn.metrics import pairwise_distances

class KMeans:
    def __init__(self, k = 3):
        self.k = k

    def fit(self, data):
        self.centeroids = self.init_centeroids(data)
        self.no_of_iter = 1000
        for _ in range(self.no_of_iter):
            #Assigning Clusters using the Euclidian Distance
            self.cluster_labels = self.assign_clusters(data)
            #Updating the Centeroids
            self.centeroids = self.update_centeroids(data)
        return self

    def predict(self, data):
        return self.assign_clusters(data)

    def init_centeroids(self, data):
        #Random initialization of centeroids 
        initial_cent = np.random.permutation(data.shape[0])[:self.k]
        self.centeroids = data[initial_cent]
        return self.centeroids

    def assign_clusters(self, data):
        if data.ndim == 1:
            data = data.reshape(-1,1)
        dis_to_cent = pairwise_distances(data, self.centeroids, metric ='euclidean')
        self.cluster_labels = np.argmin(dis_to_cent, axis=1)
        return self.cluster_labels

    def update_centeroids(self, data):
        self.centeroids = np.array([data[self.cluster_labels == i].mean(axis = 0 ) for i in range(self.k)])
        return self.centeroids

