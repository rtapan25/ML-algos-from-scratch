import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()

X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

from KNN import KNN

model = KNN(k = 3)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = np.sum(predictions == y_test)/len(y_test)

print(predictions)
print(accuracy)