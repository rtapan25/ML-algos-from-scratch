import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import datasets

breast_cancer_dataset = datasets.load_breast_cancer()
X, y = breast_cancer_dataset.data, breast_cancer_dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)

from logistic_reg import LogisticRegression

model = LogisticRegression()

model.fit(X_train, y_train)

y_predicted = model.predict(X_test)

def accuracy(y_true, y_predicted):
    return np.sum((y_true == y_predicted)/ len(y_true))

#print(y_predicted)
print(accuracy(y_test, y_predicted))