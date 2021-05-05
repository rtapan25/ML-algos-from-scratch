import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import datasets

X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=101)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 101)

from linear_reg import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)

y_predicted = model.predict(X_test)

#Mean Squared Error
def mse(y_true, y_predicted):
    return np.mean((y_true-y_predicted)**2)

mse_value = mse(y_test, y_predicted)

print(y_predicted)
print(mse_value)

#Ploting
y_line = model.predict(X)
fig = plt.figure(figsize=(7,7))
p1 = plt.scatter(X_train, y_train)
p2 = plt.scatter(X_test, y_test)
plt.plot(X, y_line, color = 'black')
plt.show()