import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


df = pd.read_csv("Pricing.csv")

X = df.iloc[:, 0].values.reshape(-1, 1)  # values converts it into a numpy array
Y = df.iloc[:, 1].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
reg = LinearRegression()  # create object for the class
reg.fit(X, Y)  # perform linear regression
Y_pred = reg.predict(X)  # make predictions

plt.xlabel('Area')
plt.ylabel('Price($AUD)')
plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()

