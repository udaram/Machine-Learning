import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
import csv

#Getting data from csv file
filename = "dataset.csv"
dataset = pd.read_csv(filename)

x = dataset.iloc[:,[1]].values
y = dataset.iloc[:,[2]].values
plt.scatter(x,y,color='r',s=10)

#linear regression
from sklearn.linear_model import LinearRegression

linear_reg = LinearRegression()
linear_reg.fit(x,y)
y_predict = linear_reg.predict(x)
plt.plot(x,y_predict,color='b')

#printing values
#print('Slope:' ,linear_reg.coef_)
#print('Intercept:', linear_reg.intercept_)

#polynomial Regression of degree 3
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=3)
x_poly = poly_reg.fit_transform(x)
poly_reg.fit(x_poly,y)
linear_reg2 = LinearRegression()
linear_reg2.fit(x_poly,y)
plt.plot(x,linear_reg2.predict(x_poly),color='g')
plt.title("Comparision b/w linear & polynomial regression")
plt.xlabel("Standard/level")
plt.ylabel("Salary Rs./month")
plt.show()

