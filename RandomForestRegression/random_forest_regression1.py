import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import csv

#Getting data from csv file
filename = "dataset.csv"
dataset = pd.read_csv(filename)

x = dataset.iloc[:,[1]].values
y = dataset.iloc[:,[2]].values
plt.scatter(x,y,color='r',s=30,label='points')

#regression
from sklearn.ensemble import RandomForestRegressor

reg = RandomForestRegressor()
reg.fit(x,y)
y_predict = reg.predict(x)
print(mean_squared_error(y_predict,y))

plt.plot(x,y_predict,color='b',label='Regression line')
plt.title("Random Forest regression1")
plt.xlabel("Standard/level")
plt.ylabel("Salary Rs./month")
plt.legend()
plt.show()

