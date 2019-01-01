import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import csv
import operator
from sklearn.utils import shuffle
import pandas as pd
#Getting data from csv file
filename = "train.csv"
dataset = pd.read_csv(filename)

#to shuffle data
dataset=shuffle(dataset)

#shortnenig data for clearifiction
dataset = dataset[:100]

x = dataset.iloc[:,[0]].values
y = dataset.iloc[:,[1]].values
plt.scatter(x,y,color='r',s=30,label='points')

#regression
from sklearn.ensemble import RandomForestRegressor

reg = RandomForestRegressor()
reg.fit(x,y)
y_predict = reg.predict(x)

sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(x,y_predict), key=sort_axis)
x,y_predict= zip(*sorted_zip)

plt.plot(x,y_predict,color='b',label='Regression line')

plt.title("Random Forest regression2")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()
