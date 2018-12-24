import pandas as pd
import numpy as np
import operator
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

#data 
data = pd.read_csv('melb_data.csv')

x = data.iloc[:,[0]].values
y = data.iloc[:,[1]].values

classifier = DecisionTreeRegressor(random_state=0)
classifier.fit(x,y)
predict = classifier.predict(x)


#data visualisation 
plt.scatter(x,y,c='r')
sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(x,predict), key=sort_axis)
x, predict = zip(*sorted_zip)
plt.plot(x,predict,c='b')
plt.show()
