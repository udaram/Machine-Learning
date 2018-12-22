import numpy as np
from sklearn import preprocessing, cross_validation,svm
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('breast-cancer-wisconsin.data')
data.replace('?',-99999,inplace=True)# to handle meassing data
data.drop(['id'], 1 ,inplace=True) #drop not usable data

x = np.array(data.drop(['class'],1))
y = np.array(data['class'])

x_train,x_test,y_train,y_test = cross_validation.train_test_split(x,y,test_size = 0.2) #20% suffle of data for cross validation 

clf = svm.SVC()#support vector machine classifier
clf.fit(x_train,y_train)

accuracy = clf.score(x_test,y_test)
print(accuracy)

example_measures = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,1,1,2,3,2,1]])
example_measures = example_measures.reshape(len(example_measures),-1)

prediction = clf.predict(example_measures)
print(prediction)

