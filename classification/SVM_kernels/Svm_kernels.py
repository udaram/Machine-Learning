
#importing libraries and packages
import numpy as np
import pandas as pd
from sklearn.svm import SVC 
from sklearn import preprocessing,cross_validation


#Data 
data = pd.read_csv('breast-cancer-wisconsin.data')

#handling Dummy data and measing values 
data.replace('?',-99999,inplace=True)# to handle meassing data
data.drop(['id'], 1 ,inplace=True) #drop not usable data

x = np.array(data.drop(['class'],1))
y = np.array(data['class'])

#splitting data
x_train,x_test,y_train,y_test = cross_validation.train_test_split(x,y,test_size = 0.2) 

#training the models
Classifier = SVC(kernel='linear',random_state=0,decision_function_shape='ovr')#score with ovo and ovr
#Classifier = SVC(kernel='rbf',random_state=0,decision_function_shape='ovo')
#Classifier = SVC(kernel='poly',random_state=0,decision_function_shape='ovr')
#Classifier = SVC(kernel='sigmoid',random_state=0,decision_function_shape='ovr')
Classifier.fit(x_train,y_train)

#confidence
confidence = Classifier.score(x_test,y_test)

print("Confidence score of model::",confidence)
