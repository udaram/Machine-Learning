#importing libaries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn import preprocessing
import random
from sklearn.metrics import accuracy_score,confusion_matrix


train_data = pd.read_csv("train.csv") 


#handle missing data
train_data.fillna(-9999,inplace=True)
test_data = train_data[-50:]
train_data=train_data[:-50]
#splitting the data
x_train = train_data.iloc[:,[2,5,6,7,9]].values
y_train = train_data.iloc[:,[1]].values
x_test =test_data.iloc[:,[2,5,6,7,9]].values
y_test= test_data.iloc[:,[1]].values
 
#print(x_test)
#scaling data
sc = preprocessing.StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#fitting to model
clf  = RandomForestClassifier(n_jobs=10,random_state=0)
clf.fit(x_train,y_train)

y_pred=clf.predict(x_test)

print(accuracy_score(y_test,y_pred))

#confusion matrix
conf_matrix = confusion_matrix(y_test,y_pred)
print(conf_matrix)
