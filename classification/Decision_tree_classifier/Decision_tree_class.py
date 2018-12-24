#importing packages

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn import preprocessing,cross_validation

#data 
data = pd.read_csv('social_Network_ads.csv')

x = data.iloc[:,[2,3]].values
y = data.iloc[:,[4]].values

x_train,x_test,y_train,y_test = cross_validation.train_test_split(x,y,test_size=0.3,random_state=1)

#scaling the features
scaler = preprocessing.StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

classifier = DecisionTreeClassifier(criterion='entropy',random_state=0)

classifier.fit(x_train,y_train)

#predict test set results
predict = classifier.predict(x_test)

#accuracy 
accuracy = classifier.score(x_test,y_test)
print(accuracy)

#visualising data with prediction 
colors = []
for i in y_train:
    if i==0:colors.append('r')
    else :colors.append('k')
import matplotlib.pyplot as plt
plt.scatter(x_train[:,[0]],x_train[:,[1]],color=colors,s=25)
colors = []
for i in predict:
    if i==0:colors.append('r')
    else :colors.append('k')
plt.scatter(x_test[:,[0]],x_test[:,[1]],color='g',s=50)
plt.scatter(x_test[:,[0]],x_test[:,[1]],color=colors,s=25)
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.show()



