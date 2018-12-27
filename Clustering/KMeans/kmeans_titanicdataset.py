import matplotlib.pyplot as plt 
from matplotlib import style
style.use('ggplot')
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import preprocessing,cross_validation

def handle_non_numeric_data(data):
    columns = data.columns.values
    for column in columns:
        text_digit_values={}
        def convert_to_numeric(val):
            return text_digit_values[val]   
        if data[column].dtype != np.int64 and data[column].dtype != np.float64:
           column_list = data[column].values.tolist()
           unique_values = set(column_list)
           x = 0
           for value in unique_values:
               if value not in text_digit_values:
                  text_digit_values[value]=x
                  x = x + 1
           data[column] = list(map(convert_to_numeric,data[column]))
       
    return data

#data  
data = pd.read_excel('titanic.xls')
data.drop(['body','name'],1,inplace = True)
data.fillna(0,inplace=True)

#function to handle non-numeric data columns
data = handle_non_numeric_data(data)

x = np.array(data.drop(['survived'],1).astype(float))
x = preprocessing.scale(x)
y = np.array(data['survived'])

#fitting and trainnig model
model = KMeans(n_clusters=2)
model.fit(x)

correct = 0
for i in range(len(x)):
    predict = np.array(x[i].astype(float))
    predict = predict.reshape(-1,len(predict))
    prediction = model.predict(predict)
    if prediction[0] ==y[i]:
       correct +=1

print(correct/len(x))
    

