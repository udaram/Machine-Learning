import numpy as np
import pandas as pd
import warnings
from math import sqrt 
from collections import Counter 
import random

#KNN algorithm
def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')
    distance = []
    for group in data :
        for features in data[group]:
            euclidean_dist = np.linalg.norm(np.array(features) - np.array(predict)) #best & faster method to calculate euclidian distance than other methods 
            distance.append([euclidean_dist,group])
    distance = sorted(distance)
    vote = []
    for i in distance[:3]:
       vote.append(i[1])
    #votes = [i[1] for i in sorted(distance)[:k]]  can also be used
    #print(Counter(vote).most_common(1))
    vote_result = Counter(vote).most_common(1)[0][0]
    return vote_result

def main():
    data = pd.read_csv('breast-cancer-wisconsin.data')
    data.replace('?',-99999,inplace=True)# to handle meassing data 
    data.drop(['id'], 1 ,inplace=True) #drop not usable data
    data = data.astype(float).values.tolist() #type conversion of data
    random.shuffle(data)
    test_size = 0.2 #20 % data for test the moddle
    train_set = {2:[],4:[]} #cretaing a dictionary to train model
    test_set = {2:[],4:[]} 
    train_data = data[:-int(test_size*len(data))]#first 80% data 
    test_data = data[-int(test_size*len(data)):] #last 20% data 

    for i in train_data:
        train_set[i[-1]].append(i[:-1])

    for i in test_data:
        test_set[i[-1]].append(i[:-1])


    #To check accuracy of KNN implemented model
    correct = 0
    total = 0
    for group in test_set:
        for features in test_set[group]:
            result = k_nearest_neighbors(train_set,features,k=5)
            if result == group:
               correct = correct+1
            total = total+1
    print('Accuracy::',correct/total)
main()
