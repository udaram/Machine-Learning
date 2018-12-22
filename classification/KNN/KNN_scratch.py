import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import warnings
from math import sqrt 
from collections import Counter 
style.use('fivethirtyeight')

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
    print(Counter(vote).most_common(1))
    vote_result = Counter(vote).most_common(1)[0][0]
    return vote_result

#main
dataset ={'k':[[1,2],[2,3],[3,1]] , 'r':[[6,5],[7,7],[8,6]]}
new_features = [5,7]

for i in dataset:
    for ii in dataset[i]:
        plt.scatter(ii[0],ii[1],s=100,color=i)

plt.scatter(new_features[0], new_features[1], s=100) 

result = k_nearest_neighbors(dataset,new_features,3)
print("New feature will be of Group::",result)
plt.scatter(new_features[0], new_features[1], s=30,color = result[0])
plt.title("KNN implementation code Result")
plt.xlabel('Feature1')
plt.ylabel('Feature2')
plt.show()


