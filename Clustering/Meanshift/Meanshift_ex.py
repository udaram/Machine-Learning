#Mean-shift Hierarchical clustering 

import numpy as np 
from sklearn.cluster import MeanShift
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import style
style.use("ggplot")

centers = [[1,1,1],[5,5,5],[3,10,10]]
#data from dataset packages 
X ,_ =make_blobs(n_samples = 400,centers=centers,cluster_std = 1.5)

#fitting to meanshift
ms = MeanShift()
ms.fit(X)

labels = ms.labels_
cluster_centers = ms.cluster_centers_

#printing neccessary details
print(cluster_centers)
n_clusters_ = len(np.unique(labels))
print("Number of estimated clusters:", n_clusters_)

#data visualisation
colors = ['r','g','b','c','k','y','m']
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')

for i in range(len(X)):
    ax.scatter(X[i][0],X[i][1],X[i][2],c=colors[labels[i]],marker='o')

ax.scatter(cluster_centers[:,0],cluster_centers[:,1],cluster_centers[:,2],marker='x',color='k',s=150,linewidth=5,zorder=10)
plt.title("Hirarchical Clustering Using MeanShift")
plt.show()

