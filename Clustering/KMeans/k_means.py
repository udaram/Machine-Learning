import matplotlib.pyplot as plt 
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans

#data
x = np.array([[1,2],[1.5,1.8],[5,8],[8,8],[1,0.6],[9,11]])

#plt.scatter(x[:,0],x[:,1],s=100)
#plt.show()

clf = KMeans(n_clusters=2)
clf.fit(x)

centroids = clf.cluster_centers_
labels = clf.labels_

colors = ['g.','r.','c.']

for i in range(len(x)):
    plt.plot(x[i][0],x[i][1],colors[labels[i]],markersize=30)
plt.scatter(centroids[:,0],centroids[:,1],marker='*',s=50,c='k')
plt.title("K-Means")
plt.show()
