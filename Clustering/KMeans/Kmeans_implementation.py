import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np

class KMeans:
   def __init__(self,k=2,tolerance=0.001,max_itr=300):
       self.k = k
       self.tolerance = tolerance
       self.max_itr = max_itr
       
   def fit(self,data):
       
       self.centroids = {}
       
       #initialising first k features as centroids
       for i in range(self.k):
           self.centroids[i] = data[i]
       #iterating through max_itr
       for i in range(self.max_itr):
           self.classifications = {}
           
           
           for i in range(self.k):
               self.classifications[i] = []
               
           for featureset in data:
               distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
               classification = distances.index(min(distances))
               self.classifications[classification].append(featureset)
           
           prev_centroids = dict(self.centroids)
           
           for classification in self.classifications:
               self.centroids[classification] = np.average(self.classifications[classification],axis=0)
               
           optimised = True
           
           for c in self.centroids:
               original_centroid = prev_centroids[c]
               current_centroid = self.centroids[c]
               if np.sum((current_centroid - original_centroid)/original_centroid*100.0) > self.tolerance:
                  optimized = False
           if optimized:
              break;
   
   def predict(self,data):
       distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
       classification = distances.index(min(distances))
       return classification
   
              
#data
x = np.array([[1,2],[1.5,1.8],[5,8],[8,8],[1,0.6],[9,11]])

colors = ["g","r","c","b","k"]

model = KMeans()
model.fit(x)

for centroid in model.centroids:
    plt.scatter(model.centroids[centroid][0],model.centroids[centroid][1],marker='o',color='k',s=100,linewidth=5)

for classification in model.classifications:
    color = colors[classification]
    for featureset in model.classifications[classification]:
        plt.scatter(featureset[0], featureset[1], marker="x", color=color, s=100, linewidths=50)
        
unknowns = np.array([[1,3],
                     [8,9],
                     [0,3],
                     [5,4],
                     [6,4]])

for unknown in unknowns:
    classification = model.predict(unknown)
    plt.scatter(unknown[0], unknown[1], marker="*", color=colors[classification], s=100, linewidths=3)

plt.title('K_means implementation Result')
plt.show()


