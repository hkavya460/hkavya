import random
import pandas as pd
import numpy as np
import math
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import warnings


#kmeans  clustering  - based on eucliedan distance  (centroid - point )**2
np.random.seed(0)
x = np.random.standard_normal((50,2))
print(x)
#intalizing the number of clusters to 2
k = 2
centroids = x[np.random.choice(range(len(x)),k,replace=False)]
clusters = {i :[] for i in range(k)}
for data in x:
    distance = [np.linalg.norm(data - centroid) for centroid in centroids]
    index_dist = np.argmin(distance)
    clusters[index_dist].append(data)

for i in  range(k):
    centroids[i] = np.mean(clusters[i],axis=0)


print(clusters)
fig,ax = plt.subplots(figsize=(8,6))
for i ,(clusters,centroid) in enumerate (zip(clusters.values(),centroids)):
    color ="blue" if i ==0 else "green"
    plt.scatter(np.array(clusters)[:,0],np.array(clusters)[:,1],color=color)
    plt.scatter(centroid[0],centroid[1],marker="*",s=100,label="centroid",color="red")


plt.legend()
plt.show()


































# print(x)

wcss_list = []
# plt.scatter(x[:,0],x[:,1],color="red")
# plt.show()
#using sharp elbow method to find the number of clusters  using wcss (within cluster sum square)

# for i in range  (1 ,11):
#     np.random.seed(0)
#     k_model = KMeans(n_clusters= i , init = 'k-means++', random_state=42)
#     k_model.fit(x)
#     wcss_list.append(k_model.inertia_)
# plt.plot(range(1,11),wcss_list,color = "blue")
# plt.show()


# by this elbow method  number of cluster is 5

# k_model = KMeans(n_clusters=5,init= 'k-means++',random_state=42)
# k_model.fit(x)
# y_pred = k_model.predict(x)
# # Plotting the clusters
# plt.scatter(x[:, 0], x[:, 1], c=y_pred, cmap='viridis')
# plt.scatter(k_model.cluster_centers_[:, 0], k_model.cluster_centers_[:, 1], s=100, c='red', marker='+', label='Centroids')
# plt.title('KMeans Clustering')
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.legend()
# plt.show()












