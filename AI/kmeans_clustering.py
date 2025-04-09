import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split , KFold
from sklearn.metrics import accuracy_score ,r2_score ,mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

#Kmeans clustering  from scracth

#################################################################

def kmeans_clustering():
    x = np.array([[1,4],[1,3],[0,4],[5,1],[6,2],[4,0]])
##plot the observations to see how the values are distributed

    k=  2 # forming 2 clusters
    centroids = x[np.random.choice(np.arange(1,len(x)),k,replace=False)] # forming 2 random data points as  centroids
    print(centroids)
    clusters = {i:[] for  i in  range (k)}  # 2 empty list for appending the datapoints based on euclidean distance
    print(clusters)

    for data in x:
        distance = [np.linalg.norm(data - centroid)  for centroid in centroids] # finding the distance between centroid and datapoint
        dist_index = np.argmin(distance)

        clusters[dist_index].append(data)

    for i in range(k): # updating the centroid based on mean value of the cluster
        centroids[i] = np.mean(clusters[i],axis=0)
    print(clusters)


    model = KMeans(n_clusters=2,random_state=42,n_init=1)
    model.fit(x)
    y = model.labels_


    plt.scatter(x[:,0],x[:,1],c=y)
    plt.title("kmeans clustering")
    plt.show()

def main():
    kmeans_clustering()
if __name__=="__main__":
    main()







