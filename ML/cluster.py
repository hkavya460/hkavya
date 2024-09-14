import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as shc
from scipy.cluster.hierarchy import linkage ,dendrogram
from ISLP import load_data
import warnings
from sklearn.decomposition import PCA

#Generting the random data  with 50 samples  and 2 features


# np.random.seed(0)
# x = np.random.standard_normal((50,2))
# print(x)
#
# # x[25:,0] += 3
# # x[25:,1]  -=4
#
# kmeans1 = KMeans(n_clusters=2 ,random_state=2 , n_init=1).fit(x)
# kmeans = KMeans(n_clusters=3,random_state=3,n_init=20).fit(x)
# #intertia - sum of square of error
# c = kmeans1.inertia_ , kmeans.inertia_
# print(c)
# y = kmeans.labels_
# print(y)
# fig,ax = plt.subplots(1,1,figsize=(8,8))
# ax.scatter(x[:,0],x[:,1],c=y)
# ax.set_xlabel("kmeans clusterring with cluster")
# plt.show()
#
#
# #####################################################
# #Hierarchical Clustering
#
#
# dendro =shc.dendrogram(linkage(x,method="ward"))
# plt.title("Dendrogram plot")
# plt.show()
#
# HClust = AgglomerativeClustering
# linkage_metric = linkage(x,'complete')
# hc_comp = HClust(distance_threshold =0,n_clusters=None ,linkage=linkage_metric)
# #
# # hc_comp.fit(x)
# # havg = HClust(distance_threshold =0,n_cluster=None,linkage='average')
# # havg.fit(x)
# #
# # hsingle = HClust(distance_threshold=0,n_cluster=None,linkage="ward")
# # hsingle.fit(x)
#
# D = np.zeros((x.shape[0],x.shape[0]))
# # print(D)
# for i in range (x.shape[0]):
#     xd = np.multiply.outer(np.ones(x.shape[0]),x[i])
#     D[i] = np.sqrt(sum(x-xd)**2,1)
#     # print(D[i])
#
# hsingle_com = HClust(distance_threshold=0,n_clusters= 5,metric='precomputed',linkage='single')
# hsingle_com.fit(D)
# cargs = {'colour_threshold':-np.inf,'above_threshold_colour':'black'}
# linkage_comp = compute_linkage(hc_comp)
# fig,ax =plt.subplots(1,1 ,figsize=(8,8))
# dendrogram(linkage_comp,color_threshold=4,ax=ax,above_threshold_color='black')
# plt.show()



#####################################################

#IA_2 3rd question

x = np.array([[1,4],[1,3],[0,4],[5,1],[6,2],[4,0]])


##plot the observations

k=  2
centroids = x[np.random.choice(np.arange(1,len(x)),k,replace=False)]
print(centroids)

clusters = {i:[] for  i in  range (k)}
print(clusters)

for data in x:
    distance = [np.linalg.norm(data - centroid)  for centroid in centroids]
    dist_index = np.argmin(distance)

    clusters[dist_index].append(data)

for i in range(k):
    centroids[i] = np.mean(clusters[i],axis=0)
print(clusters)


model = KMeans(n_clusters=2,random_state=42,n_init=1)
model.fit(x)
y = model.labels_


# plt.scatter(x[:,0],x[:,1],c=y)
# plt.show()



########################################
nci = load_data('NCI60')
nci_data = nci['data']
print(nci_data.shape)
y = nci['labels']
print(y.value_counts())

#a)
pca = PCA(n_components=10)
scale = StandardScaler()
scaled_data = scale.fit_transform(nci_data)
score = pca.fit_transform(scaled_data)

variance = pca.explained_variance_ratio_
cum_sum = pca.explained_variance_ratio_.cumsum()
print(cum_sum)
fig,ax  =plt.subplots(1,1,figsize=(8,6))

ax.scatter(score[:,0],score[:,1],score[:,2],label=y)
plt.title("pca components ")
# plt.xlabel()
plt.show()

######################





