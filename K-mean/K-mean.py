from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
from sklearn.datasets import make_circles
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage

#raw_dataX, raw_dataY=make_blobs(n_samples=300,n_features=2,centers=7)
#raw_dataX, raw_dataY=make_circles(n_samples=300,noise=0.01)
#raw_dataX, raw_dataY=make_moons(n_samples=300,noise=0.1)


raw_dataX=pd.read_csv('2d_points.tsv',sep='\t').to_numpy()
plt.subplot(1,2,1)
plt.scatter(raw_dataX[:,0],raw_dataX[:,1],c='blue')
'''
linked=linkage(raw_dataX,'ward')
plt.figure(figsize=(10,7))
dendrogram(linked,
           orientation='top',
           distance_sort='descending',
           show_leaf_counts=True)
'''

model=AgglomerativeClustering(n_clusters=None,distance_threshold=10)
model.fit(raw_dataX)

print(model.n_clusters_)


model2=KMeans(n_clusters=model.n_clusters_)
start_time=time.time()
model2.fit(raw_dataX)
worktime=time.time()-start_time

plt.subplot(1,2,2)
plt.scatter(raw_dataX[:,0],raw_dataX[:,1],c=model2.fit_predict(raw_dataX))

print(worktime)
plt.show()