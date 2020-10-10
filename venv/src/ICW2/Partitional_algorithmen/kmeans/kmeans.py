#https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
from sklearn.cluster import KMeans
import numpy as np
# for data set
from sklearn.datasets.samples_generator import make_blobs
#time measure
import time
import matplotlib.pyplot as plt
##X = np.array([[1, 2], [1, 4], [1, 0],
##              [10, 2], [10, 4], [10, 0]])

n_points_per_cluster_total = 1000000
size_colum = 100
centers = np.random.randint(-1000, 1000, size=(size_colum,size_colum))

X, labels_true = make_blobs(n_samples=n_points_per_cluster_total,centers=centers, n_features=size_colum, cluster_std=0.4, random_state=0)
print(X.shape)
db_time = time.time()
kmeans = KMeans(random_state=0, n_jobs=-1).fit(X)
kmeans.predict(X)

db_time_process = time.time() - db_time
print('Elapsed time to cluster in MInibatch kmeans :  %.4f s ' % db_time_process)

print(len(kmeans.cluster_centers_))
# main website: https://realpython.com/k-means-clustering-python/#understanding-the-k-means-algorithm

# paramter KMeans(n_clusters=100, random_state=0, n_jobs=-1).fit(X)
### 100 dimension
# 1 milion 206 s,
# 2 milion , 426.6510 s
# 3 milion  764.9357 s
#4 milion 996.6241 s
# 5 milion 1215.7721 s
# 6 milion 1417.5992 s
# 7 milion
#  8 milion

#### 100 dimension but do not have the parameter n_clusters
#kmeans = KMeans(n_clusters=100, random_state=0, n_jobs=-1).fit(X)
# 1 mi
# 2 mi
# 3 mi


### 200 dimension

###Evaluating Clustering Performance Using Advanced Techniques
