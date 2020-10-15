# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html#sklearn.cluster.MiniBatchKMeans
## very okie
from sklearn.cluster import MiniBatchKMeans
import numpy as np
# for data set
from sklearn.datasets.samples_generator import make_blobs
#time measure
import time
import matplotlib.pyplot as plt
# get dataset
n_points_per_cluster_total = 3000000
size_colum = 100
centers = np.random.randint(-20, 20, size=(size_colum,size_colum))

X, labels_true = make_blobs(n_samples=n_points_per_cluster_total,centers=centers, n_features=size_colum, cluster_std=0.4, random_state=0)
print(X.shape)

db_time = time.time()
kmeans = MiniBatchKMeans(n_clusters=100,
random_state = 0,
batch_size = 300,
max_iter = 100).fit(X)

db_time_process = time.time() - db_time
print('Elapsed time to cluster in MInibatch kmeans :  %.4f s ' % db_time_process)

cluster = kmeans.cluster_centers_
labels = kmeans.labels_
print(len(cluster))

# fragen auf stack over flow https://stackoverflow.com/questions/47270604/python-kmeans-clustering-for-large- datasets
# he so la : kmeans = MiniBatchKMeans(nclusters = 100, randomstate = 0, batchsize = 300, maxiter = 100).fit(X)
# 1trieu = 56s < kmeans = 206s
# 3trrieu = 79s < kmeans = 764.9357 s
# 4trieu = 116.7942s(5000000,100)
# 5trieu ElapsedtimetoclusterinMInibatchkmeans : 190.0804s
