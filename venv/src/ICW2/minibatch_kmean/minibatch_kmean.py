# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html#sklearn.cluster.MiniBatchKMeans
from sklearn.cluster import MiniBatchKMeans
import numpy as np
# for data set
from sklearn.datasets.samples_generator import make_blobs
#time measure
import time
# X = np.array([[1, 2], [1, 4], [1, 0],
#               [4, 2], [4, 0], [4, 4],
#               [4, 5], [0, 1], [2, 2],
#               [3, 2], [5, 5], [1, -1]])
# manually fit on batches
# kmeans = MiniBatchKMeans(n_clusters=2,
#random_state = 0,
#batch_size = 6)
# kmeans = kmeans.partial_fit(X[0:6,:])
# print(kmeans.cluster_centers_)
# kmeans = kmeans.partial_fit(X[6:12,:])
# kmeans.cluster_centers_
#
#
# kmeans.predict([[0, 0], [4, 4]])
#
# # fit on the whole data
n_points_per_cluster_total = 1000000
size_colum = 100
centers = np.random.randint(-1000, 1000, size=(size_colum,size_colum))

X, labels_true = make_blobs(n_samples=n_points_per_cluster_total,centers=centers, n_features=size_colum, cluster_std=0.4, random_state=0)
print(X.shape)

db_time = time.time()

kmeans = MiniBatchKMeans(n_clusters=102,
random_state = 0,
batch_size = 300,
max_iter = 100).fit(X)

db_time_process = time.time() - db_time
print('Elapsed time to cluster in MInibatch kmeans :  %.4f s ' % db_time_process)

cluster = kmeans.cluster_centers_
labels = kmeans.labels_
print(labels)
print(len(cluster))
#
