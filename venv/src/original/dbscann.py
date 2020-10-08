print(__doc__)

import numpy as np
import time
import datetime

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
#make the random number is same every each run
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial import distance
#from rtree import index
np.random.seed(0)
# #############################################################################
# Generate sample data
n_points_per_cluster_total = 10000
print("total points")
print(n_points_per_cluster_total)

size_colum = 100
centers = np.random.randint(-100, 100, size=(size_colum,size_colum))

X, labels_true = make_blobs(n_samples=n_points_per_cluster_total,
                            centers=centers, center_box=(-100.0, 100.0),
                            n_features=size_colum, cluster_std=0.4,
                            random_state=0)
X = StandardScaler().fit_transform(X)
print(X)

# #############################################################################
# Indexing  dataset make_blobs
# idx = index.Index()
# left, bottom, right, top = (-5, -5, 5, 5)
# X = idx.insert(X, (left, bottom, right, top))
# print(X)

# #############################################################################
# Compute DBSCAN
db_time = time.time()
epsilon  = 6
print("epsilon")
print( epsilon)
min_samples = 10
print("min_samples")
print( min_samples)
#ball tree is the best
db = DBSCAN(eps=epsilon,algorithm='ball_tree',leaf_size=10, min_samples=min_samples, n_jobs = -1).fit(X)
#array false for core samples mask
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
#set for db.core sample indices as true
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
db_time_process = time.time() - db_time

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
print('The total point are  %d' % n_points_per_cluster_total)
print('Elapsed time to cluster in DBSCANN :  %.4f s ' % db_time_process)
# #############################################################################
# Plot result
# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]

#change to 3 dimension in PCA
#pca = PCA(n_components=3)
#data_set_then = pca.fit_transform(X)
#change data set to 3 dimension with TSNE
# data_set_then = TSNE(n_components=3).fit_transform(X)
#
# fig = plt.figure(figsize=(12, 12))
# ax = plt.axes(projection='3d')
# #ax.view_init(60, 35)
# #for data in order to plot
# for k, col in zip(unique_labels, colors):
#     if k == -1:
#         # Black used for noise.
#         col = [0, 0, 0, 1]
#     # true false array
#     class_member_mask = (labels == k)
# # plot for cluster
#     #xy = X[class_member_mask & core_samples_mask]
#     xy = data_set_then[class_member_mask & core_samples_mask]
#     ax.plot3D(xy[:, 0], xy[:, 1], xy[:, 2],  'o', markerfacecolor=tuple(col),
#              markeredgecolor='k', markersize=14)
#     #plt.plot(xy[:, 0], xy[:, 1])
#
# #the graph for noise point
#     xy = X[class_member_mask & ~core_samples_mask]
#
#     plt.plot(xy[:, 0], xy[:, 1], xy[:, 2], 'o', markerfacecolor=tuple(col),
#              markeredgecolor='k', markersize=6)
#
# plt.xlabel(str(datetime.timedelta(seconds=round(db_time_process,4), )) + ' s')
# plt.ylabel('Epsilon:'+ str(epsilon) +',mini_samples:'+str(min_samples))
# plt.title('Number of clusters: %d' % n_clusters_ +
#             " Total point are: %d" % n_points_per_cluster_total +
#             "\n Size of colum is %d" % size_colum )
# plt.grid(True)
# plt.savefig('praxis/dbscan_%d'%n_points_per_cluster_total+'.png')
# plt.show()
