print(__doc__)

import numpy as np
import time
import datetime

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
#make the random number is same every each run
np.random.seed(0)
# #############################################################################
# Generate sample data
centers = [[1, 1], [-1, -1], [1, -1], [1, -1]]
n_points_per_cluster_total = 1000
size_colum = 200
n_samples = int(n_points_per_cluster_total *  size_colum / 2)

X, labels_true = make_blobs(n_samples=n_samples,centers=centers, n_features=3, cluster_std=0.4, random_state=0)
print(X)
X = X.reshape(n_points_per_cluster_total,size_colum)
print(X)
X = StandardScaler().fit_transform(X)
# def generate_data_sample_uniform(low, high, length_vector,size_colum):
#     return np.random.uniform(low, high, size = (length_vector, size_colum))
#
# print(X)
# def generate_10_cluster_sample_random(n_points_per_cluster_total, size_matrix):
#     n_points_per_cluster = int(n_points_per_cluster_total / 10)
#     C1 =  .8 * np.random.randn(n_points_per_cluster, size_matrix)
#     X = np.vstack((C1))
#     return X
#X = generate_data_sample(low = -1, high = 1, length_vector = 1000, size_colum=200)
#X = generate_10_cluster_sample_random(n_points_per_cluster_total, size_colum)
#X = StandardScaler().fit_transform(X)
#print(X)
# #############################################################################
# Compute DBSCAN
db_time = time.time()
db = DBSCAN(eps=0.5, min_samples=10).fit(X)

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
import matplotlib.pyplot as plt

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]
    # true false array
    class_member_mask = (labels == k)
# plot for cluster
    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)
#the graph for noise point
    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.xlabel(str(datetime.timedelta(seconds=round(db_time_process,4), )) + ' s')
plt.ylabel('Achse Y')
plt.title('Number of clusters: %d' % n_clusters_ +
            "The total point are: %d" % n_points_per_cluster_total +
            "\n Size of colum is %d" % size_colum )
plt.grid(True)
plt.savefig('praxis/dbscan_%d'%n_points_per_cluster_total+'.png')
plt.show()
