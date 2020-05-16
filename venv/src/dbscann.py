print(__doc__)

import numpy as np
import time

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# #############################################################################
# Generate sample data
def generate_data_sample(length_vector,size_colum, low, high):
    return np.random.uniform(low, high, size = (length_vector, size_colum))
#X = generate_data_sample(10000, 2, -10, 10)
def generate_10_cluster_sample(n_points_per_cluster):
    C1 = [-5, -2] + .8 * np.random.randn(n_points_per_cluster, 2)
    C2 = [1, -1] + .1 * np.random.randn(n_points_per_cluster, 2)
    C3 = [1, -2] + .2 * np.random.randn(n_points_per_cluster, 2)
    C4 = [-2, 3] + .3 * np.random.randn(n_points_per_cluster, 2)
    C5 = [3, -2] + 1.6 * np.random.randn(n_points_per_cluster, 2)
    C6 = [5, 6] + 2 * np.random.randn(n_points_per_cluster, 2)
    C7 = [1, 6] + 2 * np.random.randn(n_points_per_cluster, 2)
    C8 = [5, -3] + 2 * np.random.randn(n_points_per_cluster, 2)
    C9 = [1, -2] + 2 * np.random.randn(n_points_per_cluster, 2)
    C10 = [1, 9] + 2 * np.random.randn(n_points_per_cluster, 2)
    X = np.vstack((C1, C2, C3, C4, C5, C6, C7, C8, C9, C10))
    return X
n_points_per_cluster = 50000
X = generate_10_cluster_sample(n_points_per_cluster)
print(X)
# #############################################################################
# Compute DBSCAN
db_time = time.time()
db = DBSCAN(eps=0.5, min_samples=5).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
db_time_process = time.time() - db_time

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
print('The point are  %d' % n_points_per_cluster)
print('Time for processing DBSCANN :  %.4f s ' % db_time_process)
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

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
