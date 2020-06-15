# -*- coding: utf-8 -*-
"""
===================================
Demo of HDBSCAN clustering algorithm
===================================
Finds a clustering that has the greatest stability over a range
of epsilon values for standard DBSCAN. This allows clusterings
of different densities unlike DBSCAN.
"""
#print(__doc__)

import numpy as np
import time
import datetime

from hdbscan import HDBSCAN
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
#make the random number is same every each run
np.random.seed(0)

# def make_var_density_blobs(n_samples=1000, centers=[[0, 0]], cluster_std=[0.5], random_state=0):
#     samples_per_blob = n_samples // len(centers)
#     blobs = [make_blobs(n_samples=samples_per_blob, centers=[c], cluster_std=cluster_std[i])[0]
#              for i, c in enumerate(centers)]
#     labels = [i * np.ones(samples_per_blob) for i in range(len(centers))]
#     return np.vstack(blobs), np.hstack(labels)
#
# def generate_10_cluster_sample_random(n_points_per_cluster_total, size_matrix):
#     C1 =  .8 * np.random.randn(n_points_per_cluster_total, size_matrix)
#     X = np.vstack((C1))
#     return X

#############################################################################
# Generate Date Set
n_points_per_cluster_total = 50000
size_colum = 100
centers = np.random.randint(-100, 100, size=(size_colum,size_colum))

X, labels_true = make_blobs(n_samples=n_points_per_cluster_total,centers=centers,
                            n_features=size_colum, cluster_std=0.4, random_state=0)
print(X.shape)

##############################################################################
# Compute HDBSCAN
hdb_t1 = time.time()
hdb = HDBSCAN(min_cluster_size=10).fit(X)
hdb_labels = hdb.labels_
hdb_elapsed_time = time.time() - hdb_t1

n_clusters_hdb_ = len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0)

print('\n\n++ HDBSCAN Results')
print('Len of Cluster are : ', len(X))
print('Estimated number of clusters: %d' % n_clusters_hdb_)
print('Elapsed time to cluster: %.4f s' % hdb_elapsed_time)
# Plot result
import matplotlib.pyplot as plt

# Black removed and is used for noise instead.
hdb_unique_labels = set(hdb_labels)
hdb_colors = plt.cm.Spectral(np.linspace(0, 1, len(hdb_unique_labels)))
fig = plt.figure()
ax = plt.axes(projection='3d')
pca = PCA(3)
projected = pca.fit_transform(X)

for k, col in zip(hdb_unique_labels, hdb_colors):
    if k == -1:
        # Black used for noise.
        col = 'k'
    ax.plot3D(projected[hdb_labels == k, 0], X[hdb_labels == k, 1],X[hdb_labels == k, 2], 'o', markerfacecolor=col,
                  markeredgecolor='k', markersize=6)

plt.xlabel(str(datetime.timedelta(seconds=round(hdb_elapsed_time,4), )) + ' s')
plt.ylabel('Achse Y')
plt.title('Number of clusters: %d' % n_clusters_hdb_ +
            "\n The total point are: %d" % n_points_per_cluster_total +
            "\n Size of colum is %d" % size_colum )
plt.grid(True)
plt.savefig('praxis/hdbscan_%d'%n_points_per_cluster_total+'.png')
plt.show()