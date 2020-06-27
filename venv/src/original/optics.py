# Authors: Shane Grigsby <refuge@rocktalus.com>
#          Adrin Jalali <adrin.jalali@gmail.com>
# License: BSD 3 clause

from sklearn.cluster import OPTICS, cluster_optics_dbscan
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import time
import datetime
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# Create data sample for algorithmus
np.random.seed(0)
n_points_per_cluster_total = 10000
size_colum = 100
centers = np.random.randint(-100, 100, size=(size_colum,size_colum))

X, labels_true = make_blobs(n_samples=n_points_per_cluster_total,centers=centers,
                            n_features=size_colum, cluster_std=0.4, random_state=0)
X = StandardScaler().fit_transform(X)

#handle optics algorithm
op_time = time.time()
clust = OPTICS(min_samples=50, xi=.05, min_cluster_size=.05, algorithm="kd_tree",n_jobs=-1).fit(X)
space = np.arange(len(X))
reachability = clust.reachability_[clust.ordering_]
labels = clust.labels_[clust.ordering_]
op_time_processing = time.time() - op_time
op_labels = clust.labels_
n_clusters_op_ = len(set(op_labels)) - (1 if -1 in op_labels else 0)
print('The Len of cluster is: ', len(X))
print('Cluster sind ', n_clusters_op_)
print('The point are  %d' % n_points_per_cluster_total)
print('Elapsed time to cluster in Optics :  %.4f s ' % op_time_processing)
# End handle Optics
# OPTICS
#convert to 3 dimension by PCA
#pca = PCA(3)
#data_set_then = pca.fit_transform(X)

# data_set_then = TSNE(n_components=3).fit_transform(X)
#
# fig = plt.figure()
# ax = plt.axes(projection='3d')
#
# colors = ['g.', 'r.', 'b.', 'y.', 'c.', '.o', 'v.']
# for klass, color in zip(range(0, 5), colors):
#     Xk = data_set_then[clust.labels_ == klass]
#     # old ax.plot3D(Xk[:, 0], Xk[:, 1],Xk[:, 2], color, alpha=0.3)
#
#     ax.plot3D(data_set_then[clust.labels_ == -1, 0], data_set_then[clust.labels_ == -1, 1],
#           data_set_then[clust.labels_ == -1, 2], 'k+', alpha=0.3)
#
#     #ax.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3, marker='.')
#     #ax.plot(X[clust.labels_ == -1, 0], X[clust.labels_ == -1, 1], 'k+', alpha=0.1)
#
# #true or false: clust.labels_ == -1
#
# plt.xlabel(str(datetime.timedelta(seconds=round(op_time_processing,4), )) + ' s')
# plt.ylabel('Achse Y')
# plt.title('Number of clusters: %d' % n_clusters_op_ +
#             "\n The total point are: %d" % n_points_per_cluster_total +
#             "\n Size of colum is %d" % size_colum )
# plt.grid(True)
# plt.savefig('praxis/optics_%d'%n_points_per_cluster_total+'.png')
# plt.tight_layout()
# plt.show()