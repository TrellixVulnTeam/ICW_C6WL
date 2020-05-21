# Authors: Shane Grigsby <refuge@rocktalus.com>
#          Adrin Jalali <adrin.jalali@gmail.com>
# License: BSD 3 clause

from sklearn.cluster import OPTICS, cluster_optics_dbscan
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import time

# Generate sample data
np.random.seed(0)
def generate_10_cluster_sample(n_points_per_cluster_total):
    n_points_per_cluster = int(n_points_per_cluster_total / 10)
    C1 = [-5, -2 ] + .8 * np.random.randn(n_points_per_cluster, 2)
    C2 = [1, -1 ] + .1 * np.random.randn(n_points_per_cluster, 2)
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
n_points_per_cluster_total = 50000
X = generate_10_cluster_sample(n_points_per_cluster_total)
#handle optics algorithm
op_time = time.time()
clust = OPTICS(min_samples=50, xi=.05, min_cluster_size=.05).fit(X)
space = np.arange(len(X))
reachability = clust.reachability_[clust.ordering_]
labels = clust.labels_[clust.ordering_]
op_time_processing = time.time() - op_time
op_labels = clust.labels_
n_clusters_op_ = len(set(op_labels)) - (1 if -1 in op_labels else 0)
#print('The Len of cluster is: ', len(X))
#print('The point are  %d' % n_points_per_cluster_total)
#print('Elapsed time to cluster in Optics :  %.4f s ' % op_time_processing)
# End handle Optics
# OPTICS
colors = ['g.', 'r.', 'b.', 'y.', 'c.']
for klass, color in zip(range(0, 5), colors):
    Xk = X[clust.labels_ == klass]
    plt.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
plt.plot(X[clust.labels_ == -1, 0], X[clust.labels_ == -1, 1], 'k+', alpha=0.1)

plt.xlabel('Achse X')
plt.ylabel('Achse Y')
plt.title('Optics: number of clusters %d' % n_clusters_op_ +
            " ,The total point are: %d" % n_points_per_cluster_total +
            "\n Elapsed time to cluster:  %.4f s" % op_time_processing )
plt.grid(True)
plt.savefig('praxis/optics_%d'%n_points_per_cluster_total+'.png')
plt.tight_layout()
plt.show()