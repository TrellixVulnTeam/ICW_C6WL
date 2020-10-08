#https://stackoverflow.com/questions/26246015/python-dbscan-in-3-dimensional-space
class PDBSCAN(object):

    def __init__(self, eps=0, min_points=2):
        self.eps = eps
        self.min_points = min_points
        self.visited = []
        self.noise = []
        self.clusters = []
        self.dp = []

    def cluster(self, data_points):
        self.visited = []
        self.dp = data_points
        # count cluster
        c = 0
        for point in data_points:
            if point not in self.visited:
                self.visited.append(point)
                # find neighbours
                neighbours = self.region_query(point)
                if len(neighbours) < self.min_points:
                    self.noise.append(point)
                else:
                    c += 1
                    self.expand_cluster(c, neighbours)

    def expand_cluster(self, cluster_number, p_neighbours):
        cluster = ("Cluster: %d" % cluster_number, [])
        self.clusters.append(cluster)
        new_points = p_neighbours
        while new_points:
            new_points = self.pool(cluster, new_points)

    # find neighbours
    def region_query(self, p):
        result = []
        for d in self.dp:
            distance = (((d[0] - p[0])**2 + (d[1] - p[1])**2 + (d[2] - p[2])**2)**0.5)
            if distance <= self.eps:
                result.append(d)
        return result

    def pool(self, cluster, p_neighbours):
        new_neighbours = []
        for n in p_neighbours:
            if n not in self.visited:
                self.visited.append(n)
                n_neighbours = self.region_query(n)
                if len(n_neighbours) >= self.min_points:
                    new_neighbours = self.unexplored(p_neighbours, n_neighbours)
            for c in self.clusters:
                if n not in c[1] and n not in cluster[1]:
                    cluster[1].append(n)
        return new_neighbours

    @staticmethod
    def unexplored(x, y):
        z = []
        for p in y:
            if p not in x:
                z.append(p)
        return z
###############################################################################################
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
from rtree import index
np.random.seed(0)
# #############################################################################
# Generate sample data
n_points_per_cluster_total = 1000
print("total points")
print(n_points_per_cluster_total)

size_colum = 100
centers = np.random.randint(-100, 100, size=(size_colum,size_colum))

X, labels_true = make_blobs(n_samples=n_points_per_cluster_total,
                            centers=centers, center_box=(-100.0, 100.0),
                            n_features=size_colum, cluster_std=0.4,
                            random_state=0)
X = StandardScaler().fit_transform(X)
#print(X)
db = PDBSCAN(eps=6,min_points=10)
cluster = db.cluster(X)
print(cluster)