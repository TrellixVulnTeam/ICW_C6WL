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
np.random.seed(0)
# #############################################################################
# Generate sample data
# the following code should be added to the question's code (it uses G and db)

import igraph

# use igraph to calculate Jaccard distances quickly
edges = zip(*nx.to_edgelist(G))
G1 = igraph.Graph(len(G), zip(*edges[:2]))
D = 1 - np.array(G1.similarity_jaccard(loops=False))

# DBSCAN is much faster with metric='precomputed'
t = time.time()
db1 = dbscan(D, metric='precomputed', eps=0.85, min_samples=2)
print("clustering took %.5f seconds" %(time.time()-t))

#assert np.array_equal(db, db1)