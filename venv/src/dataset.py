from __future__ import print_function
import time
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
#matplotlib inline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.manifold import TSNE

print(__doc__)

import time
import datetime

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
np.random.seed(0)
# #############################################################################

n_points_per_cluster_total = 1000
size_colum = 100
centers = np.random.randint(-100, 100, size=(size_colum,size_colum))
print(centers)
X, labels_true = make_blobs(n_samples=n_points_per_cluster_total,centers=centers,n_features=size_colum, cluster_std=0.4, random_state=0)
#https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
X = StandardScaler().fit_transform(X)
print(X)

fig = plt.figure(figsize=(12, 12))
#ax = plt.axes(projection='3d')
#ax.view_init(60, 35)
colors = ['g', 'b', 'v', 'y', 'c', 'o', 'v', 'p', 'g', 'g']
projected = PCA(n_components=2).fit_transform(X)
#projected = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300).fit_transform(X)

for klass, color in zip(range(0, 9), colors):
    #ax.plot3D(projected[:, 0], projected[:, 1],projected[:, 2],color, alpha=0.3)
    #ax.scatter(projected[:, 0], projected[:, 1], cmap='tab10')
    #plt.plot(projected[:, 0], projected[:, 1])
    plt.scatter(projected[:, 0], projected[:, 1])
plt.xlabel('PCA 100 dimension')
plt.ylabel('component 2')
plt.show()
