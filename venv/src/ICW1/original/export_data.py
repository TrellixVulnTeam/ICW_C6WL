import pandas as pd
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
from sklearn.manifold import TSNE
from scipy.spatial import distance
# return same sample dataset every time
np.random.seed(0)

# Generate sample data
n_points_per_cluster_total =5000000
print("total points")
print(n_points_per_cluster_total)

size_colum = 100
centers = np.random.randint(-100, 100, size=(size_colum,size_colum))

X, labels_true = make_blobs(n_samples=n_points_per_cluster_total,
                            centers=centers, center_box=(-100.0, 100.0),
                            n_features=size_colum, cluster_std=0.4,
                            random_state=0)
X = StandardScaler().fit_transform(X)
print(X.shape)
#-----------------------------------------
# save data to csv file in order to use ELKI open source
dataset = pd.DataFrame(X)
dataset.to_csv(str(n_points_per_cluster_total) + '_makeblobs_100d.csv',sep=' ',index=False, header=False)