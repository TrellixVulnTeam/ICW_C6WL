#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
An example demonstrating PowerIterationClustering.
Run with:
  bin/spark-submit examples/src/main/python/ml/power_iteration_clustering_example.py
"""
# $example on$
from pyspark.ml.clustering import PowerIterationClustering
# $example off$
from pyspark.sql import SparkSession


import numpy as np
import time
import datetime

from hdbscan import HDBSCAN
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.decomposition import PCA
#make the random number is same every each run
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial import distance
np.random.seed(0)
if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("Hdbscann")\
        .getOrCreate()

    # # $example on$
    # df = spark.createDataFrame([
    #     (0, 1, 1.0),
    #     (0, 2, 1.0),
    #     (1, 2, 1.0),
    #     (3, 4, 1.0),
    #     (4, 0, 0.1)
    # ], ["src", "dst", "weight"])

    #pic = PowerIterationClustering(k=2, maxIter=20, initMode="degree", weightCol="weight")

    # Shows the cluster assignment
    #pic.assignClusters(df).show()
    # $example off$
    n_points_per_cluster_total = 100000
    print("total points: " + str(n_points_per_cluster_total))

    size_colum = 100
    centers = np.random.randint(-100, 100, size=(size_colum, size_colum))

    X, labels_true = make_blobs(n_samples=n_points_per_cluster_total,
                                centers=centers, center_box=(-100.0, 100.0),
                                n_features=size_colum, cluster_std=0.4,
                                random_state=0)
    X = StandardScaler().fit_transform(X)
    print("X.shape: " + str(X.shape))
    # #############################################################################
    # Compute HDBSCAN
    hdb_t1 = time.time()
    min_cluster_size  = 10
    print("min_cluster_size : " + str(min_cluster_size))
    hdb = HDBSCAN(min_cluster_size=min_cluster_size, algorithm='prims_balltree', core_dist_n_jobs=-1, ).fit(X)
    hdb_labels = hdb.labels_
    hdb_elapsed_time = time.time() - hdb_t1

    n_clusters_hdb_ = len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0)

    print('\n\n++ HDBSCAN Results')
    print('Len of Cluster are : ', len(X))
    print('Estimated number of clusters: %d' % n_clusters_hdb_)
    print('Elapsed time to cluster: %.4f s' % hdb_elapsed_time)
    spark.stop()
