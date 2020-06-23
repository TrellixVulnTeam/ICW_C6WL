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
        .appName("dbscann")\
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
    n_points_per_cluster_total = 1000000
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
    # Compute DBSCAN
    db_time = time.time()
    epsilon = 12
    print("epsilon: " + str(epsilon))
    min_samples = 10
    print("min_samples: m" + str(min_samples))
    db = DBSCAN(eps=epsilon, algorithm='ball_tree', min_samples=min_samples, n_jobs=-1).fit(X)
    # array false for core samples mask
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    # set for db.core sample indices as true
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
    spark.stop()
