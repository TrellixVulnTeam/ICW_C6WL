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
An example demonstrating k-means clustering.
Run with:
  bin/spark-submit examples/src/main/python/ml/kmeans_example.py

This example requires NumPy (http://www.numpy.org/).
"""
from __future__ import print_function
import numpy as np
import time
import datetime

from hdbscan import HDBSCAN
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
import csv

# $example on$
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
# $example off$

from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
if __name__ == "__main__":
    conf =  SparkConf().setAppName("HDBScanExample").set

    sc = SparkContext(conf=conf)

    n_points_per_cluster_total = 500000
    print("total points")
    print(n_points_per_cluster_total)

    size_colum = 100
    centers = np.random.randint(-100, 100, size=(size_colum, size_colum))

    X, labels_true = make_blobs(n_samples=n_points_per_cluster_total,
                                centers=centers, center_box=(-100.0, 100.0),
                                n_features=size_colum, cluster_std=0.4,
                                random_state=0)
    X = StandardScaler().fit_transform(X)
    # read data with method 1, Parallelized Collections
    #https://spark.apache.org/docs/latest/rdd-programming-guide.html
    distData = sc.parallelize(X)
    collectedData = np.array(distData.collect())
    # Compute HDBSCAN
    hdb_t1 = time.time()
    min_cluster_size = 10
    hdb = HDBSCAN(min_cluster_size=min_cluster_size, algorithm='boruvka_kdtree', core_dist_n_jobs=-1, ).fit(X)
    hdb_labels = hdb.labels_
    hdb_elapsed_time = time.time() - hdb_t1

    n_clusters_hdb_ = len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0)

    print('\n\n++ HDBSCAN Results')
    print('Len of Cluster are : ', len(X))
    print('Estimated number of clusters: %d' % n_clusters_hdb_)
    print('Elapsed time to cluster: %.4f s' % hdb_elapsed_time)

    sc.stop()
