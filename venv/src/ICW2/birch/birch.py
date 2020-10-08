#https://www.datatechnotes.com/2019/09/clustering-example-with-birch-method-in.html
from sklearn.cluster import Birch
import numpy as np
import matplotlib.pyplot as plt
#data set
from sklearn.datasets.samples_generator import make_blobs
# caculated time
import time

np.random.seed(0)
# p1 = np.random.randint(-1000,1000,size= 10000)
# p2 = np.random.randint(-1000,1000,size= 10000)
# p3 = np.random.randint(-1000,1000,size= 10000)
# data = np.array(np.concatenate([p1, p2, p3]))
# x_range = range(len(data))
# print(x_range)
# x = np.array(list(zip(x_range, data))).reshape(600, 100)
# # plt.scatter(x[:,0], x[:,1])
# # plt.show()

# van tinh from icw1
n_points_per_cluster_total = 10000
size_colum = 100
centers = np.random.randint(-100, 100, size=(size_colum,size_colum))

x, labels_true = make_blobs(n_samples=n_points_per_cluster_total,centers=centers, n_features=size_colum, cluster_std=0.4, random_state=0)
print(x.shape)
# computed time of algorithmus
db_time = time.time()

bclust=Birch(branching_factor=100, threshold=.5).fit(x)
print(bclust)

labels = bclust.predict(x)

db_time_process = time.time() - db_time
print('Elapsed time to cluster in DBSCANN :  %.4f s ' % db_time_process)
print(len(labels))
plt.scatter(x[:,0], x[:,1], c=labels)
plt.show()