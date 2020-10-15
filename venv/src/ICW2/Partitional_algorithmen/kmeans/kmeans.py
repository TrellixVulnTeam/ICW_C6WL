#https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
#https://de.wikipedia.org/wiki/K-Means-Algorithmus
# main website: https://realpython.com/k-means-clustering-python/#understanding-the-k-means-algorithm

# Vorteil:
# k-Means ++ versucht bessere Startpunkte zu finden.[9]
# Der Filtering-Algorithmus verwendet als Datenstruktur einen k-d-Baum.[10]
# Der k-Means-Algorithmus kann beschleunigt werden unter Berücksichtigung der Dreiecksungleichung.[11]
# Bisecting k-means beginnt mit {\displaystyle k=2}k=2, und teilt dann immer den größten Cluster, bis das gewünschte k erreicht ist.
# X-means beginnt mit {\displaystyle k=2}k=2 und erhöht k so lange, bis sich ein sekundäres Kriterium (Akaike-Informationskriterium,
# oder bayessches Informationskriterium) nicht weiter verbessert.

##### Nachteile:
# das ergebniss hängt von den Parameter wie dbscan ab, deswegen nicht optimiertes Ergebniss
# noise punkte wurden nicht automatisch schnell entdeckt


# Dlib[17]
# ELKI enthält die Varianten von Lloyd und MacQueen, dazu verschiedene Strategien für die Startwerte wie k-means++,
# und Varianten des Algorithmus wie k-medians, k-medoids und PAM.
# GNU R enthält die Varianten von Hartigan, Lloyd und MacQueen, und zusätzliche Variationen im Erweiterungspaket „flexclust“.
# OpenCV enthält eine auf Bildverarbeitung optimierte Version von k-means (inkl. k-means++ seeding)
# Scikit-learn enthält k-means, inkl. Elkans Variante und k-means++.
# Weka enthält k-means (inkl. k-means++ seeding) und die Erweiterung x-means.
#### K means ++


# malen: https://scikit-learn.org/stable/auto_examples/cluster/plot_mini_batch_kmeans.html#sphx-glr-auto-examples-cluster-plot-mini-batch-kmeans-py

from sklearn.cluster import KMeans
import numpy as np
# for data set
from sklearn.datasets.samples_generator import make_blobs
#time measure
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

n_points_per_cluster_total = 7000000
size_colum = 100
centers = np.random.randint(-20, 20, size=(size_colum,size_colum))

X, labels_true = make_blobs(n_samples=n_points_per_cluster_total,centers=centers, n_features=size_colum, cluster_std=0.4, random_state=0)
print(X.shape)

#################################################################### plot data before run
# unique_labels = set(labels_true)
# colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# pca = PCA(3)
# projected = pca.fit_transform(X)
#
# for k, col in zip(unique_labels, colors):
#     if k == -1:
#         # Black used for noise.
#         col = 'k'
#     ax.plot3D(projected[labels_true == k, 0], projected[labels_true == k, 1],projected[labels_true == k, 2], 'o', markerfacecolor=col,
#                   markeredgecolor='k', markersize=6)
#
# plt.xlabel('Achse X')
# plt.ylabel('Achse Y')
# plt.title('PCA')
# plt.grid(True)
# plt.savefig('pca/before_%d'%len(X)+'.png')
# plt.show()
################################################################################

################################################################################
db_time = time.time()
kmeans = KMeans(random_state=0, n_jobs=-1).fit(X)
kmeans.predict(X)

db_time_process = time.time() - db_time
print('Elapsed time to cluster in kmeans :  %.4f s ' % db_time_process)

print(len(kmeans.cluster_centers_))
################################################################################
### Plot data after running
unique_labels = set(kmeans.labels_)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
fig = plt.figure()
ax = plt.axes(projection='3d')
pca = PCA(3)
projected = pca.fit_transform(X)

for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = 'k'
    ax.plot3D(projected[kmeans.labels_ == k, 0], projected[kmeans.labels_ == k, 1],projected[kmeans.labels_ == k, 2], 'o', markerfacecolor=col,
                  markeredgecolor='k', markersize=6)

plt.xlabel('Achse X')
plt.ylabel('Achse Y')
plt.title('PCA')
plt.grid(True)
plt.savefig('pca/after_%d'%len(X)+'.png')
plt.show()
################################################################################
### PRACTIVE paramter KMeans(n_clusters=100, random_state=0, n_jobs=-1).fit(X)
### 100 dimension
# 1 milion 206 s,
# 2 milion , 426.6510 s
# 3 milion  764.9357 s
#4 milion 996.6241 s
# 5 milion 1215.7721 s
# 6 milion 1417.5992 s = 23min
# 7 milion  2846.3684 s
#  8 milion

#### 100 dimension but do not have the parameter n_clusters
#kmeans = KMeans(n_clusters=100, random_state=0, n_jobs=-1).fit(X)
# 1 mi
# 2 mi
# 3 mi


### 200 dimension

###Evaluating Clustering Performance Using Advanced Techniques