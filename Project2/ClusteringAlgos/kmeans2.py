from __future__ import division
from index import get_cluster_group, get_incidence_matrix, get_categories
from kmeans import *
from pprint import pprint
from pca import pca
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#Read data, process and extract truth value
filename = "../cho.txt"
data = readfile(filename)
choice = "hard"
ids, data, global_truth = process_input(data)
ids = ids.astype(int)
global_truth = global_truth.astype(int)

unique_clusters = set(global_truth) #unique number of clusters
NUM_CLUSTERS = len(unique_clusters) if -1 not in unique_clusters else len(unique_clusters) - 1

#-----------------run kmeans for a specific number of times with randomly initialized centres ------------------------
if (choice == "random"):
	res = np.zeros((NUM_CLUSTERS, data.shape[1]))
	for i in range(20):
		CENTROIDS = choose_initial_centroids(data, NUM_CLUSTERS)
		clusters = process_kmeans(data, CENTROIDS, NUM_CLUSTERS)
		res = np.add(res, CENTROIDS)

	#reassign centroids with average of all the runs
	CENTROIDS = res/20
#--------------------------------------------------------------------------------------------------------
elif (choice == "hard"):
	# run kmeans with specified row ids (comment above block)
	centre_ids = [1, 83, 207, 311, 385]
	CENTROIDS = choose_initial_centroids_by_ids(centre_ids,ids, data, NUM_CLUSTERS)


# run kmeans final time
clusters = process_kmeans(data, CENTROIDS, NUM_CLUSTERS)
print(clusters)

# clusters  = KMeans(n_clusters=NUM_CLUSTERS, init='random', n_init = 20).fit_predict(data)
cluster_group = get_cluster_group(ids, clusters)
truth_group = get_cluster_group(ids, global_truth)
# pprint(cluster_group, indent=2)
kmean_matrix = get_incidence_matrix(clusters, cluster_group)
truth_matrix = get_incidence_matrix(global_truth, truth_group)
categories = get_categories(kmean_matrix, truth_matrix)

rand = (categories[0][0] + categories[1][1]) / np.sum(categories)
jaccard = categories[1][1] / (categories[1][0] + categories[0][1] + categories[1][1])

print("Rand Coeff for K-means algorithm: ", rand)
print("Jaccard Coeff for K-means algorithm: ", jaccard)

data_pca = pca(data)
plot_pca(data_pca, clusters, filename)
plot_pca(data_pca, global_truth, filename)
 

