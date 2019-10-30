from index import get_cluster_group, get_incidence_matrix, get_categories
from kmeans import *
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np



#Read data, process and extract truth value
filename = "new_dataset_1.txt"
data = readfile(filename)
ids, data, global_truth = process_input(data)
ids = ids.astype(int)
global_truth = global_truth.astype(int)

#------------parameters----------------------------------------------------------------------------
choice = "random"
unique_clusters = set(global_truth) #unique number of clusters
# NUM_CLUSTERS = len(unique_clusters) if -1 not in unique_clusters else len(unique_clusters) - 1
NUM_CLUSTERS = 3
NUM_iters = 10
centre_ids = [3, 5, 7]
#-----------------run kmeans for a specific number of times with randomly initialized centres ------------------------
if (choice == "random"):
	res = np.zeros((NUM_CLUSTERS, data.shape[1]))
	for i in range(20):
		CENTROIDS = choose_initial_centroids(data, NUM_CLUSTERS)
		clusters, res_centres = process_kmeans(data, CENTROIDS, NUM_CLUSTERS, NUM_iters)
		res = np.add(res, res_centres)
	#reassign centroids with average of all the runs
	CENTROIDS = res/20
#--------------------------------------------------------------------------------------------------------
elif (choice == "hard"):
	CENTROIDS = choose_initial_centroids_by_ids(centre_ids, data)


# run kmeans final time
clusters, res_centres = process_kmeans(data, CENTROIDS, NUM_CLUSTERS, NUM_iters)

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

data_pca = PCA(n_components=2).fit_transform(data)
plot_pca(data_pca, clusters, filename)
plot_pca(data_pca, global_truth, filename)
 

