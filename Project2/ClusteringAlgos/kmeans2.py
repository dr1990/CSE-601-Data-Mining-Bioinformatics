from __future__ import division
from index import get_cluster_group, get_incidence_matrix, get_categories
from pprint import pprint
from pca import pca
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

def readfile(filename):
    data = pd.read_csv(filename, header=None, sep="\t")
    data = np.array(data.values)
    return data

def process_input(data):
	#remove outliers
	data = data[np.where(~(data[:,1] == -1))]
	ids = data[:,0]
	global_truth = data[:,1]
	data = np.delete(data, 1, 1)
	data = np.delete(data, 0, 1)
	return ids, data, global_truth

def choose_initial_centroids(data, num_clusters):
	#random initialization for centres
	return np.random.permutation(data)[:num_clusters]
	# print(CENTROIDS.shape)

def distance(point1, point2):
	#Similarity measure between two data points i.e. euclidean distance
	return math.sqrt(np.sum(np.square(point1 - point2)))

def recalculate_centroid(cluster_members, size):
	res = np.zeros(size)
	for member in cluster_members:
		# print(np.array(member))
		res = np.add(res, np.array(member))
	res = res/len(cluster_members)
	return res 

def process_kmeans(data, centroids):
	#clusters is an array such that clusters[i] = j means that ith data object belongs to j
	clusters = [0] * data.shape[0]
	#check for change in cluster membership
	previous_clusters = None
	while(previous_clusters != clusters):
		previous_clusters = clusters
		#for each datapoint
		for j, gene in enumerate(data):	
			min_dist = float("inf")
			#calculate its distance with each centroid and find the closest point
			for i,centroid in enumerate(centroids):
				dist = distance(data[j], centroid)
				if(dist < min_dist):
					min_dist = dist
					clusters[j] = i + 1
		temp_centroids = np.empty((NUM_CLUSTERS, data.shape[1]), dtype=np.float32)		
		for i, centre in enumerate(centroids):
			#associate each datapoint with a cluster
			cluster_members = []
			for j, cluster_value in enumerate(clusters):
				if(cluster_value == i+1):
					cluster_members.append(data[j])
			#check for empty clusters
			if(len(cluster_members) > 0):
				#calculate centroid for new cluster
				temp_centroids[i] = recalculate_centroid(cluster_members, data.shape[1])
			else:
				#todo for now random centroid
				temp_centroids[i] = data[np.random.choice(len(data), 1)]
		centroids = temp_centroids
	# print(previous_clusters)
	# print(clusters)
	return clusters

def plot_pca(pca, label, file):
    distinct_lable = set([])
    # label = label.tolist()
    for x in label:
        distinct_lable.add(x)

    # Color code the labels based on the distinct labels
    label_map = {}
    i = 0
    for x in distinct_lable:
        label_map.update({x: i})
        i += 1

    # Color code different disease to unique number
    color = [plt.cm.jet(float(val) / max(label_map.values())) for val in label_map.values()]

    for key, value in label_map.items():
        x = [float(k) for (t, k) in enumerate(pca[:, 0]) if label[t] == key]
        y = [float(k) for (t, k) in enumerate(pca[:, 1]) if label[t] == key]
        plt.scatter(x, y, c=color[value], label=str(key))

    plt.title("Scatter Plot for " + file + ". Algorithm: PCA")
    plt.legend()
    plt.show()

filename = "../iyer.txt"
data = readfile(filename)
ids, data, global_truth = process_input(data)
ids = ids.astype(int)
global_truth = global_truth.astype(int)

unique_clusters = set(global_truth) #unique number of clusters
NUM_CLUSTERS = len(unique_clusters) if -1 not in unique_clusters else len(unique_clusters) - 1

#run kmeans for a specific number of times with randomly initialized centres
res = np.zeros((NUM_CLUSTERS, data.shape[1]))
for i in range(20): 
	CENTROIDS = choose_initial_centroids(data, NUM_CLUSTERS)
	clusters = process_kmeans(data, CENTROIDS)
	res = np.add(res, CENTROIDS)

#reassign centroids with average of all the runs
CENTROIDS = res/20
#run k means final time
clusters = process_kmeans(data, CENTROIDS)
# print(clusters)
# print(CENTROIDS)

cluster_group = get_cluster_group(ids, clusters)
truth_group = get_cluster_group(ids, global_truth)
pprint(cluster_group, indent=2)
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