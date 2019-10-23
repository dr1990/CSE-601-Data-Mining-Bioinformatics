import pandas as pd
import numpy as np
import math
from index import get_categories, get_incidence_matrix


NUM_CLUSTERS = 10
CENTROIDS = []
def readfile(filename):
    data = pd.read_csv(filename, header=None, sep="\t")
    data = np.array(data.values)
    return data

def process_input(data):
	#remove outliers
	data = data[np.where(~(data[:,1] == -1))]
	global_truth = data[:,1]
	data = np.delete(data, 1, 1)
	data = np.delete(data, 0, 1)
	return data, global_truth

def distance(point1, point2):
	#Similarity measure between two data points i.e. euclidean distance
	return math.sqrt(np.sum(np.square(point1 - point2))) 

def choose_initial_centroids(data):
	global CENTROIDS
	# data_temp = data
	# CENTROIDS.append(np.mean(data, 0))
	# while(len(CENTROIDS) < NUM_CLUSTERS):
	# 	for centroid in CENTROIDS:
	# 		min_dist = 9999999
	# 		min_gene = None
	# 		for index, gene in enumerate(data_temp):
	# 			dist = cal_mean_square_distance(gene, centroid)
	# 			if(min_dist > dist and dist > 0 ):
	# 				min_dist = dist
	# 				min_gene = gene
	# 				np.delete(data_temp, index, 0)
	# 	CENTROIDS.append(min_gene)
	CENTROIDS = np.random.permutation(data)[:NUM_CLUSTERS]
	# print(CENTROIDS.shape)

	# print(CENTROIDS)
def recalculate_centroid(cluster_members, size):
	res = np.zeros(size)
	for member in cluster_members:
		# print(np.array(member))
		res = np.add(res, np.array(member))
	res = res/len(cluster_members)
	return res

def process_kmeans(data):
	global CENTROIDS
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
			for i,centroid in enumerate(CENTROIDS):
				dist = distance(data[j], centroid)
				if(dist < min_dist):
					min_dist = dist
					clusters[j] = i + 1
		temp_centroids = np.empty((NUM_CLUSTERS, data.shape[1]), dtype=np.float32)		
		for i, centroids in enumerate(CENTROIDS):
			#associate each datapoint with a centre
			cluster_members = []
			for j, cluster_value in enumerate(clusters):
				if(cluster_value == i):
					cluster_members.append(data[j])
			if(len(cluster_members) > 0):
				#check for empty clusters
				temp_centroids[i] = recalculate_centroid(cluster_members, data.shape[1])
			else:
				choice = (data[np.random.choice(len(data), 1)])
				if choice not in temp_centroids:
					temp_centroids[i] = choice
		CENTROIDS = temp_centroids
	return clusters
	# print(clusters)
	# print(previous_clusters)
def get_cluster_group(id, clusters):
    for i in range(len(id)):
        if clusters[i] in cluster_group.keys():
            values = cluster_group[clusters[i]]
        else:
            values = list()
        values.append(id[i])
        cluster_group[clusters[i]] = values


filename = "../iyer.txt"
data = readfile(filename)
# ground_truth = np.array(data[:, 1], dtype='int')
data, global_truth = process_input(data)
choose_initial_centroids(data)
res = np.zeros(CENTROIDS.shape)

#run kmeans for a specific number of times with randomly initialized centres
for i in range(10): 
	clusters = process_kmeans(data)
	res = np.add(res, CENTROIDS)

#reassign centroids with avergae of all the runs
CENTROIDS = res/10

#run k means final time
clusters = process_kmeans(data)
print(clusters)
# cluster_group = dict()
# id = [x for x in range(len(data))]
# get_cluster_group(id, clusters)
# print(cluster_group)

# incidence_matrix_global_truth = get_incidence_matrix(global_truth)
# incidence_matrix_kmean = get_incidence_matrix(clusters)

# categories = get_categories(incidence_matrix_global_truth, incidence_matrix_kmean)
# rand = (categories[0][0] + categories[1][1]) / np.sum(categories)
# jaccard = categories[1][1] / (categories[1][0] + categories[0][1] + categories[1][1])

# print("Rand: ", rand)
# print("Jaccard: ", jaccard)