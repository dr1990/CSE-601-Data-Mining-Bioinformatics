from pprint import pprint
from numpy.linalg import eigh
from kmeans import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def readfile(filename):
    data = pd.read_csv(filename, header=None, sep="\t")
    data = np.array(data.values)
    return data


def init_param(data):
    global attr
    global dim
    global n_data
    global clusters
    global no_of_cluster

    attr = data[:, 2:]
    dim = np.shape(data)[1] - 2
    n_data = np.shape(data)[0]
    clusters = set(np.array(data[:, 1], dtype='int'))
    no_of_cluster = len(clusters) if -1 not in clusters else len(clusters) - 1

def is_sym(a):
    return np.allclose(a, a.T, rtol=n_data, atol=n_data)


def knn(attr, sigma):
    wt = np.zeros((n_data, n_data))
    for i in range(n_data):
        for j in range(i, n_data):
            if i == j:
                wt[i][j] = 0
            else:
                wt[i][j] = np.exp(-1 * np.sum(((attr[i] - attr[j]) ** 2)) / (sigma ** 2))
                wt[j][i] = wt[i][j]

    # for i in range(n_data):
    #     max[i] = wt[i].argsort()[-k:][::-1]
    #     for j in range(n_data):
    #         if j not in max[i]:
    #             wt[i][j] = 0.0

    sym = is_sym(wt)
    print(sym)
    return wt


def get_diag(W):
    d = np.zeros((n_data, n_data))

    for i in range(n_data):
        d[i][i] = np.sum(W[i])

    return d


def get_adjacency_matrix(attr, sigma):
    w = np.zeros((n_data, n_data))
    k = 10
    w = knn(attr, sigma)

    return w

def getNbyKMatrix(k, eigen_map, eig_val):
    eig_val = np.sort(eig_val)
    reduced_data = np.empty((n_data, k))
    for i in range(k):
        reduced_data[:,i] = eigen_map[eig_val[i]]
    return reduced_data

filename = '../iyer.txt'
# filename = 'iyer.txt'
sigma = 2.0
data = readfile(filename)
init_param(data)

W = get_adjacency_matrix(attr, sigma)

D = get_diag(W)

L = np.reshape((D - W), (n_data, n_data))

sym = is_sym(L)
print(sym)

eig_val, eig_vec = eigh(L)

eigen_map = dict()

for i in range (len(eig_val)):
    eigen_map[eig_val[i]] = eig_vec[:,i]

reduced_data = getNbyKMatrix(no_of_cluster, eigen_map, eig_val)

res = np.zeros((no_of_cluster, reduced_data.shape[1]))
for i in range(20): 
    CENTROIDS = choose_initial_centroids(reduced_data, no_of_cluster)
    clusters = process_kmeans(reduced_data, CENTROIDS, no_of_cluster)
    res = np.add(res, CENTROIDS)
#reassign centroids with average of all the runs
CENTROIDS = res/20
#run k means final time
clusters = process_kmeans(reduced_data, CENTROIDS, no_of_cluster)
# clusters  = KMeans(n_clusters=no_of_cluster, init='random', n_init = 20).fit_predict(data)

global_truth = data[:,1]
ids = data[:,0]
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

# print(clusters)
# print(CENTROIDS)
