from pprint import pprint
from index import get_cluster_group, get_incidence_matrix, get_categories
from numpy.linalg import eigh
from kmeans import choose_initial_centroids_by_ids, plot_pca
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def readfile(filename):
    data = pd.read_csv(filename, header=None, sep="\t")
    data = np.array(data.values)
    return data


def init_param(data, num_clusters):
    global attr
    global dim
    global n_data
    global clusters
    global no_of_cluster

    attr = data[:, 2:]
    dim = np.shape(data)[1] - 2
    n_data = np.shape(data)[0]
    clusters = set(np.array(data[:, 1], dtype='int'))
    if num_clusters:
        no_of_cluster = num_clusters
    else:
        no_of_cluster = len(clusters) if -1 not in clusters else len(clusters) - 1


def fully_connected_graph(attr, sigma):
    wt = np.zeros((n_data, n_data))
    for i in range(n_data):
        for j in range(i, n_data):
            wt[i][j] = np.exp(-1 * np.sum(((attr[i] - attr[j]) ** 2)) / (sigma ** 2))
            wt[j][i] = wt[i][j]
    return wt


def get_diag(W):
    d = np.zeros((n_data, n_data))

    for i in range(n_data):
        d[i][i] = np.sum(W[i])
    return d

def getNbyKMatrix(eig_vec, eig_val):
    k = max_eigen_gap_num(eig_val)
    print(k)
    eig_val = eig_val[np.argsort(eig_val)]
    eig_vec = eig_vec[:, np.argsort(eig_val)]
    reduced_data = eig_vec[:, 1:k]
    return reduced_data


def max_eigen_gap_num(eigen_val):
    max_gap = -999999
    k = 0
    for i in range(len(eigen_val) - 1):
        diff = abs(eigen_val[i] - eig_val[i + 1])
        if (diff > max_gap):
            max_gap = diff
            k = i + 2  # index at zero
    return k


filename = 'new_dataset_1.txt'
# filename = 'cho.txt'

sigma = 2.75
num_clusters = 3
choice = "random"
centers = [10, 25, 44]
max_iters = 100

data = readfile(filename)
init_param(data, num_clusters)

W = fully_connected_graph(attr, sigma)

D = get_diag(W)

L = np.reshape((D - W), (n_data, n_data))

eig_val, eig_vec = eigh(L)

reduced_data = getNbyKMatrix(eig_vec, eig_val)
if choice == "hard":
    init = choose_initial_centroids_by_ids(centers, reduced_data)
else:
    init = "random"
clusters = KMeans(n_clusters=no_of_cluster, init=init, n_init=20, max_iter=max_iters).fit_predict(reduced_data)
global_truth = data[:, 1]
ids = data[:, 0]
cluster_group = get_cluster_group(ids, clusters)
truth_group = get_cluster_group(ids, global_truth)
# pprint(cluster_group, indent=2)
kmean_matrix = get_incidence_matrix(clusters, cluster_group)
truth_matrix = get_incidence_matrix(global_truth, truth_group)
categories = get_categories(kmean_matrix, truth_matrix)

rand = (categories[0][0] + categories[1][1]) / np.sum(categories)
jaccard = categories[1][1] / (categories[1][0] + categories[0][1] + categories[1][1])

print("Rand Coeff for Spectral algorithm: ", rand)
print("Jaccard Coeff for Spectral algorithm: ", jaccard)
data_pca = PCA(n_components=2).fit_transform(attr)
plot_pca(data_pca, clusters, filename)

# # print(clusters)
# # print(CENTROIDS)
