import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from pprint import pprint

data = pd.read_csv("../../iyer.txt", sep="\t", index_col=0,
                   header=None)

data = data[~(data[1] == -1)]  # removing outliers (-1 rows)
data_labels = data[1]  # ground truth values
cluster_count = data_labels.max()
gene_ids = data.index  # row numbers
del data[1]  # deleting the truth values column

geneId_map = {index: gene_id for index, gene_id in enumerate(data.index)}
# creating the clusters dictionary that initially considers every point as a cluster
clusters = {x: [x] for x in geneId_map.values()}

data = data.values  # converting Dataframe into a numpy array
# print(data)

# Normalizing input data
data = (data - data.mean()) / (data.max() - data.min())
data

# creating the distance matrix
# cdist(metric = 'euclidean') is the Euclidean distance of all datapoints with each other
dist = cdist(data, data, metric='euclidean')
# print(dist)

# setting all diagonal points to infinity (which are originally 0). Done to find min value in the matrix easily.
dist[dist[:,:] == 0] = np.inf

rowCount = dist.shape[0]

while (rowCount != cluster_count):
    # argmin() will give the flattened position of the minimum element in the dist matrix
    # unravel_index() will give the (i,j)th postion of the minimum element in the dist matrix
    x, y = np.unravel_index(dist.argmin(), dist.shape)
    dist[x, :] = np.minimum(dist[x, :], dist[y, :])
    dist[:, x] = dist[x, :]
    dist[x, x] = np.inf
    dist[y, :] = np.inf
    dist[:, y] = np.inf

    clusters[geneId_map.get(x)].extend(clusters[geneId_map.get(y)])
    del clusters[geneId_map.get(y)]
    rowCount -= 1

pprint(clusters)


