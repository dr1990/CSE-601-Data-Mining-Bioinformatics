import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from collections import deque, OrderedDict
from pprint import pprint
import seaborn as sb
from matplotlib import pyplot
from ClusteringAlgos.index import get_cluster_group, get_incidence_matrix, get_categories

from ClusteringAlgos import pca

# fileName = 'iyer.txt'
fileName = 'cho.txt'
data = pd.read_csv("../../" + fileName, sep="\t", index_col=0, header=None)

data_ground_truth = data[1]  # ground truth values
del data[1]  # deleting the truth values column
geneIds = data.index

# Cluster assignment for every datapoint (initially all datapoints are assigned cluster -1 i.e. noise)
clusterMap = {x: -1 for x in data.index}

data = data.values  # converting Dataframe into a numpy array

# ***Need to ask professor if normalization is required
# Normalizing input data
# data = (data - data.mean()) / (data.max() - data.min())
# print(data)

# Appending a row at the 0th index
data = np.vstack((np.zeros((1, data.shape[1])), data))

visited = set()

# DBSCAN algorithm
# Input:
#  eps - Epsilon distance
#  minPts - Min points parameter
def DBSCAN(eps, minPts):
    C = 0  # cluster number
    for i in geneIds:
        if (i not in visited):
            visited.add(i)
            neighbourPoints = regionQuery(i, eps)
            if len(neighbourPoints) < minPts:
                continue
            C += 1
            expandCluster(i, neighbourPoints, C, eps, minPts)


# Method that expands the cluster to neighbouring points
# P - datapoint id
# neighbourPoints - Set containing ids of datapoints within 'eps' distance of point P
# eps - Epsilon distance
# C - current cluster number
# minPts - Minimum points in the cluster
def expandCluster(P, neighbourPoints, C, eps, minPts):
    clusterMap[P] = C
    while len(neighbourPoints) != 0:
        nPoint = neighbourPoints.pop()
        if nPoint not in visited:
            visited.add(nPoint)
            npNeighbours = regionQuery(nPoint, eps)
            if len(npNeighbours) >= minPts:
                neighbourPoints.extend(npNeighbours)
        if clusterMap[nPoint] == -1:
            clusterMap[nPoint] = C

# Method to find all the points within 'eps' distance of datapoint 'P'
# P - datapoint under consideration
# - Epsilon distance
def regionQuery(P, eps):
    neighbourPoints = deque()

    # Calculating the Euclidean distance datapoint P w.r.t. all datapoints
    distVector = cdist(data, data[P].reshape(1, -1), metric='euclidean')
    for i in geneIds:
        if (distVector[i] <= eps):
            neighbourPoints.append(i)

    return neighbourPoints

# Validation of assigned cluster using RAND and Jaccard coefficients
def validate(DBSCAN_clusters):
    cluster_group = get_cluster_group(geneIds, list(data_ground_truth))
    cluster_group = OrderedDict(sorted(cluster_group.items()))
    incidence_matrix_gt = get_incidence_matrix(data_ground_truth, cluster_group)
    cluster_group_DBSCAN = get_cluster_group(geneIds, DBSCAN_clusters)
    cluster_group_DBSCAN = OrderedDict(sorted(cluster_group_DBSCAN.items()))
    incidence_matrix_gmm = get_incidence_matrix(DBSCAN_clusters, cluster_group_DBSCAN)
    categories = get_categories(incidence_matrix_gt, incidence_matrix_gmm)

    rand = (categories[0][0] + categories[1][1]) / np.sum(categories)
    jaccard = categories[1][1] / (categories[1][0] + categories[0][1] + categories[1][1])

    print("Rand: ", rand)
    print("Jaccard: ", jaccard)


DBSCAN(1.03, 4)
pprint(clusterMap)

validate(list(clusterMap.values()))


# Plotting data (ground truth & realized clusters)
pca_data = pca.pca(data[1:,:])
pca_data_df = pd.DataFrame(pca_data, columns=['x','y'], index=geneIds)
pca_data_df['labels_GT'] = data_ground_truth
pca_data_df['labels_DBSCAN'] = list(clusterMap.values())

plot1 = sb.scatterplot(data= pca_data_df, x='x', y='y', hue='labels_GT', legend='full', palette='Accent', marker='x')
plot1.set_title(fileName + ' Ground Truth')
pyplot.show()

plot2 = sb.scatterplot(data= pca_data_df, x = 'x', y= 'y', hue='labels_DBSCAN', legend='full', palette='Accent', marker='x')
plot2.set_title('Clusters formed using DBSCAN on ' + fileName)
pyplot.show()


