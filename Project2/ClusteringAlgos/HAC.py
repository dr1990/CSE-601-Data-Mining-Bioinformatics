import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from collections import OrderedDict
from pprint import pprint
import seaborn as sb
from matplotlib import pyplot
from ClusteringAlgos import pca
from ClusteringAlgos.index import get_cluster_group, get_incidence_matrix, get_categories

# fileName = 'iyer.txt'
# fileName= 'cho.txt'
fileName = 'new_dataset_2.txt'
data = pd.read_csv("../" + fileName, sep="\t", index_col=0, header=None)

# data = data[~(data[1] == -1)]  # removing outliers (-1 rows)
data_ground_truth = data[1]  # ground truth values
# cluster_count = data_ground_truth.max()
cluster_count = int(input('Enter number of clusters - '))
geneIds = data.index  # row numbers
del data[1]  # deleting the truth values column

# maintaining a index to geneId map
# added to keep track of row indexes in case -1 rows are deleted
geneId_map = {index: gene_id for index, gene_id in enumerate(data.index)}

# creating the clusters dictionary that initially considers every point as a cluster
clusters = {x: [x] for x in geneId_map.values()}

data = data.values  # converting Dataframe into a numpy array

# Normalizing input data
data = (data - data.mean()) / (data.max() - data.min())

# Creating the distance matrix
# cdist(metric = 'euclidean') is the Euclidean distance of all datapoints with each other
dist = cdist(data, data, metric='euclidean')

# Setting all diagonal points to infinity (which are originally 0).
# Done to find min value in the matrix easily.
dist[dist[:,:] == 0] = np.inf

rowCount = dist.shape[0]

while (rowCount != cluster_count):
    # argmin() will give the flattened position of the minimum element in the dist matrix
    # unravel_index() will give the (i,j)th postion of the minimum element in the dist matrix
    x, y = np.unravel_index(dist.argmin(), dist.shape)

    # Merge datapoints x and y into one cluster
    dist[x, :] = np.minimum(dist[x, :], dist[y, :])
    dist[:, x] = dist[x, :]
    dist[x, x] = np.inf
    dist[y, :] = np.inf
    dist[:, y] = np.inf
    print('Merging', clusters[geneId_map.get(x)], ' and ', clusters[geneId_map.get(y)])
    clusters[geneId_map.get(x)].extend(clusters[geneId_map.get(y)])
    del clusters[geneId_map.get(y)]
    rowCount -= 1

# Validation of assigned cluster using RAND and Jaccard coefficients
def validate(HAC_clusters):
    cluster_group = get_cluster_group(geneIds, list(data_ground_truth))
    cluster_group = OrderedDict(sorted(cluster_group.items()))
    incidence_matrix_gt = get_incidence_matrix(data_ground_truth, cluster_group)
    cluster_group_DBSCAN = get_cluster_group(geneIds, HAC_clusters)
    cluster_group_DBSCAN = OrderedDict(sorted(cluster_group_DBSCAN.items()))
    incidence_matrix_gmm = get_incidence_matrix(HAC_clusters, cluster_group_DBSCAN)
    categories = get_categories(incidence_matrix_gt, incidence_matrix_gmm)

    rand = (categories[0][0] + categories[1][1]) / np.sum(categories)
    jaccard = categories[1][1] / (categories[1][0] + categories[0][1] + categories[1][1])

    print("RAND: ", rand)
    print("Jaccard: ", jaccard)

pprint(clusters)

clusterMap = dict()
# Assigning cluster numbers to each data point starting from 1
# clusterMap = {geneId, clusterNumber}
for x,i in enumerate(clusters.values(), 1):
    for j in i:
        clusterMap[j] = x

clusterIds = [0] * len(clusterMap)
for k,v in clusterMap.items():
    clusterIds[k - 1] = v

validate(clusterIds)

pca_data = pca.pca(data)
pca_data_df = pd.DataFrame(pca_data, columns=['x','y'], index=geneIds)
pca_data_df['labels_GT'] = data_ground_truth
pca_data_df['labels_HAC'] = clusterIds

plot1 = sb.scatterplot(data= pca_data_df, x='x', y='y', hue='labels_GT', legend='full', palette='rainbow', marker='x')
plot1.set_title(fileName + ' Ground Truth')
pyplot.show()

plot2 = sb.scatterplot(data= pca_data_df, x = 'x', y= 'y', hue='labels_HAC', legend='full', palette='rainbow', marker='x')
plot2.set_title('Clusters formed using HAC on ' + fileName)
pyplot.show()