import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from pprint import pprint
import seaborn as sb
from matplotlib import pyplot
from ClusteringAlgos import pca

fileName = 'iyer.txt'
# fileName= 'cho.txt'
data = pd.read_csv("../../" + fileName, sep="\t", index_col=0, header=None)

# data = data[~(data[1] == -1)]  # removing outliers (-1 rows)
data_labels = data[1]  # ground truth values
cluster_count = data_labels.max()
geneIds = data.index  # row numbers
del data[1]  # deleting the truth values column

geneId_map = {index: gene_id for index, gene_id in enumerate(data.index)}
# creating the clusters dictionary that initially considers every point as a cluster
clusters = {x: [x] for x in geneId_map.values()}

data = data.values  # converting Dataframe into a numpy array
# print(data)

# Normalizing input data
data = (data - data.mean()) / (data.max() - data.min())
# print(data)

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

clusterMap = dict()

# Assigning cluster numbers to each data point starting from 1
for x,i in enumerate(clusters.values(), 1):
    for j in i:
        clusterMap[j] = x


clusterIds = [0] * len(clusterMap)

for k,v in clusterMap.items():
    clusterIds[k - 1] = v

pca_data = pca.pca(data)
pca_data_df = pd.DataFrame(pca_data, columns=['x','y'], index=geneIds)
pca_data_df['labels_GT'] = data_labels
pca_data_df['labels_HAC'] = clusterIds

plot1 = sb.scatterplot(data= pca_data_df, x='x', y='y', hue='labels_GT', legend='full', palette='Accent', marker='x')
plot1.set_title(fileName + ' Ground Truth')
pyplot.show()

plot2 = sb.scatterplot(data= pca_data_df, x = 'x', y= 'y', hue='labels_HAC', legend='full', palette='prism', marker='x')
plot2.set_title('Clusters formed using HAC on ' + fileName)
pyplot.show()

