import numpy as np

# Creates the incidence matrix indicating the similarity of cluster assignments to each datapoint
def get_incidence_matrix(ground_truth, cluster_group):
    n = len(ground_truth)
    incidence_matrix = np.zeros((n, n), dtype='int')

    for k, v in cluster_group.items():
        for i, row in enumerate(v):
            for col in range(i, len(v)):
                incidence_matrix[row - 1][v[col] - 1] = 1
                incidence_matrix[v[col] - 1][row - 1] = 1

    return incidence_matrix


# Creates a map of cluster numbers (key) to gene_ids (values)
# id - gene ids in the data set
# ground_truth - cluster numbers
def get_cluster_group(id, ground_truth):
    cluster_group = dict()
    for i in range(len(id)):
        if ground_truth[i] in cluster_group.keys():
            values = cluster_group[ground_truth[i]]
        else:
            values = list()
        values.append(i)
        cluster_group[ground_truth[i]] = values
    return cluster_group

# Takes 2 maps (cluster number -> gene_ids) and computes a 2x2 matrix
# to indicate TP, FP, TN, FN
def get_categories(im_a, im_b):
    categories = [[0, 0], [0, 0]]

    (m, n) = np.shape(im_a)

    for i in range(m):
        for j in range(n):
            x = im_a[i][j]
            y = im_b[i][j]
            categories[x][y] += 1
    return categories