import pandas as pd
import numpy as np


def readfile(filename):
    data = pd.read_csv(filename, header=None, sep="\t")
    data = np.array(data.values)
    return data


def write(data):
    f = open('data.txt', 'a+')
    shape = np.shape(data)
    for x in range(shape[0]):
        for y in range(shape[0]):
            f.write(str(data[x][y]) + " ")
        f.write("\n")
    f.close()


def get_incidence_matrix(ground_truth):
    n = len(ground_truth)
    incidence_matrix = np.zeros((n, n), dtype='int')

    for k, v in cluster_group.items():
        for i, row in enumerate(v):
            for col in range(i, len(v)):
                incidence_matrix[row - 1][v[col] - 1] = 1
                incidence_matrix[v[col] - 1][row - 1] = 1

    # write(incidence_matrix)
    return incidence_matrix


cluster_group = dict()


def get_cluster_group(id, ground_truth):
    for i in range(len(id)):
        if ground_truth[i] in cluster_group.keys():
            values = cluster_group[ground_truth[i]]
        else:
            values = list()
        values.append(id[i])
        cluster_group[ground_truth[i]] = values


def get_categories(im_a, im_b):
    categories = [[0, 0], [0, 0]]

    (m, n) = np.shape(im_a)

    for i in range(m):
        for j in range(n):
            x = im_a[i][j]
            y = im_b[i][j]
            categories[x][y] += 1

            # if x == y and x == 1:
            #     categories[1][1] += 1
            # elif x == y and x == 0:
            #     categories[0][0] += 1
            # elif x != y and x == 1:
            #     categories[1][0] += 1
            # elif x != y and x == 0:
            #     categories[0][1] += 1
    return categories


# filename = '../cho.txt'
filename = '../iyer.txt'
data = readfile(filename)
id = np.array(data[:, 0], dtype='int')
ground_truth = np.array(data[:, 1], dtype='int')
get_cluster_group(id, ground_truth)
incidence_matrix_gt = get_incidence_matrix(ground_truth)
categories = get_categories(incidence_matrix_gt, incidence_matrix_gt)

rand = (categories[0][0] + categories[1][1]) / np.sum(categories)
jaccard = categories[1][1] / (categories[1][0] + categories[0][1] + categories[1][1])

print("Rand: ", rand)
print("Jaccard: ", jaccard)