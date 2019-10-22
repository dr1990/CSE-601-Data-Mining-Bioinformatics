import numpy as np
import pandas as pd
from numpy.linalg import eigh
import matplotlib.pyplot as plt


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

    print()


def is_sym(a):
    return np.allclose(a, a.T, rtol=n_data, atol=n_data)


def knn(attr, k, sigma):
    wt = np.zeros((n_data, n_data))
    max = np.zeros((n_data, k))
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


def get_adjacency_matrix(attr):
    w = np.zeros((n_data, n_data))
    k = 10
    sigma = 4
    w = knn(attr, k, sigma)

    return w


filename = 'cho.txt'
# filename = 'iyer.txt'
data = readfile(filename)
init_param(data)

W = get_adjacency_matrix(attr)

D = get_diag(W)

L = np.reshape((D - W), (n_data, n_data))

sym = is_sym(L)
print(sym)

eig_val, eig_vec = eigh(L)

# val = sorted(np.reshape(eig_val, (-1, 1)))
val = np.argpartition(eig_val, 5)

fig = plt.figure(figsize=[12, 6])
ax = fig.gca
plt.subplot(221)
plt.plot(eig_val)
plt.subplot(222)
plt.plot(eig_vec[:, 0])
plt.subplot(223)
plt.plot(eig_vec[:, 1])
plt.subplot(224)
plt.plot(eig_vec[:, 2])

print()
