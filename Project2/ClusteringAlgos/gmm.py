import numpy as np
import pandas as pd
from numpy import linalg as la
from scipy.stats import multivariate_normal
from index import get_cluster_group, get_incidence_matrix, get_categories
from pca import pca
import collections
import json
import numpy

import matplotlib.pyplot as plt


def plot_pca(pca, label, file):
    distinct_lable = set([])
    # label = label.tolist()
    for x in label:
        distinct_lable.add(x)

    # Color code the labels based on the distinct labels
    label_map = {}
    i = 0
    for x in distinct_lable:
        label_map.update({x: i})
        i += 1

    # Color code different disease to unique number
    demon = max(label_map.values())
    if demon == 0:
        demon = smoothing_value
    color = [plt.cm.jet(float(val) / demon) for val in label_map.values()]

    for key, value in label_map.items():
        x = [float(k) for (t, k) in enumerate(pca[:, 0]) if label[t] == key]
        y = [float(k) for (t, k) in enumerate(pca[:, 1]) if label[t] == key]
        plt.scatter(x, y, c=color[value], label=str(key))

    plt.title("Scatter Plot for " + file + ". Algorithm: GMM")
    plt.legend()
    plt.show()


def readfile(filename):
    d = pd.read_csv(filename, header=None, sep="\t")
    d = np.array(d.values)
    return d


def EM(mu, cov, pi):
    r = np.zeros((n_data, no_of_cluster))
    ll = 0
    prev_ll = -9999999999

    for _ in range(niter):

        # E-Step
        pdf = np.zeros((n_data, no_of_cluster))
        for k in range(no_of_cluster):
            t = multivariate_normal(mu[k], cov[k], allow_singular=True).pdf(attr)
            t = pi[k] * np.reshape(t, (-1, 1))
            pdf[:, k] = t.T

        sm = 0
        for k in range(no_of_cluster):
            sm += pdf[:, k] * pi[k]

        for k in range(no_of_cluster):
            r[:, k] = (pdf[:, k] * pi[k]) / (sm + smoothing_value)

        pi = np.sum(r, axis=0) / attr.shape[0]

        N_K = np.sum(r, axis=0)

        mu = np.dot(r.T, attr)
        for t in range(no_of_cluster):
            mu[t] = mu[t] / (N_K[t] + smoothing_value)

        # covariance
        for k in range(no_of_cluster):
            diff = (attr - mu[k])
            sm = np.dot(r[:, k] * diff.T, diff)
            cov[k] = sm / (N_K[k] + smoothing_value)

        # ll = 0
        # for k in range(no_of_cluster):
        #     ll += np.log(np.sum(pi[k] * (multivariate_normal(mu[k], cov[k], allow_singular=True).pdf(attr))))

        # ll = np.sum(np.sum(r))

        # if (ll - prev_ll) < conv_threshold:
        #     break

        # print(_, ll, prev_ll, (ll - prev_ll))
        # prev_ll = ll

    print("******** Mu ********")
    print(mu)
    print("******** Cov ********")
    print(cov)
    print("******** pi ********")
    print(pi)
    print("****************")
    return r, mu, cov, pi


def coef(gmm_truth):
    id = np.array(data[:, 0], dtype='int')
    ground_truth = np.array(data[:, 1], dtype='int')
    cluster_group = get_cluster_group(id, ground_truth)
    cluster_group = collections.OrderedDict(sorted(cluster_group.items()))
    incidence_matrix_gt = get_incidence_matrix(ground_truth, cluster_group)

    cluster_group_gmm = get_cluster_group(id, gmm_truth)
    cluster_group_gmm = collections.OrderedDict(sorted(cluster_group_gmm.items()))
    incidence_matrix_gmm = get_incidence_matrix(gmm_truth, cluster_group_gmm)

    categories = get_categories(incidence_matrix_gt, incidence_matrix_gmm)

    rand = (categories[0][0] + categories[1][1]) / np.sum(categories)
    jaccard = categories[1][1] / (categories[1][0] + categories[0][1] + categories[1][1])

    print("Rand: ", rand)
    print("Jaccard: ", jaccard)


def read_input():
    global data
    global niter
    global attr
    global dim
    global n_data
    global clusters
    global no_of_cluster
    global pi
    global mu
    global cov
    global smoothing_value
    global conv_threshold
    global max_iter

    # mu = numpy.array(json.loads(input("Enter mean: ")))
    # cov = numpy.array(json.loads(input("Enter cov: ")))
    # pi = numpy.array(json.loads(input("Enter pi: ")))
    # no_of_cluster = int(input("Enter no. of cluster: "))
    # niter = int(input("Enter max iteration: "))
    # conv_threshold = float(input("Enter convergence threshold: "))
    # smoothing_value = float(input("Enter smoothing value: "))

    # smoothing_value = 0
    # initialize parameters

    # # mu = np.array([[0, 0], [0, 4], [4, 4]])
    # mu = np.array([[0, 0], [1, 1]])
    # # cov = np.array([[[1, 1], [1, 1]], [[2, 2], [2, 2]], [[3, 3], [3, 3]]])
    # cov = np.array([[[1, 1], [1, 1]], [[2, 2], [2, 2]]])
    # pi = np.array([0.1, 0.1, 0.1, 0.1, 0.1,0.1, 0.1, 0.1, 0.1, 0.1])

    max_iter = 1000
    attr = data[:, 2:]
    smoothing_value = 0.000000001
    conv_threshold = 0.000000001
    niter = 100
    clusters = set(np.array(data[:, 1], dtype='int'))
    no_of_cluster = len(clusters) if -1 not in clusters else len(clusters) - 1

    dim = np.shape(data)[1] - 2
    n_data = np.shape(data)[0]

    np.random.seed(4000)

    rand_data = np.random.choice(n_data, no_of_cluster, replace=False)
    mu = attr[rand_data]

    pi = np.ones(no_of_cluster, dtype='float64') / no_of_cluster

    cov = np.zeros((no_of_cluster, dim, dim), dtype='float64')
    for i in range(no_of_cluster):
        # cov[i] = i + 1
        np.fill_diagonal(cov[i], 1)


# filename = 'cho.txt'
# filename = 'iyer.txt'
filename = 'GMM_tab_seperated.txt'

data = readfile(filename)
read_input()

r, mu, cov, pi = EM(mu, cov, pi)

lab = np.argmax(r, axis=1), (-1, 1)
coef(lab[0])

if dim <= 2:
    plot_pca(attr, lab[0], filename)
else:
    data_pca = pca(data[:, 2:])
    plot_pca(data_pca, lab[0], filename)
