import numpy as np
import pandas as pd
from numpy import linalg as la
from scipy.stats import multivariate_normal

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
    color = [plt.cm.jet(float(val) / max(label_map.values())) for val in label_map.values()]

    for key, value in label_map.items():
        x = [float(k) for (t, k) in enumerate(pca[:, 0]) if label[t] == key]
        y = [float(k) for (t, k) in enumerate(pca[:, 1]) if label[t] == key]
        plt.scatter(x, y, c=color[value], label=str(key))

    plt.title("Scatter Plot for " + file + ". Algorithm: PCA")
    plt.legend()
    plt.show()


def pca(data):
    # Get mean along each dimension
    mean = np.mean(data, axis=0)

    # Adjust the data around mean
    adj_data = data - mean

    # find co-variance matrix
    cov = np.cov(adj_data.T)

    # Get eigen-value and eigen-vector of the co-variance matrix
    eig_val, eig_vec = la.eig(cov)

    # Pick top-two eigen-value and corresponding eigen-vector
    top_ind = eig_val.argsort()[-2:][::-1]
    top_eig_vec = eig_vec[:, top_ind]

    p = np.zeros([data.shape[0], top_eig_vec.shape[1]])

    p = np.dot(top_eig_vec.T, adj_data.T).T
    return p


def init_param(data):
    global niter
    global attr
    global dim
    global n_data
    global clusters
    global no_of_cluster
    global pi
    global mu
    global cov

    niter = 55
    attr = data[:, 2:]
    dim = np.shape(data)[1] - 2
    n_data = np.shape(data)[0]
    clusters = set(np.array(data[:, 1], dtype='int'))
    no_of_cluster = len(clusters) if -1 not in clusters else len(clusters) - 1

    # initialize parameters
    np.random.seed(4000)
    rand_data = np.random.choice(n_data, no_of_cluster, replace=False)
    mu = attr[rand_data]
    pi = np.ones(no_of_cluster, dtype='float') / no_of_cluster

    cov = np.zeros((no_of_cluster, dim, dim), dtype='float')
    for i in range(no_of_cluster):
        np.fill_diagonal(cov[i], 4)
    print()


def readfile(filename):
    data = pd.read_csv(filename, header=None, sep="\t")
    data = np.array(data.values)

    ret = []
    for i in range(data.shape[0]):
        if data[i, 1] != -1:
            ret.append(data[i])
        else:
            print()
    out = np.array(ret)
    return out


def EM1(mu, cov, pi):
    r = np.zeros((n_data, no_of_cluster))

    for _ in range(niter):
        # E-Step

        pdf = np.zeros((n_data, no_of_cluster))
        for k in range(no_of_cluster):
            rvk = multivariate_normal(mu[k], cov[k], allow_singular=True)
            t = rvk.pdf(attr)
            t = pi[k] * np.reshape(t, (-1, 1))
            pdf[:, k] = t.T

        sm = 0
        for k in range(no_of_cluster):
            sm += pdf[:, k] * pi[k]

        for k in range(no_of_cluster):
            r[:, k] = (pdf[:, k] * pi[k]) / sm

        # M-Step

        pi = np.sum(r, axis=0) / attr.shape[0]
        # print(pi)

        N_K = np.sum(r, axis=0)

        mu = np.dot(r.T, attr)
        for t in range(no_of_cluster):
            mu[t] = mu[t] / N_K[t]

        # covariance
        for k in range(no_of_cluster):
            diff = (attr - mu[k])
            sm = np.dot(r[:, k] * diff.T, diff)
            cov[k] = sm / N_K[k]

    return r, mu, cov, pi


def EM(mu, cov, pi):
    r = np.zeros((n_data, no_of_cluster))
    for _ in range(niter):

        # E-Step
        for i in range(n_data):
            denom = 0
            mv = np.zeros(no_of_cluster)
            for k in range(no_of_cluster):
                mv[k] = multivariate_normal.pdf(attr[i], mean=mu[k], cov=cov[k], allow_singular=True)
            for k in range(no_of_cluster):
                r[i][k] = pi[k] * mv[k] / np.sum(mv)

        # M-Step

        N_K = np.sum(r, axis=0, dtype=np.float)
        new_mu = np.dot(r.T, attr)
        for t in range(no_of_cluster):
            new_mu[t] = new_mu[t] / N_K[t]

        mu = new_mu

        new_cov = np.zeros(shape=(no_of_cluster, dim, dim), dtype='float')
        for k in range(no_of_cluster):
            v = np.zeros(shape=(dim, dim), dtype='float')
            for j in range(len(attr)):
                diff = np.reshape((attr[j] - mu[k]), (-1, 1))
                v += r[j][k] * np.dot(diff, diff.T)
            new_cov[k] = v / N_K[k]

            pi[k] = N_K[k] / n_data
        cov = new_cov

    return r, mu, cov, pi


# filename = 'cho.txt'
filename = 'iyer.txt'
data = readfile(filename)

init_param(data)
# r, mu, cov, pi = EM(mu, cov, pi)
r, mu, cov, pi = EM1(mu, cov, pi)

lab = np.argmax(r, axis=1), (-1, 1)

data_pca = pca(data[:, 2:])

orig_label = data[:, 1]

plot_pca(data_pca, lab[0], filename)
plot_pca(data_pca, orig_label, filename)
