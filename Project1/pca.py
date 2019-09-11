import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt


def getvalue(value):
    # Remove "\n" from the last
    if isinstance(value, str):
        return value[:-1]
    else:
        return value


def convert_to_array(file):
    # with open(file, 'r') as f:
    #     lines = [[getvalue(val) for val in l.split('\t')] for l in f]

    f = open(file, 'r')
    lines = f.read().splitlines()
    lines = [[val for val in l.split('\t')] for l in lines]
    lines = np.array(lines)
    f.close()
    return lines


def pca(data):
    # print(data)
    mean = np.mean(data, axis=0)
    # print(data - mean)

    adj_data = data - mean
    cov = np.cov(adj_data.T)
    # print(cov)

    eig_val, eig_vec = la.eig(cov)
    print(eig_val)
    print(eig_vec.shape)

    # eig_val = eig_val[:2]

    top_two_ind = eig_val.argsort()[-2:][::-1]
    top_eig_vec = eig_vec[:, top_two_ind]
    # print("#########")
    print(top_eig_vec)

    pca = np.dot(data, top_eig_vec)
    print("#########")
    print(pca)


def convert_to_num(data):
    s = np.array(data).shape
    out = np.zeros(s, dtype=float)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            out[i][j] = float(data[i][j])
    return out


def main():
    file = "pca_b.txt"
    lines = convert_to_array(file)

    r = lines.shape[0]
    c = lines.shape[1]

    data = lines[:, :-1]
    label = lines[:, c - 1:c]

    data = convert_to_num(data)
    pca(data)
    plt.scatter(data[0], data[1])


if __name__ == '__main__':
    main()
