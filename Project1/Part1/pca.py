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
    # print(mean)
    # print("########")
    # print(data - mean)

    adj_data = data - mean
    cov = np.cov(adj_data.T)

    eig_val, eig_vec = la.eig(cov)

    top_ind = eig_val.argsort()[-2:][::-1]
    top_eig_vec = eig_vec[:, top_ind]

    p = np.zeros([data.shape[0], top_eig_vec.shape[1]])

    p = np.dot(top_eig_vec.T, adj_data.T).T
    return p


def convert_to_num(data):
    s = np.array(data).shape
    out = np.zeros(s, dtype=float)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            out[i][j] = float(data[i][j])
    return out


def plot_pca(pca, label):
    np.random.seed(252354234)

    distinct_lable = set([])

    for x in label:
        distinct_lable.add(x[0])

    label_map = {}
    i = 0
    for x in distinct_lable:
        label_map.update({x: i})
        i += 1

    colors = [plt.cm.jet(float(i) / max(label_map.values())) for i in label_map.values()]

    p = np.append(pca, label, axis=1)
    for key, value in label_map.items():
        x = [float(k) for (t, k) in enumerate(p[:, 0]) if p[t, 2] == key]
        y = [float(k) for (t, k) in enumerate(p[:, 1]) if p[t, 2] == key]
        plt.scatter(x, y, c=colors[value], label=str(key))

    plt.title(" Scatter Plot")
    plt.legend()
    plt.show()


def main():
    file = "pca_c.txt"
    lines = convert_to_array(file)

    r = lines.shape[0]
    c = lines.shape[1]

    data = lines[:, :-1]
    l = lines[:, c - 1:c]
    label = np.array(l)

    # print(label)

    data = convert_to_num(data)
    p = pca(data)

    # p = np.append(out, label, axis=1)
    # print(p)
    plot_pca(p, label)


if __name__ == '__main__':
    main()
