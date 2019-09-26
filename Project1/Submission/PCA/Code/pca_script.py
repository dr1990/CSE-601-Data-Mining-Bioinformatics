import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt


# Transform file data into numpy array for processing
def convert_to_array(file):
    f = open(file, 'r')
    lines = f.read().splitlines()
    lines = [[val for val in l.split('\t')] for l in lines]
    lines = np.array(lines)
    f.close()
    return lines


# Calculating PCA
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


# convert string data to number format
def convert_to_num(data):
    s = np.array(data).shape
    out = np.zeros(s, dtype=float)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            out[i][j] = float(data[i][j])
    return out


def plot_pca(pca, label, file):
    distinct_lable = set([])

    for x in label:
        distinct_lable.add(x[0])

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


def main():
    file = "pca_demo.txt"
    lines = convert_to_array(file)

    # Separate features and label
    c = lines.shape[1]

    data = lines[:, :-1]
    l = lines[:, c - 1:c]
    label = np.array(l)

    data = convert_to_num(data)

    # Run PCA algorithm on the data
    p = pca(data)

    # Scatter Plot of the of data in reduced dimension
    plot_pca(p, label, file)


if __name__ == '__main__':
    main()
