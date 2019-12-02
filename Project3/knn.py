import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def readfile(filename):
    data = pd.read_csv(filename, header=None, sep="\t")
    data = np.array(data.values)

    str_ind = list()
        
    for i in range(len(data[0])):
        if isinstance(data[0][i], str):
            str_ind.append(i)

    for i in str_ind:
        uniq = np.unique(data[:, i])
        uniq_vals = list(range(len(uniq)))
        str_val_dict = dict(zip(uniq, uniq_vals))
        for j in range(len(data[:, i])):
            data[j][i] = str_val_dict.get(data[j][i])

    return data


def get_confusion_matrix(train_label, pred):
    shape = np.shape(train_label)
    row = shape[0]

    cm = np.zeros((2, 2))

    for i in range(row):
        x = int(train_label[i])
        y = int(pred[i])
        cm[x][y] += 1

    return cm


def knn(k, train_data, test_data, train_label, test_label):
    shape = np.shape(test_data)
    row = shape[0]

    knn_matrix = np.zeros((row, k))
    pred = np.zeros((row, 1))

    for i, x in enumerate(test_data):
        x = np.reshape(x, (-1, 1)).T
        dm = cdist(train_data, x, metric='euclidean')

        dist = dm.flatten()
        knn_matrix[i] = dist.argsort()[:k]

        wt = [0, 0]
        for j in range(k):
            ind = int(knn_matrix[i][j])
            # wt[int(train_label[ind])] += 1
            wt[int(train_label[ind])] += 1 / dist[ind]

        wt = np.asarray(wt)
        pred[i] = wt.argmax(axis=0)

    cm = get_confusion_matrix(test_label, pred)

    a = cm[0][0]
    b = cm[0][1]
    c = cm[1][0]
    d = cm[1][1]

    accuracy = (float)(a + d) / (float)(a + b + c + d)
    precision = (float)(a) / (float)(a + c)
    recall = (float)(a) / (float)(a + b)
    f1_measure = (float)(2 * a) / (float)(2 * a + b + c)

    return accuracy, precision, recall, f1_measure


def get_train_data(split_data, ind):
    return np.vstack([x for i, x in enumerate(split_data) if i != ind])


def main(k, n, data):
    accuracy_list = list()
    precision_list = list()
    recall_list = list()
    f1_measure_list = list()

    # n-fold validation
    split_data = np.array_split(data, n)

    for i in range(n):
        test = split_data[i]
        train = get_train_data(split_data, i)

        col = np.shape(test)[1]

        test_data = test[:, 0: col - 1]
        test_label = test[:, col - 1]
        train_data = train[:, 0: col - 1]
        train_label = train[:, col - 1]

        accuracy, precision, recall, f1_measure = knn(k, train_data, test_data, train_label, test_label)
        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_measure_list.append(f1_measure)

    print("Accuracy: ", sum(accuracy_list) / n)
    print("Precision: ", sum(precision_list) / n)
    print("Recall: ", sum(recall_list) / n)
    print("F1-Measure: ", sum(f1_measure_list) / n)
    print("")
    return sum(accuracy_list) / n, sum(precision_list) / n, sum(recall_list) / n, sum(f1_measure_list) / n


import matplotlib.pyplot as plt

if __name__ == '__main__':

    file = 'project3_dataset1.txt'
    k = 10
    data = readfile(file)

    # 10-fold validation
    n = 10

    # Enable this to plot performance measures vs k
    plot = False
    if plot:
        a_list = list()
        b_list = list()
        c_list = list()
        d_list = list()

        for k in range(1, 15):
            # k = 6
            a, b, c, d = main(k, n, data)
            a_list.append(a)
            b_list.append(b)
            c_list.append(c)
            d_list.append(d)

        fig = plt.figure(figsize=[12, 6])
        ax = fig.gca
        plt.subplot(221)
        plt.plot(a_list)
        plt.xlabel("Accuracy")
        plt.ylabel("k-values")
        # plt.show()
        plt.subplot(222)
        plt.plot(b_list)
        plt.xlabel("precision")
        plt.ylabel("k-values")
        # plt.show()
        plt.subplot(223)
        plt.plot(c_list)
        plt.xlabel("Recall")
        plt.ylabel("k-values")
        # plt.show()
        plt.subplot(224)
        plt.plot(d_list)
        plt.xlabel("F1-Measure")
        plt.ylabel("k-values")
        plt.show()
        print()

    else:
        main(k, n, data)
