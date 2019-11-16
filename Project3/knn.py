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
    col = shape[1]

    knn_matrix = np.zeros((row, k))
    pred = np.zeros((row, 1))

    for i, x in enumerate(test_data):
        x = np.reshape(x, (-1, 1)).T
        dm = cdist(train_data, x, metric='euclidean')

        dist = dm.flatten()
        knn_matrix[i] = dist.argsort()[:k]

        # for i in range(row):
        wt = [0, 0]
        for j in range(k):
            ind = int(knn_matrix[i][j])
            # v = dist[ind]
            # if v != 0:
            # wt[int(train_label[ind])] += 1 / v
            wt[int(train_label[ind])] += 1

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


file = 'project3_dataset1.txt'
data = readfile(file)
k = 5
# n-fold validation
n = 10
split_data = np.array_split(data, n)

accuracy_list = list()
precision_list = list()
recall_list = list()
f1_measure_list = list()

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
