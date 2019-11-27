import pandas as pd
import numpy as np
import time


def get_confusion_matrix(train_label, pred):
    shape = np.shape(train_label)
    row = shape[0]

    cm = np.zeros((2, 2))

    for i in range(row):
        x = int(train_label[i])
        y = int(pred[i])
        cm[x][y] += 1

    return cm


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


class Tree(object):
    """docstring for TreeNode"""

    def __init__(self, label=None, index=-1):
        # index = -1 and label not None denotes leaf node
        self.split_index = index
        self.label = label
        self.test_attr = None
        self.left_child = None
        self.right_child = None
        self.gini_imp = float("inf")
        self.left_child_data = None
        self.right_child_data = None
        self.gini_left = None
        self.gini_right = None


def get_gini_index(left, right):
    class_count = np.zeros((2, 2), dtype='float')

    left = np.asarray(left)
    right = np.asarray(right)

    if len(left) != 0:
        for x in left[:, -1]:
            class_count[int(x)][0] += 1
        # class_count[:, 0] = class_count[:, 0] / len(left)

    if len(right) != 0:
        for x in right[:, -1]:
            class_count[int(x)][1] += 1
        # class_count[:, 1] = class_count[:, 1] / len(right)

    total_len = len(left) + len(right)

    l0 = class_count[0][0] / len(left)
    l1 = class_count[1][0] / len(left)
    r0 = class_count[0][1] / len(right)
    r1 = class_count[1][1] / len(right)

    gini_impurity_left = float(1.0 - (l0 ** 2) - (l1 ** 2))
    gini_impurity_right = float(1.0 - (r0 ** 2) - (r1 ** 2))

    return ((gini_impurity_left * len(left)) + (
            gini_impurity_right * len(right))) / total_len, gini_impurity_left, gini_impurity_right


class DecisionTree:
    """docstring for DecisionTree"""

    def __init__(self):
        self.Tree = Tree()

    def gini_node(self, node_data):
        class_count = [0, 0]
        node_data = np.asarray(node_data)
        for x in node_data[:, -1]:
            class_count[int(x)] += 1

        l1 = len(node_data)

        lc1 = class_count[0]
        lc2 = class_count[1]

        gini_impurity = (1 - ((lc1 / l1) ** 2 + (lc2 / l1) ** 2))
        return gini_impurity

    def split_data(self, data, candidate, col):
        left = list()
        right = list()

        for x in data:
            if x[col] <= candidate:
                left.append(x)
            else:
                right.append(x)
        return left, right

    def get_best_candidate(self, data, candidates, col, classes):  # fig 4.16
        min_gini_split = float("inf")

        prime_candidate = 0

        # Needs to be optimized

        gini_imp_left = 0
        gini_imp_right = 0

        for candidate in candidates:
            class_count = np.zeros((2, 2), dtype='float')

            for x in data:
                if x[col] <= candidate:
                    class_count[int(x[-1])][0] += 1
                else:
                    class_count[int(x[-1])][1] += 1

            # gini_split, gini_left, gini_right = self.get_gini_index(left, right)

            ####

            # GINI Impurity Calculation
            total_len = np.sum(class_count)

            l0 = class_count[0][0] / np.sum(class_count[:, 0])
            l1 = class_count[1][0] / np.sum(class_count[:, 0])
            r0 = class_count[0][1] / np.sum(class_count[:, 1])
            r1 = class_count[1][1] / np.sum(class_count[:, 1])

            gini_impurity_left = float(1.0 - (l0 ** 2) - (l1 ** 2))
            gini_impurity_right = float(1.0 - (r0 ** 2) - (r1 ** 2))

            gini_split = ((gini_impurity_left * np.sum(class_count[:, 0])) + (gini_impurity_right * np.sum(class_count[:, 1]))) / total_len

            #####

            if gini_split < min_gini_split:
                min_gini_split = gini_split
                prime_candidate = candidate
                gini_imp_left = gini_impurity_left
                gini_imp_right = gini_impurity_right

        return min_gini_split, prime_candidate, gini_imp_left, gini_imp_right

    def generate_candidates(self, col_values):
        candidates = list()

        prev = 0
        for value in col_values:
            if prev == 0:
                prev = value
            else:
                candidates.append((prev + value) / 2)
                prev = value

        return candidates

    def find_best_split(self, data):
        prev_gain = float("inf")
        col = data.shape[1]
        classes = np.unique(data[:, col - 1])
        gini_imp = float("inf")
        split_val = 0
        split_col = 0
        gini_imp_left = 0
        gini_imp_right = 0

        gini_p = self.gini_node(data)  # TODO: 1. Why do we need this? 2. can't we save this value?

        for i in range(col - 1):
            unique = np.unique(data[:, i])
            candidates = self.generate_candidates(sorted(unique))
            # df = pd.DataFrame(data)
            # sorted_data = np.asarray(df.sort_values(by=i))

            gini_val, sv, gl, gr = self.get_best_candidate(data, candidates, i, classes)

            if prev_gain > gini_val:
                prev_gain = gini_val
                split_val = sv
                split_col = i
                gini_imp = gini_val
                gini_imp_left = gl
                gini_imp_right = gr

        if gini_p < gini_imp:
            return -1, None, None, None, None

        return split_col, split_val, gini_imp, gini_imp_left, gini_imp_right

    def get_max_label(self, data):
        cnt = [0, 0]

        for x in data[:, -1]:
            cnt[int(x)] += 1

        if cnt[0] > cnt[1]:
            return 0
        else:
            return 1

    def buildTree(self, root, data):
        if root is None:
            root = Tree()

        split_index, val, gini_impurity, gl, gr = self.find_best_split(data)

        # No further split
        if split_index == -1:
            root.label = self.get_max_label(data)
            root.left_child_data = None
            root.right_child_data = None
            return root

        root.split_index = split_index
        root.test_attr = val
        root.gini_imp = gini_impurity
        root.gini_left = gl
        root.gini_right = gr

        lc, rc = self.split_data(data, val, split_index)

        root.left_child_data = np.asarray(lc)
        root.right_child_data = np.asarray(rc)

        if len(root.left_child_data) != 0:
            if len(np.unique(root.left_child_data[:, -1])) == 1:
                root.left_child = Tree()
                root.left_child.label = self.get_max_label(root.left_child_data)
                root.left_child_data = None
            else:
                root.left_child = self.buildTree(root.left_child, root.left_child_data)
                root.left_child_data = None

        if len(root.right_child_data) != 0:
            if len(np.unique(root.right_child_data[:, -1])) == 1:
                root.right_child = Tree()
                root.right_child.label = self.get_max_label(root.right_child_data)
                root.right_child_data = None
            else:
                root.right_child = self.buildTree(root.right_child, root.right_child_data)
                root.right_child_data = None

        return root

    def fit(self, data):
        self.Tree = self.buildTree(self.Tree, data)
        print("Tree Built.")

    def classify(self, data):
        labels = []
        for row in range(len(data)):
            root = self.Tree

            while root.split_index != -1 and root.label is None:
                if root.test_attr is None:
                    root = root.children[str(data.loc[row, root.split_index])]
                else:
                    if data[row][root.split_index] <= root.test_attr:
                        root = root.left_child
                    else:
                        root = root.right_child
            labels.append(int(root.label))
        self.classified_labels = labels

    def accuracy_measures(self, truelabels):
        cm = get_confusion_matrix(truelabels, self.classified_labels)
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


def main(file, n):
    data = readfile(file)

    accuracy_list = list()
    precision_list = list()
    recall_list = list()
    f1_measure_list = list()
    # n-fold validation
    split_data = np.array_split(data, n)

    for i in range(n):
        test = split_data[i]
        train = get_train_data(split_data, i)

        dc = DecisionTree()
        dc.fit(train)
        dc.classify(test)
        accuracy, precision, recall, f1_measure = dc.accuracy_measures(test[:, -1])

        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_measure_list.append(f1_measure)

        # print("Accuracy: ", accuracy)
        # print("Precision: ", precision)
        # print("Recall: ", recall)
        # print("F1-Measure: ", f1_measure)
        # print()

    print("Accuracy: ", sum(accuracy_list) / n)
    print("Precision: ", sum(precision_list) / n)
    print("Recall: ", sum(recall_list) / n)
    print("F1-Measure: ", sum(f1_measure_list) / n)


if __name__ == '__main__':
    file = 'project3_dataset1.txt'
    n = 10
    start = time.time()
    main(file, n)
    done = time.time()
    elapsed = done - start
    print(elapsed)
