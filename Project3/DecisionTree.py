import pandas as pd
import numpy as np
import statistics
import random
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer


def get_confusion_matrix(train_label, pred):
    shape = np.shape(train_label)
    row = shape[0]

    cm = np.zeros((2, 2))

    for i in range(row):
        x = int(train_label[i])
        y = int(pred[i])
        cm[x][y] += 1

    return cm


str_ind = list()


def readfile(filename):
    data = pd.read_csv(filename, header=None, sep="\t")
    data = np.array(data.values)

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


class DecisionTree:
    """docstring for DecisionTree"""

    def __init__(self):
        # self.attributeset = attributeset
        self.Tree = Tree()
        self.min_threshold = 3
        self.classified_labels = None

    def evaluate_stop_condition(self, data):
        if (len(np.unique(data[data.shape[1] - 1])) == 1 or (len(data) <= self.min_threshold)):
            return True
        else:
            return False

    def gini_node(self, node_data):

        # Assuming only two classes exist
        # classet1 = [x for x in node_data if x[number_of_columns] == classes[0]]
        # classet2 = [x for x in node_data if x[number_of_columns] == classes[1]]

        class_count = [0, 0]
        node_data = np.asarray(node_data)
        for x in node_data[:, -1]:
            class_count[int(x)] += 1

        l1 = len(node_data)

        lc1 = class_count[0]
        lc2 = class_count[1]

        gini_impurity = (1 - ((lc1 / l1) ** 2 + (lc2 / l1) ** 2))
        return gini_impurity

    def get_gini_index(self, left, right):
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

        return ((gini_impurity_left * len(left)) + (gini_impurity_right * len(right))) / total_len

    def get_best_candidate(self, data, candidates, col, classes):  # fig 4.16
        min_gini_split = float("inf")
        # prime_candidate = candidates[0] #TODO: ??
        prime_candidate = 0

        # Needs to be optimized
        left_child = list()
        right_child = list()

        for candidate in candidates:
            left = list()
            right = list()
            for x in data:
                if x[col] <= candidate:
                    left.append(x)  # can be optimized. we can calculate class_count values here.
                else:
                    right.append(x)  # can be optimized. we can calculate class_count values here.

            gini_split = self.get_gini_index(left, right)
            if gini_split < min_gini_split:
                min_gini_split = gini_split
                prime_candidate = candidate
                left_child = left
                right_child = right

        return min_gini_split, prime_candidate, left_child, right_child

    def generate_candidates(self, col_values, std_dev):
        candidates = list()
        # if min(col_values) == 0:
        #     previous = 0
        # else:
        #     previous = min(col_values) - std_dev

        prev = 0
        for value in col_values:
            if prev == 0:
                prev = value
            else:
                candidates.append((prev + value) / 2)
                prev = value
        # candidates.append(col_values[-1] + std_dev)
        return candidates

    def find_best_split(self, root, data):
        tree = Tree()
        prev_gain = float("inf")
        row = data.shape[0]
        col = data.shape[1]
        classes = np.unique(data[:, col - 1])
        left_child = list()
        right_child = list()

        gini_p = self.gini_node(data)  # TODO: 1. Why do we need this? 2. can't we save this value?

        for i in range(col - 1):
            unique = np.unique(data[:, i])
            candidates = self.generate_candidates(sorted(unique), 0)
            gini_val, sv, lc, rc = self.get_best_candidate(data, candidates, i, classes)

            if prev_gain > gini_val:
                prev_gain = gini_val
                left_child = lc
                right_child = rc
                split_val = sv
                split_col = i
                gini_imp = gini_val

            # gain_split.append(gain)
            # col_value_map[i] = split_val

        # randomness added to choose from more than 1 max gain indices
        # max_index = random.choice(np.where(gain_split == np.amax(gain_split))).flatten()

        if gini_p < gini_imp:
            return -1, None, None, None, None

        return split_col, split_val, left_child, right_child, gini_imp
        # return max_index[0], col_value_map[max_index[0]]

    def split_attr(self, data, val, child_index):
        result_set = []
        if val is None:
            unique = data[child_index].unique()
            for value in unique:
                set1 = data.loc[data[child_index] == value]
                result_set.append(set1)
        else:
            set1 = data.loc[data[child_index] <= val]
            set2 = data.loc[data[child_index] > val]
            result_set.appxend(set1)
            result_set.append(set2)
        return result_set

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

        split_index, val, lc, rc, gini_imp = self.find_best_split(root, data)

        # No further split
        if split_index == -1:
            root.label = self.get_max_label(data)
            root.left_child_data = None
            root.right_child_data = None
            return root

        root.split_index = split_index
        root.test_attr = val
        root.gini_imp = gini_imp

        left_child = np.asarray(lc)
        right_child = np.asarray(rc)

        root.left_child_data = left_child
        root.right_child_data = right_child

        if len(left_child) != 0:
            if len(np.unique(left_child[:, -1])) == 1:
                root.left_child = Tree()
                root.left_child.label = self.get_max_label(root.left_child_data)
                root.left_child_data = None
            else:
                root.left_child = self.buildTree(root.left_child, root.left_child_data)
                root.left_child_data = None

        if len(right_child) != 0:
            if len(np.unique(right_child[:, -1])) == 1:
                root.right_child = Tree()
                root.right_child.label = self.get_max_label(root.right_child_data)
                root.right_child_data = None
            else:
                root.right_child = self.buildTree(root.right_child, root.right_child_data)
                root.right_child_data = None

        return root

    def fit(self, data):
        # root = Tree()
        # root.gini_imp = self.gini_node(data)
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
                        # root = root.children["less than equal" + str(root.test_attr)]
                        root = root.left_child
                    else:
                        # root = root.children["greater than " + str(root.test_attr)]
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
        accuracy, precision, recall, f1_measure = dc.accuracy_measures(test[:, test.shape[1] - 1])

        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_measure_list.append(f1_measure)

    print("Accuracy: ", sum(accuracy_list) / n)
    print("Precision: ", sum(precision_list) / n)
    print("Recall: ", sum(recall_list) / n)
    print("F1-Measure: ", sum(f1_measure_list) / n)


if __name__ == '__main__':
    file = 'project3_dataset1.txt'
    n = 10
    main(file, n)
