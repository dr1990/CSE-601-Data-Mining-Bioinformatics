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
        self.visited_node = list()


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

    def __init__(self, variable_feature=False, m=0):
        self.Tree = Tree()
        self.threshold = 1
        self.variable_feature = variable_feature
        self.m = m
        # self.visited_feature = list()

    def stopping_condition(self, data):
        return len(data) <= self.threshold or (len(np.unique(data[:, -1])) == 1)
        # return (len(np.unique(data[:, -1])) == 1)

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

    def get_best_candidate(self, data, candidates, col, label_count):  # fig 4.16
        min_gini_split = float("inf")
        prime_candidate = 0

        gini_imp_left = 0
        gini_imp_right = 0

        candidate_gini_table = np.zeros((len(candidates), 2, 2), dtype='int')

        # candidate_gini_table[i]
        # label:  <= | >
        #   0:     a | b
        #   1:     c | d

        k = 0
        for i, candidate in enumerate(candidates):
            inc = np.zeros(2, dtype='int')
            cnt = 0
            for x in data[k:, :]:
                if x[col] <= candidate:
                    inc[int(x[-1])] += 1
                    cnt += 1
                else:
                    k += cnt
                    if i == 0:
                        candidate_gini_table[i][0][0] = inc[0]
                        candidate_gini_table[i][1][0] = inc[1]
                        candidate_gini_table[i][0][1] = label_count[0] - inc[0]
                        candidate_gini_table[i][1][1] = label_count[1] - inc[1]
                    else:
                        candidate_gini_table[i][0][0] = candidate_gini_table[i - 1][0][0] + inc[0]
                        candidate_gini_table[i][1][0] = candidate_gini_table[i - 1][1][0] + inc[1]
                        candidate_gini_table[i][0][1] = candidate_gini_table[i - 1][0][1] - inc[0]
                        candidate_gini_table[i][1][1] = candidate_gini_table[i - 1][1][1] - inc[1]
                    break

            total_len = np.sum(candidate_gini_table[i])

            l0 = candidate_gini_table[i][0][0] / np.sum(candidate_gini_table[i][:, 0])
            l1 = candidate_gini_table[i][1][0] / np.sum(candidate_gini_table[i][:, 0])
            r0 = candidate_gini_table[i][0][1] / np.sum(candidate_gini_table[i][:, 1])
            r1 = candidate_gini_table[i][1][1] / np.sum(candidate_gini_table[i][:, 1])

            gini_impurity_left = float(1.0 - (l0 ** 2) - (l1 ** 2))
            gini_impurity_right = float(1.0 - (r0 ** 2) - (r1 ** 2))

            gini_split = ((gini_impurity_left * np.sum(candidate_gini_table[i][:, 0])) + (
                    gini_impurity_right * np.sum(candidate_gini_table[i][:, 1]))) / total_len

            if gini_split < min_gini_split:
                min_gini_split = gini_split
                prime_candidate = candidate
                gini_imp_left = gini_impurity_left
                gini_imp_right = gini_impurity_right

        # if min_gini_split == 0.0:
        #     print()
        return min_gini_split, prime_candidate, gini_imp_left, gini_imp_right

    def generate_candidates(self, col_values):
        candidates = list()
        first = True
        prev = 0.0

        for value in col_values:
            if first:
                prev = value
                first = False
            else:
                candidates.append((prev + value) / 2)
                prev = value

        return candidates

    def get_feature_list(self, cols, root):
        if use_visited_col:
            available_features = [x for x in range(cols)]
            return available_features
        else:
            available_features = [x for x in range(cols) if x not in root.visited_node]
            np.random.shuffle(available_features)
            return available_features[0:self.m]

    def find_best_split(self, data, root, label_count):
        prev_gain = float("inf")
        gini_imp = float("inf")
        split_val = -1
        split_col = -1
        gini_imp_left = -1
        gini_imp_right = -1

        if self.variable_feature:
            feature_list = self.get_feature_list(data.shape[1] - 1, root)
        else:
            if use_visited_col:
                feature_list = [x for x in range(data.shape[1] - 1)]
            else:
                feature_list = [x for x in range(data.shape[1] - 1) if x not in root.visited_node]

        # print()
        for i in feature_list:
            unique = np.unique(data[:, i])
            candidates = self.generate_candidates(sorted(unique))

            df = pd.DataFrame(data)
            sorted_data = np.asarray(df.sort_values(by=i))

            gini_val, sv, gl, gr = self.get_best_candidate(sorted_data, candidates, i, label_count)

            if prev_gain > gini_val:
                prev_gain = gini_val
                split_val = sv
                split_col = i
                gini_imp = gini_val
                gini_imp_left = gl
                gini_imp_right = gr

        # if gini_imp == 0.0:
        #     print()

        # Comment this in order to repeat the column in the
        if not use_visited_col:
            if len(feature_list) == 0:
                return -1, None, None, None, None

            # root.append(split_col)

        if gini_imp == float("inf"):
            return -1, None, None, None, None
            # print()
        return split_col, split_val, gini_imp, gini_imp_left, gini_imp_right

    def get_max_label(self, data):
        cnt = [0, 0]

        for x in data[:, -1]:
            cnt[int(x)] += 1

        if cnt[0] > cnt[1]:
            return 0
        else:
            return 1

    def gini_parent(self, node_data):
        class_count = [0, 0]

        for x in node_data[:, -1]:
            class_count[int(x)] += 1

        l1 = len(node_data)

        lc1 = class_count[0]
        lc2 = class_count[1]

        gini_impurity = (1 - ((lc1 / l1) ** 2 + (lc2 / l1) ** 2))

        return gini_impurity

    def get_class_count(self, data):
        total = [0, 0]

        for x in data:
            total[int(x[-1])] += 1

        return total

    def buildTree(self, root, data):
        label_count = self.get_class_count(data)

        split_index, val, gini_impurity, gl, gr = self.find_best_split(data, root, label_count)

        gini_parent = self.gini_parent(data)

        if split_index != -1 and gini_parent < gini_impurity:
            print()
        # No further split
        if split_index == -1 or gini_impurity == 0.0 or gini_parent < gini_impurity:
            root.label = self.get_max_label(data)
            # root.left_child_data = None
            # root.right_child_data = None
            # self.visited_feature.remove(split_index)
            return root

        root.split_index = split_index
        root.test_attr = val
        root.gini_imp = gini_impurity
        root.gini_left = gl
        root.gini_right = gr
        root.visited_node.append(split_index)

        lc, rc = self.split_data(data, val, split_index)

        root.left_child_data = np.asarray(lc)
        root.right_child_data = np.asarray(rc)

        if len(root.left_child_data) != 0:
            # if len(np.unique(root.left_child_data[:, -1])) == 1:
            if self.stopping_condition(root.left_child_data):
                root.left_child = Tree()
                root.left_child.label = self.get_max_label(root.left_child_data)
                # root.left_child_data = None
            else:
                left_child = Tree()
                left_child.visited_node.extend(root.visited_node)
                root.left_child = self.buildTree(left_child, root.left_child_data)
                # root.left_child_data = None

        if len(root.right_child_data) != 0:
            # if len(np.unique(root.right_child_data[:, -1])) == 1:
            if self.stopping_condition(root.right_child_data):
                root.right_child = Tree()
                root.right_child.label = self.get_max_label(root.right_child_data)
                # root.right_child_data = None

            else:
                right_child = Tree()
                right_child.visited_node.extend(root.visited_node)
                root.right_child = self.buildTree(right_child, root.right_child_data)
                # root.right_child_data = None

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

        dc = DecisionTree(False)
        dc.fit(train)
        dc.classify(test)
        accuracy, precision, recall, f1_measure = dc.accuracy_measures(test[:, -1])

        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_measure_list.append(f1_measure)

        del dc

        # print("Accuracy: ", accuracy)
        # print("Precision: ", precision)
        # print("Recall: ", recall)
        # print("F1-Measure: ", f1_measure)
        # print()

    print("Accuracy: ", sum(accuracy_list) / n)
    print("Precision: ", sum(precision_list) / n)
    print("Recall: ", sum(recall_list) / n)
    print("F1-Measure: ", sum(f1_measure_list) / n)


use_visited_col = False

if __name__ == '__main__':
    file = 'project3_dataset1.txt'
    n = 10
    start = time.time()
    main(file, n)
    done = time.time()
    elapsed = done - start
    print(elapsed)
