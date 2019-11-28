import pandas as pd
import numpy as np
import statistics
import random
import time
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
from DecisionTree import DecisionTree, readfile, get_confusion_matrix
from collections import Counter


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

class RandomForest(object):
	"""docstring for RandomForest"""
	def __init__(self, N, m):
		self.N = N
		self.m = m
		self.Trees = list()
		for i in range(N):
			self.Trees.append(DecisionTree())
		print("Trees initialized")
		# self.classified_labels = None

	def get_bootstrapped_data(self, data):
		return data[np.random.choice(range(len(data)), len(data), replace=True)]

	def fit(self, data):
		for i in range(self.N):
			b_data = self.get_bootstrapped_data(data)
			self.Trees[i].fit(b_data)

	def classify(self, data):
		result = list()
		for i in range(self.N):
			self.Trees[i].classify(data)
			result.append(self.Trees[i].classified_labels)
		result = np.array(result, order='F').T
		print(Counter(result[1]).most_common(1)[0][0])
		self.classified_labels =  [ (Counter(result[i]).most_common(1))[0][0] for i in range(result.shape[0]) ]

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
		rf = RandomForest(5, .6)
		rf.fit(train)
		rf.classify(test)
		accuracy, precision, recall, f1_measure = rf.accuracy_measures(test[:, -1])
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
    start = time.time()
    main(file, n)
    done = time.time()
    elapsed = done - start
    print(elapsed)
