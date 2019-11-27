import pandas as pd
import numpy as np
import statistics
import random
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer

def readfile(filename):
    data = pd.read_csv(filename, header=None, sep="\t")
    # data = np.array(data.values)
    return data

class Tree(object):
	"""docstring for TreeNode"""
	def __init__(self, label=None, index=-1):

		#index = -1 and label not None denotes leaf node
		self.split_index = index
		self.label = label
		self.test_attr = None
		self.children = None
		# if attributeset:
		# 	attributeset = self.attributeset

class DecisionTree:
	"""docstring for DecisionTree"""

	def __init__(self):
		# self.attributeset = attributeset
		self.Tree = Tree()
		self.previous_split = list()
		self.min_threshold = 3
		self.classified_labels = None
		self.data_types = None

	def gini_node(self, node_data, classes, number_of_columns):
			classet1 = node_data[np.where(node_data[:,number_of_columns - 1] == classes[0])]
			classet2 = node_data[np.where(node_data[:,number_of_columns - 1] == classes[1])]
			l1 = len(node_data)
			lc1 = len(classet1)					
			lc2 = len(classet2)
			# print("impurity for "+str(value)+" is",1 - ((lc1/l1)	** 	2 + (lc2/l1) ** 2))
			gini_impurity =  (1 - ((lc1/l1)	** 	2 + (lc2/l1) ** 2))
			return gini_impurity

	def generate_candidates(self, col_values, std_dev):
		candidates = list()
		if min(col_values) == 0 :
			previous = 0
		else:
			previous = min(col_values) - std_dev
		for value in col_values:
			candidates.append((previous + value) / 2)
			previous = value
		candidates.append(col_values[-1] + std_dev)
		return candidates

	def get_best_candidate(self, data, candidates, col, classes):
		sorted_data = data.sort(axis = 0, kind = 'heapsort', order=[str(col)])
		length = len(data)
		number_of_columns = data.shape[1] - 1
		prime_candidate = candidates[0]
		set1 = data[np.where(data[:, col] <= candidates[0])]
		l1 = len(set1)
		set2 = data[np.where(data[col] > candidates[0])]
		l2 = len(set2)
		lc1_set1 = len(set1[np.where(set1[:, number_of_columns] == classes[0])])
		lc2_set1 = len(set1[np.where(set1[:, number_of_columns] == classes[1])])
		lc1_set2 = len(set2[np.where(set2[:, number_of_columns] == classes[0])])
		lc2_set2 = len(set2[np.where(set2[:, number_of_columns] == classes[1])])
		if(l1 == 0):
			gini_set1 = 0
		else:
			gini_set1 = self.gini_node_by_counts(l1, lc1_set1,lc1_set1)
		if(l2 == 0):
			gini_set2 = 0
		else:
			gini_set2 = self.gini_node_by_counts(l2, lc1_set2, lc2_set2)
		min_gini_split = (l1/(length))*gini_set1 + (l2/(length))*gini_set2
		j = 0
		for i in range(1, len(candidates)):
			if(data[j,number_of_columns] == classes[0]):
				lc1_set1 = lc1_set1 + 1
				lc1_set2 = lc1_set2 - 1
			elif (data[j,number_of_columns] == classes[1]):
				lc2_set1 = lc2_set1 + 1
				lc2_set2 = lc2_set2 - 1
			if(lc1_set1+lc2_set1 == 0):
				gini_set1 = 0
			else:			
				gini_set1 = self.gini_node_by_counts(lc1_set1+lc2_set1, lc1_set1,lc2_set1)
			if(l2 == 0):
				gini_set2 = 0
			else:
				gini_set2 = self.gini_node_by_counts(lc2_set1+lc2_set2, lc2_set1, lc2_set2)
			gini_split = (lc1_set1+lc2_set1/(lc1_set1+lc2_set1+lc2_set1+lc2_set2))*gini_set1 + (lc2_set1+lc2_set2/(lc1_set1+lc2_set1+l2))*gini_set2
			if(gini_split < min_gini_split):
				min_gini_split = gini_split
				prime_candidate = candidates[i]
		print("Min split gin for att",col," is ",min_gini_split, " with prime_candidate",prime_candidate)
		return min_gini_split, prime_candidate

	def find_best_split(self, data):
		# types = data.dtypes
		max_gain = -float("inf")
		prime_split_val = 0
		best_split_attr = 0
		number_of_columns = data.shape[1]
		parent_length = data.shape[0]
		classes = np.unique(data[:,number_of_columns - 1])
		gini_p = self.gini_node(data, classes, number_of_columns)
		print("gini_p", gini_p)
		col_value_map = dict()
		columns_rev = [x for x in range(number_of_columns) and x not in self.previous_split ]
		for i in colums_rev:
				candidates = self.generate_candidates(sorted(data[:,i]), statistics.stdev(data[:,i]))
				# print(candidates)
				gini_col, split_val = self.get_best_candidate(data, candidates, i, classes)
				# print("Gini col, split val",gini_col,split_val)
				gain = gini_p - gini_col
				if(gain > max_gain):
					max_gain = gain
					prime_split_val = split_val
					best_split_attr = i
		self.previous_split.append(max_index[0])
		print("max_index", best_split_attr)
		return best_split_attr, prime_split_val

	def evaluate_stop_condition(self, data):
		if (len(data[:,data.shape[1] - 1].unique()) == 1 or (len(data) <= self.min_threshold)):
			return True 
		else:
			return False

	def split_attr(self, data, val, child_index):
		result_set = []
		if val is None:
			unique = np.unique(data[:,child_index])
			for value in unique:
				set1 = data[np.where(data[:,child_index] == value)]
				result_set.append(set1)
		else:
			set1 = data[np.where(data[:,child_index] <= val)]
			set2 = data[np.where(data[:,child_index] > val)]
			result_set.append(set1)
			result_set.append(set2)
		return result_set

	def buildTree(self, data):
		print("len of data is", len(data))
		if self.evaluate_stop_condition(data):#check stop condition
			#if True, tree generation ends by creating Leaf
			classes = data[-1].unique()
			p = len(data)
			relative_freq = []
			if(len(classes) > 1):
				for clas in classes:
					pi = len(data.loc[data[data.shape[1] - 1] == clas])
					relative_freq.append(pi/p)
				label = classes[np.argmax(relative_freq)]
			else:
				label = data.iloc[-1].unique()[0]
			leaf = Tree(label=label)
			print("Leaf Created with label ", label)
			return leaf
		else:
			root = Tree()
			child_index, val = self.find_best_split(data, columns)
			print("split on attr ",child_index," with val", val)
			root.split_index = child_index
			result_set = self.split_attr(data, val, child_index)
			root.children = dict()
			# if val:
			# 	root.test_attr = val
			# 	child0= self.buildTree(result_set[0])
			# 	child1= self.buildTree(result_set[1])
			# 	root.children["less than equal"+str(val)] = child0
			# 	root.children["greater than "+str(val)] = child1
			# else:
			unique = np.unique(data[:,child_index])
			# print("U ",unique)
			for i,set_child in enumerate(result_set):
				child = self.buildTree(set_child, columns)
				root.children[str(unique[i])] = (child)
			return root

	def fit(self, data, data_types):
		self.data_types = data_types
		self.Tree = self.buildTree(data, len(data_types))

def preprocess(data, types):
	cols = len(types)
	labels = data[cols - 1]
	data = np.delete(np.array(data),cols-1, 1)
	listcol = [x for x in range(cols -1) if types[x] == "int64" and len(np.unique(data[:,x])) > 2 ]
	ct = make_column_transformer((listcol,OneHotEncoder(categories="auto", sparse=False)))
	val = ct.fit_transform(data)
	data = np.delete(data,listcol, 1)
	data = pd.DataFrame(np.hstack((data,val,pd.DataFrame(labels))))
	data = data.infer_objects()
	data_types = data.dtypes
	return data, data_types

def get_train_data(split_data, ind):
	return np.vstack([x for i, x in enumerate(split_data) if i != ind])

def main(file, n):
	data = readfile(file)
	data_types = data.dtypes
	data, data_types = preprocess(data, data_types)
	accuracy_list = list()
	precision_list = list()
	recall_list = list()
	f1_measure_list = list()
	data = np.array(data)
	print(len(data))
	print(data_types[1])	
	# split_data = np.array_split(data, n)
	# for i in range(n):
	# 	test = split_data[i]
	# 	train = get_train_data(split_data, i)
	# 	print(train)
	# 	dc = DecisionTree()
	# 	dc.fit(train, data_types)


if __name__ == '__main__':
    file = 'project3_dataset2.txt'
    n= 10
    main(file, n)