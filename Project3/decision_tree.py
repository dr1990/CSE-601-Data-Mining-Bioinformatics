import pandas as pd
import numpy as np
import statistics
import random
from sklearn.preprocessing import OneHotEncoder, Normalizer
from sklearn.compose import ColumnTransformer, make_column_transformer

def get_confusion_matrix(train_label, pred):
    shape = np.shape(train_label)
    row = shape[0]

    cm = np.zeros((2, 2))

    for i in range(row):
        x = int(train_label[i])
        y = int(pred.iloc[i])
        cm[x][y] += 1

    return cm

#onehotencodeing remaining
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

	def evaluate_stop_condition(self, data):
		if (len(data[data.shape[1] - 1].unique()) == 1 or (len(data) <= self.min_threshold)):
			return True 
		else:
			return False 
	def gini_node(self, node_data, classes):
			number_of_columns = node_data.shape[1] - 1
			classet1 = node_data.loc[node_data[number_of_columns] == classes[0]]
			classet2 = node_data.loc[node_data[number_of_columns] == classes[1]]
			l1 = len(node_data)
			# classet = []
			# i = 0
			# for clas in classes:
			# 	classet[i] = node_data.loc[node_data[number_of_columns - 1] == clas]
			# print("l1 is", l1)
			lc1 = len(classet1)
			# print("lc1 is", lc1)						
			lc2 = len(classet2)
			# print("lc2 is", lc2)
			# print("impurity for "+str(value)+" is",1 - ((lc1/l1)	** 	2 + (lc2/l1) ** 2))
			gini_impurity =  (1 - ((lc1/l1)	** 	2 + (lc2/l1) ** 2))
			return gini_impurity

	def gini_node_by_counts(self, l1, lc1, lc2):
		return (1 - ((lc1/l1)	** 	2 + (lc2/l1) ** 2))

	def get_best_candidate(self, data, candidates, col, classes):
		sorted_data = data.sort_values(by=col)
		length = len(data)
		number_of_columns = data.shape[1] - 1
		prime_candidate = candidates[0]
		l1 = 0
		l2 = 0
		set1 = data.loc[data[col] <= candidates[0]]
		l1 = len(set1)
		set2 = data.loc[data[col] > candidates[0]]
		l2 = len(set2)
		lc1_set1 = len(set1.loc[set1[number_of_columns] == classes[0]])
		lc2_set1 = len(set1.loc[set1[number_of_columns] == classes[1]])
		lc1_set2 = len(set2.loc[set2[number_of_columns] == classes[0]])
		lc2_set2 = len(set2.loc[set2[number_of_columns] == classes[1]])
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
			if(data.iloc[j,number_of_columns] == classes[0]):
				lc1_set1 = lc1_set1 + 1
				lc1_set2 = lc1_set2 - 1
			elif (data.iloc[j,number_of_columns] == classes[1]):
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
		return min_gini_split, prime_candidate
	# def get_best_candidate(self, data, candidates, col, classes):
	# 	min_gini_split = float("inf")
	# 	prime_candidate = candidates[0]
	# 	#Needs to be optimized
	# 	for candidate in candidates:
	# 		set1 = data.loc[data[col] <= candidate]
	# 		l1 = len(set1)
	# 		set2 = data.loc[data[col] > candidate]
	# 		l2 = len(set2)
	# 		if(l1 == 0):
	# 			gini_set1 = 0
	# 		else:
	# 			gini_set1 = self.gini_node(set1, classes)
	# 		if(l2 == 0):
	# 			gini_set2 = 0
	# 		else:
	# 			gini_set2 = self.gini_node(set2, classes)
	# 		gini_split = (l1/(l1+l2))*gini_set1 + (l2/(l1+l2))*gini_set2
	# 		if(gini_split < min_gini_split):
	# 			min_gini_split = gini_split
	# 			prime_candidate = candidate
	# 	return min_gini_split, candidate


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

	def find_best_split(self, data, columns):
		gain_split = []
		types = data.dtypes
		number_of_columns = data.shape[1]
		parent_length = data.shape[0]
		classes = data[number_of_columns - 1].unique()
		gini_p = self.gini_node(data, classes)
		print("gini_p", gini_p)
		col_value_map = dict()
		columns_rev = [x for x in range(number_of_columns) if x not in self.previous_split ]
		for i in columns_rev:
			# unique = data[i].unique()
			# # print("Len unique is",len(unique))
			# if(types[i] == "object" or types[i] == "int64"): #Nominal/Binary values
			# 	gini_col = 0
			# 	for value in unique:
			# 		set1 = data.loc[data[i]==value]
			# 		l1 = len(set1)
			# 		gini_col = gini_col + (l1/parent_length) * self.gini_node(set1,classes)
			# 	gain = gini_p - gini_col
			# 	gain_split.append(gain)
			# 	col_value_map[i] = None
			# else:
				candidates = self.generate_candidates(sorted(data[i]), statistics.stdev(data[i]))
				# print(candidates)
				gini_col, split_val = self.get_best_candidate(data, candidates, i, classes)
				# print("Gini col, split val",gini_col,split_val)
				gain = gini_p - gini_col
				gain_split.append(gain)
				col_value_map[i] = split_val
		# print(col_value_map)
		# print(gain_split)
		#randomness added to choose from more than 1 max gain indices
		max_index = random.choice(np.where(gain_split == np.amax(gain_split)))
		self.previous_split.append(max_index[0])
		print("max_index", max_index[0])
		return max_index[0],col_value_map[columns_rev[max_index[0]]]
	
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
			result_set.append(set1)
			result_set.append(set2)
		return result_set


	def buildTree(self, data, columns):
		print("len of data is", len(data))
		if self.evaluate_stop_condition(data):#check stop condition
			#if True, tree generation ends by creating Leaf
			classes = data.iloc[-1].unique()
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
			c_i = columns.where(columns == child_index)
			c_i = c_i.dropna()
			print("Delete att :", child_index)
			print(c_i)
			columns = columns.drop(labels=c_i)
			print("split on attr ",child_index," with val", val)
			root.split_index = child_index
			result_set = self.split_attr(data, val, child_index)
			root.children = dict()
			if val:
				root.test_attr = val
				child0= self.buildTree(result_set[0], columns)
				child1= self.buildTree(result_set[1], columns)
				root.children["less than equal"+str(val)] = child0
				root.children["greater than "+str(val)] = child1
			else:
				unique = data[child_index].unique()
				# print("U ",unique)
				for i,set_child in enumerate(result_set):
					child= self.buildTree(set_child, columns)
					root.children[str(unique[i])] = (child)
			return root

	def fit(self, data):
		self.Tree = self.buildTree(data, data.columns)

	def classify(self, data):
		labels = []
		for row in range(len(data)):
			root = self.Tree
			print("classify", row)
			while(root.split_index != -1 and root.label is None):
				if root.test_attr is None:
					root = root.children[str(data.iloc[row,root.split_index])]
				else:
					if(row[root.split_index] <= root.test_attr):
						root = root.children["less than equal"+str(root.test_attr)]	
					else:
						root = root.children["greater than "+str(root.test_attr)]
			labels.append(int(root.label))
		self.classified_labels = labels

	def accuracy_measures(self, truelabels):
		cm = get_confusion_matrix(self.classified_labels, truelabels)
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

def preprocess(data):
	labels = data[data.shape[1] - 1]
	data = data.drop(columns=data.shape[1] - 1)
	types = data.dtypes
	clean_up = { 4 : {"Present" : 1, "Absent" : 0} }
	data.replace(clean_up, inplace=True)
	# listcol = [x for x in data.columns if types[x] != "object" ]
	ct = make_column_transformer((data.columns,OneHotEncoder(categories="auto", sparse=False)),
									(data.columns, Normalizer()))
	val = ct.fit_transform(data)
	# data = data.drop(columns=listcol)
	data = pd.DataFrame(np.hstack((val,pd.DataFrame(labels))))
	return data

def main(file, n):
	data = readfile(file)
	data = preprocess(data)
	print(data)
	accuracy_list = list()
	precision_list = list()
	recall_list = list()
	f1_measure_list = list()
	# n-fold validation
	split_data = np.array_split(data, n)
	for i in range(n):
		test = split_data[i]
		train = get_train_data(split_data, i)
		train = pd.DataFrame(train)
		train = train.infer_objects()
		print(train.dtypes)
		dc = DecisionTree()
		dc.fit(train)
		dc.classify(test)
		accuracy, precision, recall, f1_measure = dc.accuracy_measures(test[test.shape[1]-1])
		accuracy_list.append(accuracy)
		precision_list.append(precision)
		recall_list.append(recall)
		f1_measure_list.append(f1_measure)
	print("Accuracy: ", sum(accuracy_list) / n)
	print("Precision: ", sum(precision_list) / n)
	print("Recall: ", sum(recall_list) / n)
	print("F1-Measure: ", sum(f1_measure_list) / n)

if __name__ == '__main__':
    file = 'project3_dataset2.txt'
    n= 10
    main(file, n)
