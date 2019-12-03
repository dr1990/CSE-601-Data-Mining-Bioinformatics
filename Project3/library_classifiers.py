import pandas as pd
import numpy as np
import time
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_validate
from sklearn.decomposition import PCA

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

def readfile(filename, header):
    data = pd.read_csv(filename, header=header, sep=",")
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

def sample_by_length(r, n, replace=False):
    return np.random.choice(range(r), n, replace=replace)

def randomize(data_feature, data_label):
	samples = np.random.choice(range(data_feature.shape[0]), data_feature.shape[0], replace=False)
	return data_feature[samples,:], data_label[samples,]
		
def main(file_train_data, file_train_labels, file_test_data, n):
	data = readfile(file_train_data, None)
	labels = readfile(file_train_labels, 0)
	data_feature = np.reshape(data[:, 1:], np.shape(data[:, 1:]))
	data_label = np.reshape(labels[:, 1], (-1, 1))
	data_label = np.reshape(data_label, (data_label.shape[0],))
	test = readfile(file_test_data, header=None)
	test_features = np.reshape(test[:, 1:], np.shape(test[:, 1:]))
	

	data_ones = data_feature[np.where(data_label[:,] == 1)]
	data_zero = data_feature[np.where(data_label[:,] == 0)]
	data_label_ones = data_label[np.where(data_label[:,] == 1)]
	data_label_zeroes = data_label[np.where(data_label[:,]== 0)]
	samples = sample_by_length(data_zero.shape[0], data_ones.shape[0], True)
	# print(samples)
	data_zero = data_zero[samples, :]
	data_label_zeroes = data_label_zeroes[samples,]
	data_feature = np.vstack((data_ones, data_zero))
	data_label = np.vstack((data_label_ones, data_label_zeroes)).flatten()
	# print(data_label.shape)	
	# print(data_feature)

	# LR, KNN was performing weak so scaled the data (Scaling required for distance based algo)
	scaler = StandardScaler()
	scaler.fit(data_feature)
	data_feature = scaler.transform(data_feature)
	scaler.fit(test_features)
	test_features = scaler.transform(test_features)

	# Q1 = pd.DataFrame(data_feature).quantile(0.25)
	# Q3 =  pd.DataFrame(data_feature).quantile(0.75)
	# IQR = Q3 - Q1
	# data_feature = data_feature[np.where(~((data_feature < (Q1 - 1.5 * IQR)) | (data_feature > (Q3 + 1.5 * IQR))))]

	print(data_feature.shape)
	classifiers = list()
	clf1 = LogisticRegression(solver='lbfgs', random_state=1, max_iter=500)
	clf2 = RandomForestClassifier(n_estimators=60, random_state=1)
	# clf3 = GaussianNB()
	# clf4 = KNeighborsClassifier(n_neighbors=5, algorithm='auto')
	clf5 = SVC(gamma='auto', probability=True)
	clf6 = AdaBoostClassifier(algorithm="SAMME.R", n_estimators=60, random_state=1)
	clf7 = GradientBoostingClassifier()
	eclf = VotingClassifier(estimators=[('lreg', clf1), ('RFC', clf2),
		('svm', clf5),('AdaBoostClassifier', clf6)], voting='soft')
	classifiers = [clf1, clf2, clf5, clf6, clf7, eclf]
	for clf, label in zip(classifiers, ['Logistic Regression','Random Forest', 'SVM','Ada','GBM','Ensemble']):
		data_feature, data_label = randomize(data_feature, data_label)
		scores = cross_validate(clf, data_feature, data_label, cv=n, scoring=['accuracy', 'precision', 'recall', 'f1'], return_train_score=True)
		print("   ", label)
		print("Train Acc",scores["train_accuracy"].mean())
		print("Test Acc",scores["test_accuracy"].mean())
		print("Train Precision", scores["train_precision"].mean())
		print("Test Precision",scores["test_precision"].mean())
		print("Train Recall",scores["train_recall"].mean())
		print("Test Recall",scores["test_recall"].mean())
		print("Train F1", scores["train_f1"].mean())
		print("Test F1",scores["test_f1"].mean())
		print("------------------------------------------------------------")

	data_feature, data_label = randomize(data_feature, data_label)
	partion = 450
	eclf.fit(data_feature[:partion], data_label[:partion])
	e = eclf.predict(data_feature[partion:,])
	print("Precision, Recall F1, score, and support for ensemble on a separate Validation set")
	print(precision_recall_fscore_support(data_label[partion:,], e, average="binary"))

	predicted_labels = eclf.predict(test_features)
	results_arr = list()
	for i in zip(test[0:,0].astype("int32"), predicted_labels):
		results_arr.append(i)
	result = pd.DataFrame(results_arr)
	result.to_csv("submission.csv", columns=[0,1],header=["id", "label"], index=None)

if __name__ == '__main__':
    file_train_data = 'train_features.csv'
    file_train_label = 'train_label.csv'
    file_test_data = 'test_features.csv'
    n = int(input("Cross Validation Fold: "))
    start = time.time()
    main(file_train_data, file_train_label, file_test_data, n)
    done = time.time()
    elapsed = done - start
    print(elapsed)
