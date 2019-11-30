import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA



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

		
def main(file_train_data, file_train_labels, file_test_data):
	data = readfile(file_train_data, None)
	labels = readfile(file_train_labels, 0)
	data_feature = np.reshape(data[:, 1:], np.shape(data[:, 1:]))
	data_label = np.reshape(labels[:, 1], (-1, 1))
	data_label = np.reshape(data_label, (data_label.shape[0],))
	test = readfile(file_test_data, header=None)
	test_features = np.reshape(test[:, 1:], np.shape(test[:, 1:]))
	# print(data_feature.shape,data_label.shape)
	# print(test_features)
	# LR, KNN was performing weak so scaled the data (Scaling required for distance based algo)
	scaler = StandardScaler()
	scaler.fit(data_feature)
	data_feature = scaler.transform(data_feature)
	test_features = scaler.transform(test_features)

	print(data_feature.shape)
	classifiers = list()
	clf1 = LogisticRegression(solver='liblinear', random_state=1)
	# clf2 = RandomForestClassifier(n_estimators=100, random_state=1)
	# clf3 = GaussianNB()
	clf4 = KNeighborsClassifier(n_neighbors=3, algorithm='auto')
	clf5 = SVC(gamma='auto')
	# clf6 = AdaBoostClassifier(n_estimators=100, random_state=1)
	eclf = VotingClassifier(estimators=[('lreg', clf1),  
		('knn', clf4), ('svm', clf5),], voting='hard')
	classifiers = [clf1, clf4, clf5, eclf]
	for clf, label in zip(classifiers, ['Logistic Regression', 'K Neighbors', 'SVM','Ensemble']):
		scores = cross_val_score(clf, data_feature, data_label, cv=10, scoring='accuracy')
		print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

	eclf.fit(data_feature, data_label)
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
    n = 10
    start = time.time()
    main(file_train_data, file_train_label, file_test_data)
    done = time.time()
    elapsed = done - start
    print(elapsed)

#RESULTS:

# Best Combination Achieved

# Accuracy: 0.82 (+/- 0.09) [Logistic Regression]
# Accuracy: 0.80 (+/- 0.03) [Random Forest]
# Accuracy: 0.83 (+/- 0.12) [K Neighbors]
# Accuracy: 0.84 (+/- 0.07) [SVM]
# Accuracy: 0.88 (+/- 0.07) [Ensemble]

# Accuracy: 0.81 (+/- 0.09) [Logistic Regression]
# Accuracy: 0.80 (+/- 0.03) [Random Forest]
# Accuracy: 0.80 (+/- 0.12) [K Neighbors]
# Accuracy: 0.87 (+/- 0.07) [Ensemble]


# Accuracy: 0.81 (+/- 0.09) [Logistic Regression]
# Accuracy: 0.80 (+/- 0.03) [Random Forest]
# Accuracy: 0.62 (+/- 0.10) [Naive Bayes]
# Accuracy: 0.80 (+/- 0.12) [K Neighbors]
# Accuracy: 0.79 (+/- 0.11) [Ensemble]

#Before Scaling

# Accuracy: 0.65 (+/- 0.08) [Logistic Regression]
# Accuracy: 0.80 (+/- 0.03) [Random Forest]
# Accuracy: 0.69 (+/- 0.14) [Naive Bayes]
# Accuracy: 0.74 (+/- 0.06) [K Neighbors]
# Accuracy: 0.74 (+/- 0.08) [Ensemble]

