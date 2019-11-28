import pandas as pd
from sklearn.naive_bayes import GaussianNB

train_features = pd.read_csv("train_features.csv", sep=',', header=None, index_col=0)
train_labels = pd.read_csv("train_label.csv", sep=',', header=0, index_col=0)
test_features = pd.read_csv("test_features.csv", sep=',', header=None, index_col=0)
# print(train_features)
# print(train_labels)
# print(test_features)
# train_labels['label'].values

gnb = GaussianNB()
predictions = gnb.fit(train_features, train_labels['label'].values).predict(test_features)
# print(predictions)

results_arr = list()
for i in zip(test_features.index, predictions):
    results_arr.append(i)
result = pd.DataFrame(results_arr)
result.to_csv("test_labels.csv", columns=[0,1],header=None, index=None)