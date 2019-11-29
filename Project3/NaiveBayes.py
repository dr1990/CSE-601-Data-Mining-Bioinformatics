import numpy as np
import pandas as pd

fileName = 'project3_dataset2.txt'
data = pd.read_csv(fileName, sep='\t', header=None, index_col=None)
data.rename(columns={data.columns[-1]: "class"}, inplace=True)

# Cross fold validation steps
split_count = 10

# Listing the columns containing  nominal (string) attributes
# Dataframe stores strings as 'object' data type
nominal_attr_idx = [i for i, x in enumerate(data.dtypes) if x == 'object']

# Replacing nominal attributes with corresponding integer values
for i in nominal_attr_idx:
    unique_vals = data[i].unique()
    unique_vals.sort();
    for index, j in enumerate(unique_vals):
        data[i] = data[i].replace(j, index)


# print(data)

# Method that calculates description posterior in the Naive Bayes formula i.e P(X|Hi)
# row - A row from the validation data set passed by DataFrame.apply()
# mean - Column-wise mean of training data set  [2 x *]
# std - Column-wise standard deviation of training data set [2 x *]
# class_stats - Row counts of class labels in the data set [2 x 1]
# nominal_probs - Class-based probabilites of nominal attributes
def computeDescriptorPosterior(row, mean, std, class_stats, nominal_probs):
    row_values = row.values

    num = np.square(row_values - mean)  # numerator
    den = 2 * np.square(std)  # denominator
    constant = 1 / np.sqrt(np.pi * den)

    # Gaussian probability
    prob = constant * np.exp(-num / (den))

    # Laplacian error correction for zero probabilities
    for i in range(prob.shape[0]):
        for j in range(prob.shape[1]):
            if prob[i][j] == 0:
                prob[i][j] = 1 / class_stats[i]

    # removing the nominal attribute probability from the computed results
    # not eliminated earlier for ease of computation
    column_identifiers = np.zeros_like(row_values, dtype=np.bool)
    column_identifiers[nominal_attr_idx] = True
    prob = prob[:, ~column_identifiers]

    # continuous attributes probabilities
    # multiply all computed attribute probabilites for both classes (along axis 1)
    prob = prob.prod(axis=1)

    # computing nominal attribute probabilities
    # and multiplying with computed continuous attribute probabilities
    nominal_descriptors = row_values[column_identifiers]
    for i in range(len(nominal_attr_idx)):
        nominal_prob_dict = nominal_probs[i]
        for j in range(len(prob)):
            temp = nominal_prob_dict.get((j, nominal_descriptors[i]))
            if temp == None:
                # Laplacian error correction for zero probabilities
                prob[j] = prob[j] * (1 / class_stats[j])
            else:
                prob[j] = prob[j] * temp
    return prob


# Method that performs P(Ci) * P(X|Ci)
def predictClass(row, mean, std, nominal_probs, class_stats, class_priors):
    descriptor_probs = computeDescriptorPosterior(row, mean, std, class_stats, nominal_probs)

    probs = np.multiply(class_priors, descriptor_probs)

    # return the index of the maximum probability
    # index corresponds to class label
    return probs.idxmax()


# Method to compute the metrics (accuracy, precision, recall, f1 measure) for the given
# labels of the validation data set and the labels predicted by Naive Bayes algorithm
# actual_labels - Known class labels of the validation data set
# predicted_labels - Class labels of validation data set predicted by Naive Bayes algorithm
def getMetrics(actual_labels, predicted_labels):
    confusion_matrix = np.zeros(4).reshape(-1, 2)
    for i, j in zip(actual_labels, predicted_labels):
        confusion_matrix[i][j] += 1

    a = confusion_matrix[0][0]
    b = confusion_matrix[0][1]
    c = confusion_matrix[1][0]
    d = confusion_matrix[1][1]

    accuracy = (float)(a + d) / (float)(a + b + c + d)
    precision = (float)(a) / (float)(a + c)
    recall = (float)(a) / (float)(a + b)
    f1_measure = (float)(2 * a) / (float)(2 * a + b + c)

    return accuracy, precision, recall, f1_measure


# Method that invokes the steps in Naive Bayes algorithm
def NaiveBayes(training_data, validation_data):
    validation_labels = validation_data["class"]
    training_labels = training_data["class"]

    class_stats = training_labels.value_counts()
    class_priors = class_stats / np.sum(class_stats)  # P(Ci)

    training_data_mean = training_data.groupby('class').mean().values
    training_data_std = training_data.groupby('class').std().values

    # Calculating class based probabilities for nominal attributes
    nominal_probs = list()
    for i in nominal_attr_idx:
        # temporary value computed to ensure group by operation in next step
        # does not fail if dataset contains more than one nominal attribute
        temp_val = 0 if i == 0 else 1

        nominal_stats = data[['class', i, temp_val]].groupby(['class', i]).count()
        z = nominal_stats[temp_val].astype("float")
        for i in z.index.get_level_values(0).unique():
            total = sum(z[i])
            for j in range(len(z[i])):
                z[i][j] = z[i][j] / total
        nominal_probs.append(z.to_dict())

    # eliminate last column (labels) from the data
    validation_data = validation_data[validation_data.columns[:-1]]
    training_data = training_data[training_data.columns[:-1]]

    predicted = validation_data.apply(predictClass, axis=1, args=(training_data_mean,
                                                                  training_data_std, nominal_probs,
                                                                  class_stats, class_priors))

    return getMetrics(validation_labels, predicted)


data_index_arr = np.arange(len(data))
data_index_arr_split = np.array_split(data_index_arr, split_count)

accuracy_list = list()
precision_list = list()
recall_list = list()
f1_measure_list = list()

results = list()

for i in range(split_count):
    validation_data = data.loc[data_index_arr_split[i]]
    training_data_idx = np.hstack([x for j, x in enumerate(data_index_arr_split) if j != i])
    training_data = data.loc[training_data_idx]

    accuracy, precision, recall, f1_measure = NaiveBayes(training_data, validation_data)
    accuracy_list.append(accuracy)
    precision_list.append(precision)
    recall_list.append(recall)
    f1_measure_list.append(f1_measure)

    results.append(
        [str(data_index_arr_split[i][0]) + " - " + str(data_index_arr_split[i][-1]), accuracy, precision, recall,
         f1_measure])

results = pd.DataFrame(results, index=None,
                       columns=["Validation Data", "Accuracy", "Precision", "Recall", "F1-Measure"])
results.to_csv("NB_Validation_Results.csv")
print("Accuracy: ", sum(accuracy_list) / split_count)
print("Precision: ", sum(precision_list) / split_count)
print("Recall: ", sum(recall_list) / split_count)
print("F1-Measure: ", sum(f1_measure_list) / split_count)