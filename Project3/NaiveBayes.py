import numpy as np
import pandas as pd

fileName = 'project3_dataset4.txt'
data = pd.read_csv(fileName, sep='\t', header=None, index_col=None)
data.rename(columns={data.columns[-1]: "class"}, inplace=True)

performCrossFold = False
performErrorCorrection = False
# Cross fold validation steps
split_count = 10

# Listing the columns containing  nominal (string) attributes
# Dataframe stores strings as 'object' data type
nominal_attr_idx = [i for i, x in enumerate(data.dtypes) if x == 'object']

# Replacing nominal attributes with corresponding integer values
nominal_attr_map = dict()  # stores nominal attributes index to values mapping
for i in nominal_attr_idx:
    unique_vals = data[i].unique()
    unique_vals.sort();
    val_map = dict()
    for index, j in enumerate(unique_vals):
        data[i] = data[i].replace(j, index)
        val_map[j] = index
    nominal_attr_map[i] = val_map


# data

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
    if performErrorCorrection:
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
            if performErrorCorrection and temp == None:
                # Laplacian error correction for zero probabilities
                prob[j] = prob[j] * (1 / class_stats[j])
            else:
                prob[j] = prob[j] * temp
    return prob


# Method to compute descriptor prior probability (i.e. P(X))
def computeDescPriorProb(training_data, validation_data):
    prob = 1.0
    for i in nominal_attr_idx:
        prob *= (training_data[i].value_counts()[validation_data[i][0]] / len(training_data))

    return prob


# Method that performs P(Ci) * P(X|Ci)
def predictClass(row, mean, std, nominal_probs, class_stats, class_priors, run_type):
    descriptor_probs = computeDescriptorPosterior(row, mean, std, class_stats, nominal_probs)

    probs = np.ones(len(descriptor_probs))
    for index, val in class_priors.iteritems():
        probs[index] *= val * descriptor_probs[index]

    if run_type == 'demo':
        return probs
    else:
        # return the index of the maximum probability
        # index corresponds to class label
        return probs.argmax()


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
def NaiveBayes(training_data, validation_data, run_type):
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
        temp_val = 1 if i == 0 else 0

        nominal_stats = training_data[['class', i, temp_val]].groupby(['class', i]).count()
        z = nominal_stats[temp_val].astype("float")
        r_z = z.reset_index()

        for c in r_z['class'].unique():
            temp_df = r_z[r_z['class'] == c]
            total = sum(temp_df[temp_val])
            for j in temp_df[i]:
                z[c, j] = z[c, j] / total
        nominal_probs.append(z.to_dict())

    # eliminate last column (labels) from the data
    validation_data = validation_data[validation_data.columns[:-1]]
    training_data = training_data[training_data.columns[:-1]]

    if run_type == "demo":
        probs = validation_data.apply(predictClass, axis=1, args=(training_data_mean,
                                                                  training_data_std, nominal_probs,
                                                                  class_stats, class_priors, run_type))
        probs = probs.tolist()[0]
        desc_prior_prob = computeDescPriorProb(training_data, validation_data)
        probs /= desc_prior_prob
        for i in range(len(probs)):
            print("P(H" + str(i) + "|X) = " + str(probs[i]))
        print("Prediced class - " + str(np.array(probs).argmax()))

    if run_type == "normal":
        predicted = validation_data.apply(predictClass, axis=1, args=(training_data_mean,
                                                                      training_data_std, nominal_probs,
                                                                      class_stats, class_priors, run_type))
        return getMetrics(validation_labels, predicted)


if performCrossFold:
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

        accuracy, precision, recall, f1_measure = NaiveBayes(training_data, validation_data, "normal")
        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_measure_list.append(f1_measure)

    #         results.append([str(data_index_arr_split[i][0]) + " - " + str(data_index_arr_split[i][-1]) ,accuracy, precision, recall, f1_measure])

    #     results = pd.DataFrame(results, index=None, columns=["Validation Data", "Accuracy", "Precision", "Recall", "F1-Measure"])
    #     results.to_csv("C:\\Users\\Linus-PC\\Desktop\\CSE601\\Projects\\Project3\\NB_Validation_Results.csv")
    print("Accuracy: ", sum(accuracy_list) / split_count)
    print("Precision: ", sum(precision_list) / split_count)
    print("Recall: ", sum(recall_list) / split_count)
    print("F1-Measure: ", sum(f1_measure_list) / split_count)

else:
    # Demo
    test_data = input('Please input test descriptor : ').split(",")
    for i in nominal_attr_idx:
        test_data[i] = nominal_attr_map.get(i).get(test_data[i])
    test_data.append(-1)  # to add a temporary class value to fit into generalized code
    test_data = pd.DataFrame([test_data], index=None, columns=data.columns)
    NaiveBayes(data, test_data, "demo")


# Test inputs-
# sunny,cool,high,weak
# rain,hot,high,weak (From PPT -> Classification3 slide 15)