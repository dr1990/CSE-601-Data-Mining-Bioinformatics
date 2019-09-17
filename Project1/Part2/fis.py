import numpy as np
import pandas as pd
import math

# Reading data from file and appending it with gene identifiers

file_data = pd.read_csv('/Users/deepak/Desktop/601/Project1/Part2/associationruletestdata.txt', sep='\t', header=None,
                        index_col=None)

record_count = file_data.shape[0]  # number of rows
attribute_count = file_data.shape[1]  # number of columns
print(str(record_count) + " records", str(attribute_count) + " attributes")

# Changing the column index to start from 1 instead of 0
file_data.columns = np.arange(1, attribute_count + 1)

for i in range(1, file_data.shape[1]):
    file_data[i] = 'G' + str(i) + '_' + file_data[i]

# Converting the input data into an array of sample sets
# Converted data to sets for intersection operation while matching itemsets

# Converting input data table into array of arrays, eliminating the last column (disease)
file_data_arr = file_data.values[:, :attribute_count]
# print(file_data_arr.shape)

sample_sets = []
for i in file_data_arr:
    sample_sets.append(set(i))

raw_set = []

for i in file_data_arr:
    for j in i:
        if {j} not in raw_set:
            raw_set.append({j})


# temp = set(item_sets)
# item_sets = []
# item_set = list(temp)


def get_unique_items(freq_sets):
    items = []
    for individual_set in freq_sets:
        for item in individual_set:
            items.append(item)

    return list(set(items))


def has_one_unique_items(first, second, len):
    if len <= 2:
        return True
    i = 1
    for x, y in zip(first, second):
        if i >= len - 1:
            return True
        if x != y:
            return False
        i += 1
    return True


def generateMergeSets(freq_sets, rej_sets, set_length):
    new_item_sets = []
    temp = set()
    #
    # for a in rej_sets:
    #     for b in a:
    #         temp.add(b)
    #
    # print(len(temp))
    for i in range(len(freq_sets)):
        for j in range(i, len(freq_sets)):
            if set_length < 2 or has_one_unique_items(freq_sets[i], freq_sets[j], set_length):
                union_set = freq_sets[i].union(freq_sets[j])
                if len(union_set) == set_length:
                    # flag = False
                    # for k in range(len(rej_sets)):
                    #     if len(union_set.intersection(rej_sets[k])) == len(rej_sets[k]):
                    #         flag = True
                    #         break
                    #
                    # if not flag:
                    sorted_set = set(sorted(union_set))
                    new_item_sets.append(sorted_set)
    return new_item_sets


# item_sets = []

# for i in range(1, attribute_count):
#     item_sets.append({'G' + str(i) + '_Up'})
#     item_sets.append({'G' + str(i) + '_Down'})


min_support_values = [30, 40, 50, 60, 70]

for min_support in min_support_values:
    print('-------------- Support ' + str(min_support) + ' --------------')

    flag = True  # Flag used to determine when to stop
    length = 1  # length of frequent item sets

    item_sets = raw_set

    while flag:
        item_set_support = []  # Temporary list to store support values of current item sets
        for item_set in item_sets:
            count = 0
            for sample in sample_sets:
                if len(sample.intersection(item_set)) == len(item_set):
                    count += 1
            sup = math.floor((count / record_count) * 100)
            item_set_support.append(sup)

        freq_item_sets = []
        rejected_item_sets = []
        for index, sup in enumerate(item_set_support):
            if sup >= min_support:
                freq_item_sets.append(item_sets[index])
            else:
                rejected_item_sets.append(item_sets[index])

        print('number of length-' + str(length) + ' frequent itemsets: ' + str(len(freq_item_sets)))

        if len(freq_item_sets) == 0:
            flag = False;
        else:
            length += 1
            item_sets = generateMergeSets(freq_item_sets, rejected_item_sets, length)
#             print('Length of new item set - ' + str(len(item_sets)))
