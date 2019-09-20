import numpy as np
import pandas as pd

# Reading data from file and appending it with gene identifiers

file_data = pd.read_csv('associationruletestdata.txt', sep='\t', header=None,
                        index_col=None)

record_count = file_data.shape[0]  # number of rows
attribute_count = file_data.shape[1]  # number of columns

# Changing the column index to start from 1 instead of 0
file_data.columns = np.arange(1, attribute_count + 1)

for i in range(1, file_data.shape[1]):
    file_data[i] = 'G' + str(i) + '_' + file_data[i]

# Converting the input data into an array of sample sets
# Converted data to sets for intersection operation while matching itemsets

# Converting input data table into array of arrays, eliminating the last column (disease)
file_data_arr = file_data.values[:, :attribute_count]

sample_sets = []
for i in file_data_arr:
    sample_sets.append(set(i))

# Includes all column data.
raw_set = []

for i in file_data_arr:
    for j in i:
        if {j} not in raw_set:
            raw_set.append({j})


def ap_rules(freq_k_item, H1, freq_item_support_map):
    k = len(freq_k_item)
    m = len(H1)
    if(k > m+1):
        Hmplus1 = generate_merge_sets(H1,m+1)
        for hplus1 in Hmplus1:
            confidence = freq_item_support_map[freq_k_item]/freq_item_support_map[freq_k_item.difference(hplus1)]
            if confidence >= minconfidence:
                print (repr(freq_k_item.difference(hplus1))+" implies "+repr(hplus1))
            else:
                Hmplus1.remove(hplus1)
        ap_rules(freq_k_item, Hmplus1)

def generate_rules(confidence, freq_item_sets, freq_item_support_map):
    for freq_k_item, freq_k_item_sup in freq_item_sets:
        H1= set(freq_k_item)
        ap_rules(freq_k_item, H1, freq_item_support,)

def has_unique_last_items(a, b, length):
    first = sorted(a)
    second = sorted(b)

    i = 1
    for x, y in zip(first, second):
        if i == length - 1 and x != y:
            return True
        if i == length - 1 and x == y:
            return False
        if x != y:
            return False
        i += 1
    return True


# https://www-users.cs.umn.edu/~kumar001/dmbook/ch6.pdf
# Using F(k-1) * F(k-1) method
def generate_merge_sets(freq_sets, set_length):
    new_item_sets = []

    for i in range(len(freq_sets)):
        for j in range(i, len(freq_sets)):
            if set_length < 2 or has_unique_last_items(freq_sets[i], freq_sets[j], set_length):
                union_set = freq_sets[i].union(freq_sets[j])
                if len(union_set) == set_length:
                    sorted_set = set(sorted(union_set))
                    new_item_sets.append(sorted_set)
    return new_item_sets


def main():
    min_support_values = [30, 40, 50, 60, 70]

    for min_support in min_support_values:
        print('\nSupport is set to be ' + str(min_support) + '%')

        flag = True  # Flag used to determine when to stop
        length = 1  # length of frequent item sets

        item_sets = raw_set
        sum = 0
        freq_item_support_map = dict()
        while flag:
            item_set_support = []  # Temporary list to store support values of current item sets
            for item_set in item_sets:
                count = 0
                for sample in sample_sets:
                    if len(sample.intersection(item_set)) == len(item_set):
                        count += 1
                sup = round((count * 100 / record_count))  # TODO: Verify this
                # sup = count
                item_set_support.append(sup)

            freq_item_sets = []
            
            for index, sup in enumerate(item_set_support):
                if sup >= min_support:
                    freq_item_sets.append(item_sets[index])
                    freq_item_support_map[list()] = sup

            generate_rules(confidence, frequent_item_sets, freq_item_support_map)

            sum += len(freq_item_sets)
            if len(freq_item_sets) != 0:
                print('number of length-' + str(length) + ' frequent itemsets: ' + str(len(freq_item_sets)))
            else:
                print('number of all lengths frequent itemsets:' + str(sum))
                sum = 0

            if len(freq_item_sets) == 0:
                flag = False;
            else:
                length += 1
                item_sets = generate_merge_sets(freq_item_sets, length)


if __name__ == '__main__':
    main()
