import numpy as np
import pandas as pd

global final_dict
global fis_map
global fis_support_map

final_dict = dict()        # Final data structure that holds the query results
fis_support_map = dict()   # Data structure that holds all the support values computed for the identified frequent item sets
fis_map = dict()           # Dictionary that stores all frequent itemsets (value) for a given length (key)

# Reading data from file and appending it with gene identifiers
file_data = pd.read_csv('associationruletestdata.txt', sep='\t', header=None, index_col=None)

record_count = file_data.shape[0]  # number of rows
attribute_count = file_data.shape[1]  # number of columns

# Changing the column index to start from 1 instead of 0
file_data.columns = np.arange(1, attribute_count + 1)

for i in range(1, file_data.shape[1]):
    file_data[i] = 'G' + str(i) + '_' + file_data[i]

# Converting the input data into an array of sample sets
# Converted data to sets for intersection operation while matching itemsets
# Converting input data table into array of arrays
file_data_arr = file_data.values[:, :attribute_count]

sample_sets = []
for i in file_data_arr:
    sample_sets.append(set(i))

# Includes all column data - unique items in the whole data set
raw_set = []
for i in file_data_arr:
    for j in i:
        if {j} not in raw_set:
            raw_set.append({j})


# Checks for the uniqueness for the last column in both sets.
def has_unique_last_items(set_a, set_b, length):
    first = sorted(set_a)
    second = sorted(set_b)

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
                    new_item_sets.append(union_set)
    return new_item_sets

# Method to generate frequent item sets from the initial items list, for the given minimum support values
def main(support):
    min_support_values = []
    min_support_values.extend(support)

    for min_support in min_support_values:
        print('\nSupport is set to be ' + str(min_support) + '%')

        flag = True  # Flag used to determine when to stop
        length = 1  # length of frequent item sets

        item_sets = raw_set
        sum = 0
        while flag:
            item_set_support = []  # Temporary list to store support values of current item sets
            for item_set in item_sets:
                count = 0
                for sample in sample_sets:
                    if len(sample.intersection(item_set)) == len(item_set):
                        count += 1
                sup = (count * 100 / record_count)
                item_set_support.append(sup)

            freq_item_sets = []
            for index, sup in enumerate(item_set_support):
                if sup >= min_support:
                    freq_item_sets.append(item_sets[index])
                    fis_support_map[getStr(sorted(item_sets[index]))] = sup

            fis_map[str(length)] = freq_item_sets
            sum += len(freq_item_sets)
            if len(freq_item_sets) != 0:
                print('Number of length-' + str(length) + ' frequent itemsets: ' + str(len(freq_item_sets)))
            else:
                print('Number of all lengths frequent itemsets: ' + str(sum))
                sum = 0

            if len(freq_item_sets) == 0:
                flag = False
            else:
                length += 1
                item_sets = generate_merge_sets(freq_item_sets, length)


# Generate all possible rules for k-itemsets where k < len(res)
def merge(res, confidence):
    if len(res) != 0:
        new_res = dict()
        for k1, v1 in res.items():
            key_set1 = set(k1.split("|"))
            val_set1 = set(v1.split("|"))
            if len(key_set1) > 1:
                for k2, v2 in res.items():
                    if k1 != k2:
                        key_set2 = set(k2.split("|"))
                        val_set2 = set(v2.split("|"))

                        k1k2 = key_set1.intersection(key_set2)
                        v1v2 = val_set1.union(val_set2)

                        if len(k1k2) == len(key_set1) - 1:
                            item_set = k1k2.union(v1v2)
                            diff = k1k2
                            nom = fis_support_map.get(getStr(sorted(item_set)))
                            denom = fis_support_map.get(getStr(sorted(diff)))
                            if nom is None or denom is None:
                                pass
                            else:
                                conf = ((nom / denom) * 100)

                            if conf >= confidence:
                                new_res[getStr(sorted(k1k2))] = getStr(sorted(v1v2))
                                # Key is messed up in order to make it unique.
                                # getStr(sorted(k1k2)) is the only key. Rest everything is crap.
                                final_dict[getStr(sorted(k1k2)) + ";" + getStr(sorted(v1v2)) + str(conf)] = getStr(
                                    sorted(v1v2))

        merge(new_res, confidence)


# Method to format item sets into a string to be stored as keys in in a map
def getStr(lst):
    out = ''
    for v in lst:
        out += v + "|"

    return out.strip("|")


def gen(fk, confidence):
    single_items = list(fk)  # Converting the frequent item set to a list of individual items
    fk_set = fk
    res = dict()
    for item in single_items:
        item_set = {item}
        diff = fk_set.difference(item_set)
        nom = fis_support_map.get(getStr(sorted(fk)))
        denom = fis_support_map.get(getStr(sorted(diff)))
        if nom is None or denom is None:
            pass
        else:
            conf = ((nom / denom) * 100)

            if conf >= confidence:
                res[getStr(sorted(diff))] = item
                final_dict[getStr(sorted(diff)) + ";" + item + str(conf)] = item
    merge(res, confidence)

# Method that initiates rule generation from the stored frequent item sets of different lengths
def generate_rules(confidence):
    for k, v in fis_map.items():
        if k != '1':    # No rules can be generated for length-1 frequent item sets
            for freq_item_set in v:
                gen(freq_item_set, confidence)

    print("\nNumber of rules generated for support " + str(support) + "% and confidence " + str(confidence) + "% - ", len(final_dict))

# Method to evaluate user queries
def evaluate_query():
    template_no = int(input("Enter template number: "))

    if template_no == 1:
        query = input('Enter template-1 query (RULE|BODY|HEAD;ANY|NUMBER|NONE;ITEM1,ITEM2,...): ').split(';')
        result, cnt = temp_1(query[0], query[1], query[2])
    elif template_no == 2:
        query = input('Enter template-2 query (RULE|BODY|HEAD;NUMBER): ').split(';')
        result, cnt = temp_2(query[0], query[1])
    elif template_no == 3:
        query = input('Enter template-3 query (1or1;HEAD;ANY;G10_Down;BODY;1;G59_UP): ').split(';')

        result1 = []
        result2 = []
        res = []
        if query[0][1:3] == 'or':
            if query[0][0] == '1' and query[0][3] == '1':
                result1, cnt1 = temp_1(query[1], query[2], query[3])
                result2, cnt2 = temp_1(query[4], query[5], query[6])

            elif query[0][0] == '1' and query[0][3] == '2':
                result1, cnt1 = temp_1(query[1], query[2], query[3])
                result2, cnt2 = temp_2(query[4], query[5])

            elif query[0][0] == '2' and query[0][3] == '1':
                result1, cnt1 = temp_2(query[1], query[2])
                result2, cnt2 = temp_1(query[3], query[4], query[5])

            elif query[0][0] == '2' and query[0][3] == '2':
                result1, cnt1 = temp_2(query[1], query[2])
                result2, cnt2 = temp_2(query[3], query[4])

            res = set(result1).union(set(result2))

        elif query[0][1:4] == 'and':
            if query[0][0] == '1' and query[0][4] == '1':
                result1, cnt1 = temp_1(query[1], query[2], query[3])
                result2, cnt2 = temp_1(query[4], query[5], query[6])

            elif query[0][0] == '1' and query[0][4] == '2':
                result1, cnt1 = temp_1(query[1], query[2], query[3])
                result2, cnt2 = temp_2(query[4], query[5])

            elif query[0][0] == '2' and query[0][4] == '1':
                result1, cnt1 = temp_2(query[1], query[2])
                result2, cnt2 = temp_1(query[3], query[4], query[5])

            elif query[0][0] == '2' and query[0][4] == '2':
                result1, cnt1 = temp_2(query[1], query[2])
                result2, cnt2 = temp_2(query[3], query[4])

            res = set(result1).intersection(set(result2))

        count = len(res)
        print(res)
        print("Total rules count: ", count)


# Answers Query for template-1
def temp_1(a, b, c):
    cnt = 0
    rules = []
    items = set(c.split(","))
    for k, v in final_dict.items():
        key = set(k.split(";")[0].split("|"))
        val = set(v.split("|"))

        if a == 'RULE':
            common_set = set(items).intersection(key.union(val))
        elif a == 'BODY':
            common_set = set(items).intersection(key)
        elif a == 'HEAD':
            common_set = set(items).intersection(val)

        if b == 'ANY':
            if common_set == items:
                cnt += 1
                rules.append(','.join(sorted(val)) + "->" + ','.join(sorted(key)))
        elif b == 'NONE':
            if len(common_set) == 0:
                cnt += 1
                rules.append(','.join(sorted(val)) + "->" + ','.join(sorted(key)))
        else:
            num = int(b)
            if len(common_set) == num:
                cnt += 1
                rules.append(','.join(sorted(val)) + "->" + ','.join(sorted(key)))

    print("\nTemplate 1 Query: ", a + ";" + b + ";" + c, "\nRule Count: ", cnt)
    print(rules)
    return rules, cnt


# Answers Query for template-2
def temp_2(a, b):
    cnt = 0
    rules = []

    for k, v in final_dict.items():
        set_cnt = 0
        key = set(k.split(";")[0].split("|"))
        val = set(v.split("|"))

        if a == 'RULE':
            set_cnt = len(key.union(val))
        elif a == 'BODY':
            set_cnt = len(key)
        elif a == 'HEAD':
            set_cnt = len(val)

        if set_cnt >= int(b):
            rules.append(','.join(sorted(val)) + "->" + ','.join(sorted(key)))
            cnt += 1

    print("\nTemplate 2 Query: ", a + ";" + b, "\nCount: ", cnt)
    print(rules)
    return rules, cnt


# Starting point of the program
if __name__ == '__main__':
    while True:
        task = int(input("\nTask (1 or 2):"))
        if task == 1:
            main([30, 40, 50, 60, 70])
        elif task == 2:
            support = int(input("Enter Support Value: "))
            confidence = int(input("Enter Confidence Value: "))

            final_dict = dict()
            fis_support_map = dict()
            fis_map = dict()
            main([support])
            generate_rules(confidence)
            evaluate_query()
        else:
            print("Invalid input")

        continueExec = input("\nContinue? (Y or N)")
        if continueExec != 'Y':
            break