import numpy as np
import json

############### TRAIN ####################
# Opening training file and reading contents
heart_train_file = open('heart_train.data','r')
file_contents = ""
if heart_train_file.mode == 'r':
    file_contents = heart_train_file.read()

heart_train_file.close()

# Creating variable to parse and store train data
train_data = np.empty((80,23), dtype=int)

# Initialise row and column counters
i = 0
j = 0

# Read file to parse contents and store in numpy array
for line in file_contents.split('\n'):
    for dt in line.split(","):
        # If y is 0 change to -1
        if j == 0 and dt == '0':
            train_data[i, j] = -1
        else:
            train_data[i, j] = dt
        j += 1
    j = 0
    i += 1

# Train data length and features length
train_data_length = len(train_data)
features_length = len(train_data[0]) - 1

# Function to generate 5 tree structures for all binary output combinations
def generate_trees(attr_arr):
    y = [-1,1]
    dtrees = []
    for i in range(len(y)):
        for j in range(len(y)):
            for k in range(len(y)):
                for l in range(len(y)):
                    dtrees.append({attr_arr[0]: [{1: {attr_arr[1]: [{0: {attr_arr[2]: [{1: y[i]}, {0: y[j]}]}}, {1: y[k]}]}}, {0: y[k]}]})
                    dtrees.append({attr_arr[0]: [{1: {attr_arr[1]: [{0: y[i]}, {1: {attr_arr[2]: [{1: y[j]}, {0: y[k]}]}}]}}, {0: y[l]}]})
                    dtrees.append({attr_arr[0]: [{1: y[i]}, {0: {attr_arr[1]: [{0: y[j]}, {1: {attr_arr[2]: [{1: y[k]}, {0: y[l]}]}}]}}]})
                    dtrees.append({attr_arr[0]: [{1: y[i]}, {0: {attr_arr[1]: [{0: {attr_arr[2]: [{1: y[j]}, {0: y[k]}]}}, {1: y[l]}]}}]})
                    dtrees.append({attr_arr[0]: [{1: {attr_arr[1]: [{0: y[i]}, {1: y[j]}]}}, {0: {attr_arr[2]: [{0: y[k]}, {1: y[l]}]}}]})
    return dtrees

############ GENERATING HYPOTHESIS SPACE ##############
hypo_file = open('hypo_file.data','w')

# Generating decision trees with 3 attribute splits
for i in range(features_length):
    for j in range(features_length):
        for k in range(features_length):
            hypo_file.write(json.dumps(generate_trees([i+1, j+1, k+1])) + "\n")

hypo_file.close()