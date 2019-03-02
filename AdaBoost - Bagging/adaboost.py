import numpy as np
import math
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

############### TEST ####################
# Opening test file and reading contents
heart_test_file = open('heart_test.data','r')
file_contents = ""
if heart_test_file.mode == 'r':
    file_contents = heart_test_file.read()

heart_test_file.close()

# Creating variable to parse and store test data
test_data = np.empty((187,23), dtype=int)

# Initialise row and column counters
i = 0
j = 0

# Read file to parse contents and store in numpy array
for line in file_contents.split('\n'):
    for dt in line.split(","):
        # If y is 0 change to -1
        if j == 0 and dt == '0':
            test_data[i, j] = -1
        else:
            test_data[i, j] = dt
        j += 1
    j = 0
    i += 1

# Data weights array
data_weights = np.ones((train_data_length,1), dtype=float)

# Initialise data weights array with data length
for i in range(train_data_length):
    data_weights[i,0] = 1/train_data_length

# number of boosting rounds
boosting_rounds = 10

# Recursive traversal through the tree for prediction
def recursive_predict(datarow, d_tree):

    for attr_index, attr_values in d_tree.items():
        for i in range(len(attr_values)):
            if datarow[int(attr_index)] == int(next(iter(attr_values[i]))):
                if isinstance(attr_values[i][next(iter(attr_values[i]))], int):
                    return attr_values[i][next(iter(attr_values[i]))]
                elif isinstance(attr_values[i][next(iter(attr_values[i]))], dict):
                    return recursive_predict(datarow, attr_values[i][next(iter(attr_values[i]))])

# Computes weighted error for a dataset and a given hypothesis
def compute_weighted_error(d_tree, dataset, weights):

    predict_array = []
    for i in range(len(dataset)):
        rp = recursive_predict(dataset[i], d_tree)
        if rp not in [1,-1]:
            print(rp)
        predict_array.append(rp)

    index = 0
    inner_weighted_error = 0.0
    for val1 in predict_array:
        if val1 != dataset[index][0]:
            inner_weighted_error += weights[index]
        index += 1

    return inner_weighted_error

def accuracy(alpha_arr, dt_arr, dataset):

    predict_array = []

    # Predicting values using alpha and set of weak learners
    for i in range(len(dataset)):
        calc_sum = 0.0
        for j in range(len(alpha_arr)):
            rp = recursive_predict(dataset[i], dt_arr[j])
            if rp not in [1, -1]:
                print(rp)
            calc_sum += alpha_arr[j] * rp

        if calc_sum <= 0:
            predict_array.append(-1)
        else:
            predict_array.append(1)

    # Computing accuracy
    counter = 0
    index = 0
    for val1 in predict_array:
        if val1 == dataset[index][0]:
            counter += 1
        index += 1

    return counter / len(dataset)

########################### BEGIN ADA BOOST ################################
if __name__ == '__main__':
    # Initialise alpha and hypothesis (decision tree) arrays
    alpha_array = []
    dt_array = []

    # Contains iteration number, training and testing accuracies
    itr_train_test_acc = []

    # Start boosting iterations
    for round_no in range(boosting_rounds):

        # Setting initial values for error and tree
        min_weighted_error = 99999.0
        min_decision_tree = {}

        # Iterating to produce all features of length 3
        with open("hypo_file.data") as hypo_file:
            for line in hypo_file:

                # Reading and parsing each json line
                trees = json.loads(line)

                # Building decision tree for each feature list of length 3 and compute its weighted error
                for tree in trees:

                    weighted_error = compute_weighted_error(tree, train_data, data_weights)

                    # Find the least weighted error
                    if weighted_error <= min_weighted_error:
                        min_weighted_error = weighted_error
                        min_decision_tree = tree

        hypo_file.close()

        # Compute alpha
        alpha = 0.5 * math.log((1 - min_weighted_error)/min_weighted_error)

        alpha_array.append(alpha)
        dt_array.append(min_decision_tree)

        itr_train_test_acc.append([(round_no+1), accuracy(alpha_array, dt_array, train_data), accuracy(alpha_array, dt_array, test_data)])

        # Update weights
        for k in range(train_data_length):
            data_weights[k] = (data_weights[k] * math.exp(-train_data[k][0] * recursive_predict(train_data[k], min_decision_tree) * alpha)) / (2 * math.sqrt(min_weighted_error * (1 - min_weighted_error)))

        # Printing alpha, weighted error and tree
        print("For iteration:",(round_no+1),", alpha:",alpha,"| minimum weighted error:",min_weighted_error,"| decision tree:",min_decision_tree)

    print("Accuracy for each iteration:", itr_train_test_acc)