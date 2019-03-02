import numpy as np
import math

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

# Recursive traversal through the tree for prediction
def recursive_predict(datarow, d_tree):

    for attr_index, attr_values in d_tree.items():
        for i in range(len(attr_values)):
            if datarow[attr_index] == next(iter(attr_values[i])):
                if isinstance(attr_values[i][next(iter(attr_values[i]))], int):
                    return attr_values[i][datarow[attr_index]]
                elif isinstance(attr_values[i][next(iter(attr_values[i]))], dict):
                    return recursive_predict(datarow, attr_values[i][next(iter(attr_values[i]))])

# Function computes alpha value
def compute_alpha(dataset, t_number, alphas, d_trees):

    num_sum = 0.0
    den_sum = 0.0

    # Iterating every data point
    for ii in range(len(dataset)):
        tree_sum = 0.0

        # Computing inner sum for all trees except the selected tree
        for t in range(len(d_trees)):
            if t == t_number:
                continue
            tree_sum += alphas[t] * recursive_predict(dataset[ii], dtrees[t])

        tree_sum *= -dataset[ii,0]

        # Checking if the selected tree classifies the dataset correctly
        if recursive_predict(dataset[ii], d_trees[t_number]) == dataset[ii,0]:
            num_sum += math.exp(tree_sum)
        else:
            den_sum += math.exp(tree_sum)

    return 0.5 * math.log(num_sum/den_sum)

# Compare arrays and return false if not same
def compare_arrays(array1, array2):

    for jj in range(len(array1)):
        if abs(array1[jj] - array2[jj]) > 0.001:
            return False

    return True

# Predicting values and computing accuracy
def accuracy(alpha_arr, dt_arr, dataset):

    predict_array = []

    # Predicting values using alpha and set of weak learners
    for i in range(len(dataset)):
        calc_sum = 0.0
        for j in range(len(alpha_arr)):
            calc_sum += alpha_arr[j] * recursive_predict(dataset[i], dt_arr[j])

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

################### DECISION TREES #####################
dtrees = []

# Iterating to produce all trees of height 1
for i in range(features_length):
    # Possible outputs
    y = [-1,1]

    # Iterating to produce all combinations of trees for all outputs
    for j in range(len(y)):
        for k in range(len(y)):

            # Building decision tree for each feature list of length 3 and compute its weighted error
            tree = {(i+1):[{0:y[j]},{1:y[k]}]}
            dtrees.append(tree)

################### CO-ORDINATE DESCENT ################
iteration_counter = 0

# initialise alpha array and hypothesis space
alpha_array = 0.3 * np.ones((88,1))

while iteration_counter >= 0:

    prev_alpha_array = alpha_array.copy()

    # Update all trees in round robin instead of random
    for t in range(len(dtrees)):
        loss = 0.0

        # Computing loss
        for i in range(train_data_length):
            tree_sum_loss = 0.0
            tree_counter = 0

            for tree in dtrees:
                tree_sum_loss += alpha_array[tree_counter] * recursive_predict(train_data[i], tree)
                tree_counter += 1

            loss += math.exp(-train_data[i,0] * tree_sum_loss)

        alpha_array[t] = compute_alpha(train_data, t, alpha_array, dtrees)

    if compare_arrays(prev_alpha_array, alpha_array):
        break

    iteration_counter += 1

print("Iteration number:", iteration_counter, "Loss:", loss, "Alpha:", alpha_array)
print("Accuracy on test data:", accuracy(alpha_array, dtrees, test_data))