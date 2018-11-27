import numpy as np
import random

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

# Computing entropy and information gain for given attributes
def bestAttributeSelector(dataset, attributes_used):

    entropy_map = {}

    # Computing number of -1 and 1 in dataset
    for i in range(len(dataset)):
        if dataset[i,0] not in entropy_map:
            entropy_map[dataset[i,0]] = 0.0
        entropy_map[dataset[i,0]] += 1.0

    # Computing probability of -1 and 1 in dataset and computing entropy of Y = (1,-1)
    entropy_y = 0.0
    for key, value in entropy_map.items():
        entropy_y -= (value/len(dataset)) * np.log2(value/len(dataset))

    # Computing number of values for each attribute in the dataset and calculating entropy for each y over x
    entropy_attr_y_map = {}

    for a in range(1, features_length+1):
        entropy_attr_map = {}

        if a in attributes_used:
            continue

        # Values of all features
        x = [0, 1]

        # Counting attribute values and their respective 1 and -1 counts
        for i in range(len(dataset)):
            for j in range(len(x)):
                if dataset[i][a] == x[j]:
                    # Check if attribute value is in map
                    if x[j] not in entropy_attr_map:
                        entropy_attr_map[x[j]] = {}
                        entropy_attr_map[x[j]]['total'] = 0.0

                    # Check if p or e is in attribute value sub map
                    if dataset[i][0] not in entropy_attr_map[x[j]]:
                        entropy_attr_map[x[j]][dataset[i][0]] = 0.0

                    # Increment total and p or e values
                    entropy_attr_map[x[j]][dataset[i][0]] += 1.0
                    entropy_attr_map[x[j]]['total'] += 1.0
                    break

        entropy_attr_y = 0.0

        # Computing probabilities of y given x and their entropy
        for key, value in entropy_attr_map.items():
            p_x = value['total']/len(dataset)
            p_y = 0.0

            # Computing inner sum of y given x
            for sub_key, sub_value in value.items():
                if sub_key != 'total':
                    p_y += (sub_value/value['total']) * np.log2(sub_value/value['total'])

            entropy_attr_y -= p_x * p_y

        entropy_attr_y_map[a] = entropy_attr_y

    # Computing information gain
    best_information_gain = -100
    best_attribute = -100
    for attr_key, entropy_value in  entropy_attr_y_map.items():
        information_gain = entropy_y - entropy_value
        if information_gain >= best_information_gain:
            best_information_gain = information_gain
            best_attribute = attr_key

    return best_attribute

# Splitting dataset based on an attribute
def splitDataset(dataset, attribute):

    split_attribute_data = {}

    for i in range(len(dataset)):
        if dataset[i][attribute] not in split_attribute_data:
            split_attribute_data[dataset[i][attribute]] = []
        split_attribute_data[dataset[i][attribute]].append(dataset[i])

    return split_attribute_data

# Returns the result
def evaluateDataset(dataset, parent_dataset, attributes_used):

    y_count = {}

    # If run out of attributes
    if len(attributes_used) == 1:
        if len(dataset) == 0:
            data = parent_dataset
        else:
            data = dataset

        for j in range(len(data)):
            if data[j][0] not in y_count:
                y_count[data[j][0]] = 0.0
            y_count[data[j][0]] += 1.0

        if 1 not in y_count:
            return -1

        if -1 not in y_count:
            return 1

        if y_count[1] >= y_count[-1]:
            return 1
        else:
            return -1

    if len(dataset) > 0:
        # If dataset has rows
        for i in range(len(dataset)):
            if dataset[i][0] not in y_count:
                y_count[dataset[i][0]] = 0.0
            y_count[dataset[i][0]] += 1.0

        # If pure 1 or -1 else recurse
        if 1 in y_count and y_count[1] == len(dataset):
            return 1
        elif -1 in y_count and y_count[-1] == len(dataset):
            return -1
        else:
            return 'recurse'
    else:
        # If dataset has no rows
        for j in range(len(parent_dataset)):
            if parent_dataset[j][0] not in y_count:
                y_count[parent_dataset[j][0]] = 0.0
            y_count[parent_dataset[j][0]] += 1.0

        if 1 not in y_count:
            return -1

        if -1 not in y_count:
            return 1

        if y_count[1] >= y_count[-1]:
            return 1
        else:
            return -1


def build_tree(tree_model, dataset, attributes_used):

    # Get best attribute
    best_attr = bestAttributeSelector(dataset, attributes_used)
    attributes_used.append(best_attr)

    # Init tree
    tree_model[best_attr] = []

    # Split dataset on best attribute
    split_dataset_map = splitDataset(dataset, best_attr)

    # Iterate and evaluate
    for attr_value, sub_dataset in split_dataset_map.items():
        result = evaluateDataset(sub_dataset, dataset, attributes_used)

        attribute_obj = {}
        if result == 1 or result == -1:
            attribute_obj[attr_value] = result
            tree_model[best_attr].append(attribute_obj)
        elif result == 'recurse':
            attribute_obj[attr_value] = build_tree({}, sub_dataset, attributes_used)
            tree_model[best_attr].append(attribute_obj)

    return tree_model

# Recursive traversal through the tree for prediction
def recursive_predict(datarow, d_tree):

    for attr_index, attr_values in d_tree.items():
        for i in range(len(attr_values)):
            if datarow[int(attr_index)] == next(iter(attr_values[i])):
                if isinstance(attr_values[i][next(iter(attr_values[i]))], int):
                    return attr_values[i][datarow[int(attr_index)]]
                elif isinstance(attr_values[i][next(iter(attr_values[i]))], dict):
                    return recursive_predict(datarow, attr_values[i][next(iter(attr_values[i]))])

# Predicting output for data using majority vote of multiple classifiers and computing accuracy
def compute_accuracy(d_trees, dataset):

    predict_array = []

    # Iterating dataset and trees to compute majority
    for i in range(len(dataset)):
        predicted_y_map = {}

        for tree in d_trees:
            y = recursive_predict(dataset[i], tree)
            if y not in predicted_y_map:
                predicted_y_map[y] = 0.0
            predicted_y_map[y] += 1.0

        if 1 not in predicted_y_map:
            predict_array.append(-1)
        elif -1 not in predicted_y_map:
            predict_array.append(1)
        elif predicted_y_map[1] >= predicted_y_map[-1]:
            predict_array.append(1)
        else:
            predict_array.append(-1)

    counter = 0
    index = 0
    for val1 in predict_array:
        if val1 == dataset[index][0]:
            counter += 1
        index += 1

    return counter / len(dataset)

# Resulting decision trees
dtrees = []

# Bagging iterations
for rounds in range(20):

    bootstrap_data = np.empty((train_data_length,23), dtype=int)

    # Generating bootstrap samples of same length
    for i in range(train_data_length):
        bootstrap_data[i] = np.copy(train_data[random.randint(0,79)])

    dtrees.append(build_tree({}, bootstrap_data, []))

print("Classifiers produced:", dtrees)
print("Accuracy on test data:", compute_accuracy(dtrees, test_data))
