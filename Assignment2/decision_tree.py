import numpy as np

############### TRAIN ####################
# Opening training file and reading contents
mush_train_file = open('mush_train.data','r')
file_contents = ""
if mush_train_file.mode == 'r':
    file_contents = mush_train_file.read()

mush_train_file.close()

# Creating variable to parse and store train data
train_data = np.empty((4712,23), dtype=str)

# Initialise row and column counters
i = 0
j = 0

# Read file to parse contents and store in numpy array
for line in file_contents.split('\n'):
    for dt in line.split(","):
        train_data[i, j] = dt
        j += 1
    j = 0
    i += 1

# Attribute and values mapping
attr_value_mapping = np.array([['p','e'], ['b','c','x','f','k','s'], ['f','g','y','s'],
                              ['n','b','c','g','r','p','u','e','w','y'], ['t','f'], ['a','l','c','y','f','m','n','p','s'],
                              ['a','d','f','n'], ['c','w','d'], ['b','n'],
                              ['k','n','b','h','g','r','o','p','u','e','w','y'], ['e','t'], ['b','c','u','e','z','r','?'],
                              ['f','y','k','s'], ['f','y','k','s'], ['n','b','c','g','o','p','e','w','y'],
                              ['n','b','c','g','o','p','e','w','y'], ['p','u'],
                              ['n','o','w','y'], ['n','o','t'], ['c','e','f','l','n','p','s','z'],
                              ['k','n','b','h','r','o','u','w','y'], ['a','c','n','s','v','y'], ['g','l','m','p','u','w','d']])

# Computing entropy and information gain for given attributes
def bestAttributeSelector(dataset, attributes_used):

    entropy_map = {}

    # Computing number of p and e in dataset
    for i in range(len(dataset)):
        for j in range(len(attr_value_mapping[0])):
            if dataset[i][0] == attr_value_mapping[0][j]:
                if attr_value_mapping[0][j] not in entropy_map:
                    entropy_map[attr_value_mapping[0][j]] = 0.0
                entropy_map[attr_value_mapping[0][j]] += 1.0
                break

    # Computing probability of p and e in dataset and computing entropy of Y = (p,e)
    entropy_y = 0.0
    for key, value in entropy_map.items():
        entropy_y -= (value/len(dataset)) * np.log2(value/len(dataset))

    # Computing number of values for each attribute in the dataset and calculating entropy for each y over x
    entropy_attr_y_map = {}

    for a in range(1, len(attr_value_mapping)):
        entropy_attr_map = {}

        if a in attributes_used:
            continue

        # Counting attribute values and their respective p and e counts
        for i in range(len(dataset)):
            for j in range(len(attr_value_mapping[a])):
                if dataset[i][a] == attr_value_mapping[a][j]:
                    # Check if attribute value is in map
                    if attr_value_mapping[a][j] not in entropy_attr_map:
                        entropy_attr_map[attr_value_mapping[a][j]] = {}
                        entropy_attr_map[attr_value_mapping[a][j]]['total'] = 0.0

                    # Check if p or e is in attribute value sub map
                    if dataset[i][0] not in entropy_attr_map[attr_value_mapping[a][j]]:
                        entropy_attr_map[attr_value_mapping[a][j]][dataset[i][0]] = 0.0

                    # Increment total and p or e values
                    entropy_attr_map[attr_value_mapping[a][j]][dataset[i][0]] += 1.0
                    entropy_attr_map[attr_value_mapping[a][j]]['total'] += 1.0
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
    if len(attributes_used) == len(attr_value_mapping):
        if len(dataset) == 0:
            data = parent_dataset
        else:
            data = dataset

        for j in range(len(data)):
            if data[j][0] not in y_count:
                y_count[data[j][0]] = 0.0
            y_count[data[j][0]] += 1.0

        if y_count['e'] >= y_count['p']:
            return 'e'
        else:
            return 'p'

    if len(dataset) > 0:
        # If dataset has rows
        for i in range(len(dataset)):
            if dataset[i][0] not in y_count:
                y_count[dataset[i][0]] = 0.0
            y_count[dataset[i][0]] += 1.0

        # If pure e or p else recurse
        if 'e' in y_count and y_count['e'] == len(dataset):
            return 'e'
        elif 'p' in y_count and y_count['p'] == len(dataset):
            return 'p'
        else:
            return 'recurse'
    else:
        # If dataset has no rows
        for j in range(len(parent_dataset)):
            if parent_dataset[j][0] not in y_count:
                y_count[parent_dataset[j][0]] = 0.0
            y_count[parent_dataset[j][0]] += 1.0

        if y_count['e'] >= y_count['p']:
            return 'e'
        else:
            return 'p'


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
        if result == 'p' or result == 'e':
            attribute_obj[attr_value] = result
            tree_model[best_attr].append(attribute_obj)
        elif result == 'recurse':
            attribute_obj[attr_value] = build_tree({}, sub_dataset, attributes_used)
            tree_model[best_attr].append(attribute_obj)

    return tree_model

# Creating empty tree
decision_tree = {}

# Building tree
build_tree(decision_tree, train_data, [])

############### TEST ####################
# Opening test file and reading contents
mush_test_file = open('mush_test.data','r')
file_contents = ""
if mush_test_file.mode == 'r':
    file_contents = mush_test_file.read()

mush_test_file.close()

# Creating variable to parse and store test data
test_data = np.empty((3412,23), dtype=str)

# Initialise row and column counters
i = 0
j = 0

# Read file to parse contents and store in numpy array
for line in file_contents.split('\n'):
    for dt in line.split(","):
        test_data[i, j] = dt
        j += 1
    j = 0
    i += 1

# Recursive traversal through the tree for prediction
def recursive_predict(datarow, tree):

    for attr_index, attr_values in tree.items():
        for i in range(len(attr_values)):
            if datarow[attr_index] == next(iter(attr_values[i])):
                if isinstance(attr_values[i][next(iter(attr_values[i]))], str):
                    return attr_values[i][datarow[attr_index]]
                elif isinstance(attr_values[i][next(iter(attr_values[i]))], dict):
                    return recursive_predict(datarow, attr_values[i][next(iter(attr_values[i]))])

# Prediction function for a dataset
def predict(dataset):

    predict_array = []
    for i in range(len(dataset)):
        predict_array.append(recursive_predict(dataset[i], decision_tree))

    counter = 0
    index = 0
    for val1 in predict_array:
        if val1 == dataset[index][0]:
            counter += 1
        index += 1

    return counter / len(dataset)

print("Decision Tree:", decision_tree)
print("Accuracy of training data:", predict(train_data))
print("Accuracy of testing data:", predict(test_data))

