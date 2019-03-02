import numpy as np

############### TRAIN ####################
# Opening training file and reading contents
park_train_file = open('park_train.data','r')
file_contents = ""
if park_train_file.mode == 'r':
    file_contents = park_train_file.read()

park_train_file.close()

# Creating variable to parse and store train data
train_data = np.empty((78,23), dtype=float)

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

################ VALIDATION ################
# Opening validation file and reading contents
park_valid_file = open('park_validation.data','r')
file_contents = ""
if park_valid_file.mode == 'r':
    file_contents = park_valid_file.read()

park_valid_file.close()

# Creating variable to parse and store validation data
valid_data = np.empty((58,23), dtype=float)

# Initialise row and column counters
i = 0
j = 0

# Read file to parse contents and store in numpy array
for line in file_contents.split('\n'):
    for dt in line.split(","):
        valid_data[i,j] = dt
        j += 1
    j = 0
    i += 1

# Euclidian function
def euclidian(x,y):
    return np.linalg.norm(x - y)

# Bubble sort - https://www.geeksforgeeks.org/python-program-for-bubble-sort/
def bubbleSort(arr):
    n = len(arr)
    # Traverse through all array elements
    for ii in range(n):
        # Last i elements are already in place
        for jj in range(0, n - ii - 1):
            # traverse the array from 0 to n-i-1
            # Swap if the element found is greater
            # than the next element
            if arr[jj][0] > arr[jj + 1][0]:
                arr[jj][0], arr[jj + 1][0] = arr[jj + 1][0], arr[jj][0]
                arr[jj][1], arr[jj + 1][1] = arr[jj + 1][1], arr[jj][1]

# k values array
k_array = [1,5,11,15,21]

# prediction dict
k_accuracy_dict = {}

# Best value and accuracy of k
best_value_k = -1
best_acc_k = 0

# Iterate for each k value
for k in k_array:
    correct_prediction = 0

    # Iterate validation and train data and compute distance
    for valid in range(len(valid_data)):
        distance_arr = np.empty((len(train_data),2), dtype=float)

        # Calculating euclidian distance
        for train in range(len(train_data)):
            distance_arr[train][0] = euclidian(valid_data[valid], train_data[train])
            distance_arr[train][1] = train

        # Sorting distance array
        bubbleSort(distance_arr)

        min_count = 0
        plu_count = 0

        for dists in distance_arr[0:k]:
            if train_data[int(dists[1])][0] == 1.0:
                plu_count += 1
            elif train_data[int(dists[1])][0] == 0.0:
                min_count += 1

        if plu_count > min_count:
            prediction = 1.0
        else:
            prediction = 0.0

        actual_value = valid_data[valid][0]

        if actual_value == prediction:
            correct_prediction += 1

    k_accuracy_dict[k] = correct_prediction/len(valid_data)

    # Selecting best value of k
    if k_accuracy_dict[k] >= best_acc_k:
        best_value_k = k
        best_acc_k = k_accuracy_dict[k]

################ TEST ################
# Opening test file and reading contents
park_test_file = open('park_test.data','r')
file_contents = ""
if park_test_file.mode == 'r':
    file_contents = park_test_file.read()

park_test_file.close()

# Creating variable to parse and store test data
test_data = np.empty((59,23), dtype=float)

# Initialise row and column counters
i = 0
j = 0

# Read file to parse contents and store in numpy array
for line in file_contents.split('\n'):
    for dt in line.split(","):
        test_data[i,j] = dt
        j += 1
    j = 0
    i += 1


test_correct_prediction = 0

# Iterate test and train data and compute distance
for test in range(len(test_data)):
    distance_arr = np.empty((len(train_data),2), dtype=float)

    # Calculating euclidian distance
    for train in range(len(train_data)):
        distance_arr[train][0] = euclidian(test_data[test], train_data[train])
        distance_arr[train][1] = train

    # Sorting distance array
    bubbleSort(distance_arr)

    min_count = 0
    plu_count = 0

    for dists in distance_arr[0:best_value_k]:
        if train_data[int(dists[1])][0] == 1.0:
            plu_count += 1
        elif train_data[int(dists[1])][0] == 0.0:
            min_count += 1

    if plu_count > min_count:
        prediction = 1.0
    else:
        prediction = 0.0

    actual_value = test_data[test][0]

    if actual_value == prediction:
        test_correct_prediction += 1

print("Accuracy for all k values on validation data:", k_accuracy_dict)
print("The accuracy of the knn classifier is", test_correct_prediction/len(test_data), "with the best value of k:", best_value_k)






