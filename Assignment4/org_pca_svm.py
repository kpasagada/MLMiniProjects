import numpy as np
import cvxopt as cvx

########################## TRAIN ##########################
# Opening training file and reading contents
sonar_train_file = open('sonar_train.csv','r')
file_contents = ""
if sonar_train_file.mode == 'r':
    file_contents = sonar_train_file.read()

sonar_train_file.close()

# Creating variable to parse and store train data
train_data = np.empty((104, 61), dtype=float)

# Initialise row and column counters
i = 0
j = 0

# Read file to parse contents and store in numpy array
for line in file_contents.split('\n'):
    for dt in line.split(","):
        # If y is 0 change to -1
        if j == 60 and dt == '2':
            train_data[i, j] = -1
        else:
            train_data[i,j] = dt
        j += 1
    j = 0
    i += 1

# Separating features and labels
train_data_length = len(train_data)
train_row_length = len(train_data[0])
train_data = np.append(np.array(train_data[:, train_row_length-1]).reshape((train_data_length, 1)), train_data[:, 0:train_row_length-1], axis=1)
features_length = train_row_length - 1

# SVM predict
def predict(wbs, X):
    vector_len = len(wbs)
    return np.sign(np.dot(X,wbs[0:vector_len-1]) + wbs[vector_len-1])

# Accuracy calculation
def accuracy(Y1,Y2):
    counter = 0
    index = 0
    for val1 in Y1:
        if val1 == Y2[index]:
            counter += 1
        index += 1

    return counter/len(Y1)

# Cost dictionary
cost_dict = {}

# Accuracy dictionary for train data
acc_dict_train = {}

# Slack cost variable
c = [1,10,100,1000]

for cost in c:

    #Iterate data and define matrices for quad prog minimize (1/2)xTPx+qTx subject to Gx≤h and Ax=b
    # 101 w's and 1 b
    total_len = features_length + 1 + train_data_length
    P = np.zeros((total_len, total_len))

    # w2 values
    for diag_index1 in range(features_length):
        P[diag_index1, diag_index1] = 0.5

    # Multiply by 2 for QP notation
    P = 2.0 * P

    P = cvx.matrix(P, tc='d')

    # Computing q
    q = np.zeros((total_len, 1))

    # Slack variables
    for diag_index2 in range(train_data_length):
        q[diag_index2+features_length+1, 0] = cost

    q = cvx.matrix(q, tc='d')

    # Computing h
    h = -np.ones((train_data_length*2, 1)).reshape((train_data_length*2,))
    h[train_data_length:train_data_length*2] = 0.0
    h = cvx.matrix(h, tc='d')

    # Computing G
    G = np.zeros((train_data_length*2, total_len))

    # Constraint 1 - y(wx-b)-e <= -1
    for l1 in range(train_data_length):
        y = train_data[l1, 0]
        for col_index1 in range(features_length):
            G[l1, col_index1] = -train_data[l1, col_index1+1] * y
        G[l1, features_length] = -y
        G[l1, l1+features_length+1] = -1

    # Constraint 2 -e <= 0
    for l2 in range(train_data_length):
        G[train_data_length+l2, l2+features_length+1] = -1

    G = cvx.matrix(G, tc='d')

    # Calling quad prog function to find w, slack and b for minimum cost
    wbs_vector = cvx.solvers.qp(P,q,G,h)['x']

    cost_dict[cost] = wbs_vector

    # Calculating accuracy for train data
    if len(wbs_vector) != 0:
        pred_array = []
        actual_data = []
        for i in range(train_data_length):
            slice_data = train_data[i][1:features_length+1]
            actual_data.append(train_data[i][0])
            pred_array.append(predict(wbs_vector[0:features_length+1], slice_data))

        acc_dict_train[cost] = accuracy(pred_array, actual_data)

############################# VALIDATION #########################################
# Opening validation file and reading contents
sonar_valid_file = open('sonar_valid.csv','r')
file_contents = ""
if sonar_valid_file.mode == 'r':
    file_contents = sonar_valid_file.read()

sonar_valid_file.close()

# Creating variable to parse and store validation data
valid_data = np.empty((52, 61), dtype=float)

# Initialise row and column counters
i = 0
j = 0

# Read file to parse contents and store in numpy array
for line in file_contents.split('\n'):
    for dt in line.split(","):
        # If y is 0 change to -1
        if j == 60 and dt == '2':
            valid_data[i, j] = -1
        else:
            valid_data[i, j] = dt
        j += 1
    j = 0
    i += 1

valid_row_length = len(valid_data[0])
valid_data = np.append(np.array(valid_data[:, valid_row_length-1]).reshape((len(valid_data), 1)), valid_data[:, 0:valid_row_length-1], axis=1)

# Accuracy dictionary for validation data
acc_dict_valid = {}

best_valid_cost = -100
best_valid_accuracy = -100

# Calculating accuracy for validation data
for cost, vector in cost_dict.items():
    if len(vector) != 0:
        pred_array = []
        actual_data = []
        for i in range(len(valid_data)):
            slice_data = valid_data[i][1:features_length+1]
            actual_data.append(valid_data[i][0])
            pred_array.append(predict(vector[0:features_length+1], slice_data))

        acc_dict_valid[cost] = accuracy(pred_array, actual_data)

        if acc_dict_valid[cost] >= best_valid_accuracy:
            best_valid_cost = cost
            best_valid_accuracy = acc_dict_valid[cost]

############################# TEST #########################################
# Opening test file and reading contents
sonar_test_file = open('sonar_test.csv','r')
file_contents = ""
if sonar_test_file.mode == 'r':
    file_contents = sonar_test_file.read()

sonar_test_file.close()

# Creating variable to parse and store test data
test_data = np.empty((52, 61), dtype=float)

# Initialise row and column counters
i = 0
j = 0

# Read file to parse contents and store in numpy array
for line in file_contents.split('\n'):
    for dt in line.split(","):
        # If y is 0 change to -1
        if j == 60 and dt == '2':
            test_data[i, j] = -1
        else:
            test_data[i, j] = dt
        j += 1
    j = 0
    i += 1

test_row_length = len(test_data[0])
test_data = np.append(np.array(test_data[:, test_row_length-1]).reshape((len(test_data), 1)), test_data[:, 0:test_row_length-1], axis=1)

# Calculating accuracy for test data with best cost and its vectors
pred_array = []
actual_data = []

for i in range(len(test_data)):
    slice_data = test_data[i][1:features_length+1]
    actual_data.append(test_data[i][0])
    pred_array.append(predict(cost_dict[best_valid_cost][0:features_length+1], slice_data))

print("Accuracy of the classifier on training data for each c: ", acc_dict_train)
print("Accuracy of the classifier on validation data for each c: ", acc_dict_valid)
print("Accuracy on test data:",accuracy(pred_array, actual_data),"with best cost:",best_valid_cost)