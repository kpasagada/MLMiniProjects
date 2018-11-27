import numpy as np
import cvxopt as cvx

########################## TRAIN ##########################
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
        # If y is 0 change to -1
        if j == 0 and dt == '0':
            train_data[i, j] = -1
        else:
            train_data[i,j] = dt
        j += 1
    j = 0
    i += 1

train_data_length = len(train_data)

# SVM predict
def predict(wbs,X):
    return np.sign(np.dot(X,wbs[0:22]) + wbs[22])

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
c = [1,10,100,1000,10000,100000,1000000,10000000,100000000]

for cost in c:

    #Iterate data and define matrices for quad prog minimize (1/2)xTPx+qTx subject to Gx≤h and Ax=b
    # 101 w's and 1 b
    P = np.zeros((101,101))

    # w2 values
    for diag_index1 in range(22):
        P[diag_index1,diag_index1] = 1.0

    # b value
    P[22,22] = 0.0

    # Slack variables
    for diag_index2 in range(78):
        P[diag_index2+23,diag_index2+23] = 0.0

    # Multiply by 2 for QP notation
    P = 2.0 * P

    P = cvx.matrix(P, tc='d')

    # Computing q
    q = np.zeros((101,1))
    q = cvx.matrix(q, tc='d')

    # Slack variables
    for diag_index2 in range(78):
        q[diag_index2 + 23, 0] = cost

    # Computing h
    h = -np.ones((train_data_length*2,1)).reshape((train_data_length*2,))
    h[train_data_length:train_data_length*2] = 0.0
    h = cvx.matrix(h, tc='d')

    G = np.zeros((train_data_length*2,101))

    # Constraint 1 - y(wx-b)-e <= -1
    for l1 in range(train_data_length):
        y = train_data[l1, 0]
        for col_index1 in range(22):
            G[l1, col_index1] = -train_data[l1, col_index1+1] * y
        G[l1,22] = -y
        G[l1, l1+23] = -1

    # Constraint 2 -e <= 0
    for l2 in range(train_data_length):
        G[train_data_length+l2, l2+23] = -1

    G = cvx.matrix(G, tc='d')

    # Calling quad prog function to find w, slack and b for minimum cost
    wbs_vector = cvx.solvers.qp(P,q,G,h)['x']

    cost_dict[cost] = wbs_vector

    # Calculating accuracy for train data
    if len(wbs_vector) != 0:
        pred_array = []
        actual_data = []
        for i in range(train_data_length):
            slice_data = train_data[i][1:23]
            actual_data.append(train_data[i][0])
            pred_array.append(predict(wbs_vector[0:23],slice_data))

        acc_dict_train[cost] = accuracy(pred_array,actual_data)

############################# VALIDATION #########################################
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
        # If y is 0 change to -1
        if j == 0 and dt == '0':
            valid_data[i, j] = -1
        else:
            valid_data[i,j] = dt
        j += 1
    j = 0
    i += 1

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
            slice_data = valid_data[i][1:23]
            actual_data.append(valid_data[i][0])
            pred_array.append(predict(vector[0:23], slice_data))

        acc_dict_valid[cost] = accuracy(pred_array, actual_data)

        if acc_dict_valid[cost] >= best_valid_accuracy:
            best_valid_cost = cost
            best_valid_accuracy = acc_dict_valid[cost]

############################# TEST #########################################
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
        # If y is 0 change to -1
        if j == 0 and dt == '0':
            test_data[i, j] = -1
        else:
            test_data[i,j] = dt
        j += 1
    j = 0
    i += 1

# Calculating accuracy for test data with best cost and its vectors
pred_array = []
actual_data = []

for i in range(len(test_data)):
    slice_data = test_data[i][1:23]
    actual_data.append(test_data[i][0])
    pred_array.append(predict(cost_dict[best_valid_cost][0:23], slice_data))

print("Accuracy of the classifier on training data for each c: ", acc_dict_train)
print("Accuracy of the classifier on training data for each c: ", acc_dict_valid)
print("Accuracy on test data:",accuracy(pred_array, actual_data),"with best cost:",best_valid_cost)