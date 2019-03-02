import numpy as np
import cvxopt as cvx

# SVM predict
def predict(wbs, X):
    vector_len = len(wbs)
    return np.sign(np.dot(wbs[0:vector_len-1].T, X) + wbs[vector_len-1])

# Accuracy calculation
def accuracy(Y1,Y2):
    counter = 0
    index = 0
    for val1 in Y1:
        if val1 == Y2[index]:
            counter += 1
        index += 1

    return counter/len(Y1)

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
train_data_features = train_data[:, 0:len(train_data[0])-1]
train_data_labels = train_data[:, len(train_data[0])-1]


# Selecting best features using PCA
# Computing W matrix
mean_matrix = train_data_features.sum(axis=0)/train_data_length
w_matrix = np.zeros(train_data_features.shape)

for i in range(train_data_length):
    w_matrix[i] = train_data_features[i] - mean_matrix

w_matrix = w_matrix.T

# Computing sample covariance matrix
covar_matrix = np.dot(w_matrix, w_matrix.T)

# Computing eigen values and vectors
eigen_values, eigen_vectors = np.linalg.eig(covar_matrix)

# Sorting eigen values and vectors
idx = eigen_values.argsort()[::-1]
eigen_vectors = eigen_vectors[:, idx]
top_6_eigen_values = eigen_values[0:6]

# k values
k_array = [1, 2, 3, 4, 5, 6]

# Cost dictionary
cost_dict = {}

# Accuracy dictionary for train data
acc_dict_train = {}

# Slack cost variable
c = [1, 10, 100, 1000]

for k in k_array:

    # Reconstructing projected data and add labels vector in the beginning
    projected_train_data = train_data_features.dot(eigen_vectors[:, 0:k])
    projected_train_data = np.append(np.array(train_data_labels).reshape(train_data_length, 1), projected_train_data, axis=1)

    # Temporary cost and accuracy dicts
    temp_cost_dict = {}
    temp_acc_dict_train = {}

    for cost in c:

        #Iterate data and define matrices for quad prog minimize (1/2)xTPx+qTx subject to Gx≤h and Ax=b
        # k w's and 1 b and n slacks
        total_len = k + 1 + train_data_length
        P = np.zeros((total_len, total_len))

        # w2 values
        for diag_index1 in range(k):
            P[diag_index1, diag_index1] = 0.5

        # Multiply by 2 for QP notation
        P = 2.0 * P

        P = cvx.matrix(P, tc='d')

        # Computing q
        q = np.zeros((total_len, 1))

        # Slack variables
        for diag_index2 in range(train_data_length):
            q[diag_index2+k+1, 0] = cost

        q = cvx.matrix(q, tc='d')

        # Computing h
        h = -np.ones((train_data_length*2, 1)).reshape((train_data_length*2,))
        h[train_data_length:train_data_length*2] = 0.0
        h = cvx.matrix(h, tc='d')

        G = np.zeros((train_data_length*2, total_len))

        # Constraint 1 -y(wx+b)-e <= -1
        for l1 in range(train_data_length):
            y = projected_train_data[l1, 0]
            for col_index1 in range(k):
                G[l1, col_index1] = -projected_train_data[l1, col_index1+1] * y
            G[l1, k] = -y
            G[l1, l1+k+1] = -1

        # Constraint 2 -e <= 0
        for l2 in range(train_data_length):
            G[train_data_length+l2, l2+k+1] = -1

        G = cvx.matrix(G, tc='d')

        # Calling quad prog function to find w, slack and b for minimum cost
        wbs_vector = cvx.solvers.qp(P, q, G, h)['x']

        temp_cost_dict[cost] = wbs_vector

        # Calculating accuracy for train data
        if len(wbs_vector) != 0:
            pred_array = []
            actual_data = []
            for i in range(train_data_length):
                slice_data = projected_train_data[i][1:k+1]
                actual_data.append(projected_train_data[i][0])
                pred_array.append(predict(wbs_vector[0:k+1], slice_data))

            temp_acc_dict_train[cost] = accuracy(pred_array, actual_data)

    cost_dict[k] = temp_cost_dict
    acc_dict_train[k] = temp_acc_dict_train

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

valid_data_length = len(valid_data)
valid_data_features = valid_data[:, 0:len(valid_data[0])-1]
valid_data_labels = valid_data[:, len(valid_data[0])-1]

# Selecting best features using PCA
# Computing W matrix
'''w_matrix = np.zeros(valid_data_features.shape)

for i in range(valid_data_length):
    w_matrix[i] = valid_data_features[i] - mean_matrix

w_matrix = w_matrix.T

# Computing sample covariance matrix
covar_matrix = np.dot(w_matrix, w_matrix.T)

# Computing eigen values and vectors
eigen_values, eigen_vectors = np.linalg.eig(covar_matrix)

# Sorting eigen values and vectors
idx = eigen_values.argsort()[::-1]
eigen_vectors = eigen_vectors[:, idx]'''

# Accuracy dictionary for validation data
acc_dict_valid = {}

best_valid_cost = -100
best_valid_accuracy = -100
best_k = -1

# Calculating accuracy for validation data
for k in k_array:

    # Reconstructing projected data and add labels vector in the beginning
    projected_valid_data = valid_data_features.dot(eigen_vectors[:, 0:k])
    projected_valid_data = np.append(np.array(valid_data_labels).reshape(valid_data_length, 1), projected_valid_data, axis=1)

    temp_acc_dict_valid = {}

    for cost, vector in cost_dict[k].items():
        if len(vector) != 0:
            pred_array = []
            actual_data = []
            for i in range(valid_data_length):
                slice_data = projected_valid_data[i][1:k+1]
                actual_data.append(projected_valid_data[i][0])
                pred_array.append(predict(vector[0:k+1], slice_data))

            temp_acc_dict_valid[cost] = accuracy(pred_array, actual_data)

            if temp_acc_dict_valid[cost] >= best_valid_accuracy:
                best_valid_cost = cost
                best_k = k
                best_valid_accuracy = temp_acc_dict_valid[cost]

    acc_dict_valid[k] = temp_acc_dict_valid


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

test_data_length = len(test_data)
test_data_features = test_data[:, 0:len(test_data[0])-1]
test_data_labels = test_data[:, len(test_data[0])-1]

# Selecting best features using PCA
# Computing W matrix
'''w_matrix = np.zeros(test_data_features.shape)

for i in range(test_data_length):
    w_matrix[i] = test_data_features[i] - mean_matrix

w_matrix = w_matrix.T

# Computing sample covariance matrix
covar_matrix = np.dot(w_matrix, w_matrix.T)

# Computing eigen values and vectors
eigen_values, eigen_vectors = np.linalg.eig(covar_matrix)

# Sorting eigen values and vectors
idx = eigen_values.argsort()[::-1]
eigen_vectors = eigen_vectors[:, idx]'''

# Reconstructing projected data and add labels vector in the beginning
projected_test_data = test_data_features.dot(eigen_vectors[:, 0:best_k])
projected_test_data = np.append(np.array(test_data_labels).reshape(test_data_length, 1), projected_test_data, axis=1)

# Calculating accuracy for test data with best cost and its vectors
pred_array = []
actual_data = []

for i in range(test_data_length):
    slice_data = projected_test_data[i][1:best_k+1]
    actual_data.append(projected_test_data[i][0])
    pred_array.append(predict(cost_dict[best_k][best_valid_cost][0:best_k+1], slice_data))

print("Top 6 Eigen values:", top_6_eigen_values)
print("Accuracy of the classifier on validation data for each k and c: ", acc_dict_valid)
print("Accuracy on test data:", accuracy(pred_array, actual_data), "with best cost:", best_valid_cost, "and best k:", best_k)