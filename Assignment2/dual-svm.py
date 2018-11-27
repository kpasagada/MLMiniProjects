import numpy as np
import cvxopt as cvx

# Define Gaussian Kernel formula
def gaussian_kernel(x, y, sigma=0.001):
    return np.exp((-np.linalg.norm(x-y)**2) / (2 * (sigma ** 2)))

def classifyAndPredict(d_training, d_testing, lambda_vector, d_sigma, d_cost):

    b_value = 0.0
    b_index = -1
    b_sum = 0.0

    # Computing b using some lambda > 0 and y = 1
    for i in range(len(d_training)):
        if lambda_vector[i] > 0 and lambda_vector[i] < d_cost:
            for j in range(len(d_training)):
                b_sum += lambda_vector[j] * d_training[j][0] * gaussian_kernel(d_training[i][1:23], d_training[j][1:23], d_sigma)
            b_value = d_training[i][0] - b_sum
            b_index = i
            break

    print("Choosing", b_index, "th value of training data and value of b:", b_value)

    correct_class = 0

    # Predicting y values for incoming data i.e, d_testing
    for ii in range(len(d_testing)):
        w_sum = 0.0
        for jj in range(len(d_training)):
            w_sum += lambda_vector[jj] * d_training[jj][0] * gaussian_kernel(d_training[jj][1:23], d_testing[ii][1:23], d_sigma)
        predict_val = np.sign(w_sum + b_value)

        # Checking if predicted is equal to actual
        if predict_val == d_testing[ii][0]:
            correct_class += 1

    return correct_class/len(d_testing)


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
            train_data[i, j] = dt
        j += 1
    j = 0
    i += 1

train_data_length = len(train_data)

# Slack cost variable
cost_array = [1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000]

# Gaussian sigma variable
sigma_array = [0.1, 1, 10, 100, 1000]

# Cost sigma and accuracy dict containing w and b values for training data
cost_sigma_dict = {}
acc_dict_train = {}

for cost in cost_array:

    sigma_dict = {}
    acc_dict = {}

    for sigma in sigma_array:
        # Iterate data and define matrices for quad prog minimize (1/2)xTPx+qTx subject to Gx≤h and Ax=b
        # Computing P
        P = np.zeros((78,78))

        for i in range(train_data_length):
            for j in range(train_data_length):
                P[i][j] = gaussian_kernel(train_data[i][1:23], train_data[j][1:23], sigma) * train_data[j][0] * train_data[i][0] * 0.5

        # Multiply by 2 for QP notation
        P = 2 * P

        P = cvx.matrix(P, tc='d')

        # Computing q
        q = -np.ones((78, 1)).reshape((78,))
        q = cvx.matrix(q, tc='d')

        # Computing G
        G = np.zeros((156,78))

        for i in range(train_data_length):
            G[i][i] = -1.0
            G[i + 78][i] = 1.0

        G = cvx.matrix(G, tc='d')

        # Computing h
        h = np.zeros((156,1))

        for i in range(train_data_length):
            h[i + 78][0] = cost

        h.reshape((156,))

        h = cvx.matrix(h, tc='d')

        # Computing A and b
        A = np.zeros((1,78))

        for i in range(train_data_length):
            A[0][i] = train_data[i][0]

        A = cvx.matrix(A, tc='d')

        b = np.zeros((1,1)).reshape((1,))
        b = cvx.matrix(b, tc='d')

        # Calling quad prog function to find lambda for minimum cost
        try:
            wbs_vector = cvx.solvers.qp(P,q,G,h,A,b)['x']
        except ValueError:
            print("No Solution for cost:", cost, "and sigma:", sigma)
            wbs_vector = []

        sigma_dict[sigma] = wbs_vector

        # Calculating accuracy for train data
        if len(wbs_vector) != 0:
            acc_dict[sigma] = classifyAndPredict(train_data, train_data, wbs_vector, sigma, cost)

    cost_sigma_dict[cost] = sigma_dict
    acc_dict_train[cost] = acc_dict

print(acc_dict_train)

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
best_valid_sigma = -1
best_valid_accuracy = -100

# Calculating accuracy for validation data
for cost, sigma_dict in cost_sigma_dict.items():
    acc_sigma_dict_valid = {}

    for sigma, vector in sigma_dict.items():
        acc_sigma_dict_valid[sigma] = classifyAndPredict(train_data, valid_data, vector, sigma, cost)

        if acc_sigma_dict_valid[sigma] >= best_valid_accuracy:
            best_valid_cost = cost
            best_valid_sigma = sigma
            best_valid_accuracy = acc_sigma_dict_valid[sigma]

    acc_dict_valid[cost] = acc_sigma_dict_valid

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

print("Accuracy of the classifier on training data for each c and sigma:", acc_dict_train)
print("Accuracy of the classifier on validation data for each c and sigma:", acc_dict_valid)
# Calculating accuracy on test data with best values of sigma and cost
print("Accuracy on test data:", classifyAndPredict(train_data, test_data, cost_sigma_dict[best_valid_cost][best_valid_sigma], best_valid_sigma, best_valid_cost),"with best cost:",best_valid_cost,"and best sigma:",best_valid_sigma)


