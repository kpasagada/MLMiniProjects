import numpy as np

# Compute the value of the likelihood function
def compute_loss(datarow, w, b):
    wxb = np.dot(w.T, datarow[1:23]) + b
    return (((datarow[0] + 1)/2) * wxb) - (np.log(1 + np.exp(wxb)))

# Compute w and b gradient values
def compute_wb(datarow, w, b):
    wxb = np.exp(np.dot(w.T, datarow[1:23]) + b)
    py1 = wxb / (1 + wxb)
    diff_b = ((datarow[0] + 1) / 2) - py1
    diff_w = datarow[1:23] * diff_b
    return diff_w, diff_b

# Predict values
def predict(w, b, datarow):
    wxb = np.exp(np.dot(w.T, datarow) + b)
    py_positive = wxb / (1 + wxb)
    py_negative = 1 / (1 + wxb)
    return 1.0 if py_positive >= py_negative else -1.0

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
park_train_file = open('park_train.data','r')
file_contents = ""
if park_train_file.mode == 'r':
    file_contents = park_train_file.read()

park_train_file.close()

# Creating variable to parse and store train data
train_data = np.empty((78, 23), dtype=float)

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

# Gradient ascent parameter initialization
step_size = 0.000001
lambda_array = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]

# w and b dict for all lambda values
wb_lambda_dict = {}

for lambda_value in lambda_array:

    # Gradient ascent parameters
    iteration_counter = 1
    w_vector = 0.25 * np.ones((22,), dtype=float)
    b_value = 0.75
    total_loss = 0.0

    # Iterating till convergence
    while True:

        # Saving previous loss
        prev_loss = total_loss
        total_loss = 0.0
        grad_sum_w = np.zeros((22,))
        grad_sum_b = 0.0

        # Iterating all training data
        for i in range(train_data_length):
            total_loss += compute_loss(train_data[i], w_vector, b_value)
            temp_w, temp_b = compute_wb(train_data[i], w_vector, b_value)
            grad_sum_w += temp_w
            grad_sum_b += temp_b

        # L2 regularization
        total_loss -= (lambda_value/2) * (np.linalg.norm(w_vector) ** 2)
        grad_sum_w -= lambda_value * w_vector

        print("Iteration:", iteration_counter, "Total loss:", total_loss)

        # If difference in loss in minimal
        if total_loss - prev_loss < 0.0002 and iteration_counter >= 2:
            break

        # Updating w and b values
        w_vector += step_size * grad_sum_w
        b_value += step_size * grad_sum_b

        # Increasing count
        iteration_counter += 1

    wb_lambda_dict[lambda_value] = np.append(w_vector, b_value)


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
            valid_data[i, j] = dt
        j += 1
    j = 0
    i += 1

# Accuracy dictionary for validation data
acc_dict_valid = {}

best_valid_lambda = -100
best_valid_accuracy = -100

# Calculating accuracy for validation data
for lamb, vector in wb_lambda_dict.items():
    if len(vector) != 0:
        pred_array = []
        actual_data = []
        for i in range(len(valid_data)):
            slice_data = valid_data[i][1:23]
            actual_data.append(valid_data[i][0])
            pred_array.append(predict(vector[0:22], vector[22], slice_data))

        acc_dict_valid[lamb] = accuracy(pred_array, actual_data)

        if acc_dict_valid[lamb] >= best_valid_accuracy:
            best_valid_lambda = lamb
            best_valid_accuracy = acc_dict_valid[lamb]

print("Accuracy on validation data for each lambda:", acc_dict_valid)

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
            test_data[i, j] = dt
        j += 1
    j = 0
    i += 1

# Calculating accuracy for test data
pred_array = []
actual_data = []

for i in range(len(test_data)):
    slice_data = test_data[i][1:23]
    actual_data.append(test_data[i][0])
    pred_array.append(predict(wb_lambda_dict[best_valid_lambda][0:22], wb_lambda_dict[best_valid_lambda][22], slice_data))

print("Best w vector:", wb_lambda_dict[best_valid_lambda][0:22])
print("Best b value:", wb_lambda_dict[best_valid_lambda][22])
print("Accuracy on test data:", accuracy(pred_array, actual_data), "for best lambda:", best_valid_lambda)

