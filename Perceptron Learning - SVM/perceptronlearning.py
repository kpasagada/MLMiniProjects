import numpy as np

# Opening file and reading contents
perceptron_file = open('perceptron.data','r')
file_contents = ""
if perceptron_file.mode == 'r':
    file_contents = perceptron_file.read()

# Creating variable to parse and store data
train_data = np.empty((1000,5), dtype=float)

# Initialise row and column counters
i = 0
j = 0

# Read file to parse contents and store in numpy array
for line in file_contents.split('\n'):
    for dt in line.split(","):
        train_data[i,j] = dt
        j += 1
    j = 0
    i += 1

# intializing required parameters
step_size = 1
prev_perceptron_loss = 0.0
perceptron_loss = 0.0
w = np.array([0.0,0.0,0.0,0.0])
b = 0
grad_sum_w = np.array([0.0,0.0,0.0,0.0])
grad_sum_b = 0.0

# Random previous values
prev_grad_sum_w = np.array([1.0,2.0,3.0,6.0])
prev_grad_sum_b = 123456789

iteration_counter = 1

# Iterating for minimum two iterations and until perceptron losses are the same
while iteration_counter <= 2 or ((grad_sum_w[0] - prev_grad_sum_w[0] > 0.01) or (grad_sum_w[1] - prev_grad_sum_w[1] > 0.01) or (grad_sum_w[2] - prev_grad_sum_w[2] > 0.01) or (grad_sum_w[3] - prev_grad_sum_w[3] > 0.01) or (abs(grad_sum_b - prev_grad_sum_b) > 0.01)):
    # Iteration through all 1000 data points
    for i in range(1000):
        #Computing single loss
        loss = -train_data[i,4] * (w.T.dot(train_data[i, 0:4]) + b)
        # Checking if loss is a positive value
        if loss >= 0:
            # Summing all losses for one iteration of the data set
            perceptron_loss += loss
            # Computing derivative sums of w and b
            grad_sum_w += train_data[i,4] * train_data[i, 0:4]
            grad_sum_b += train_data[i,4]
        loss = 0.0
    # Changing w and b values
    w += (step_size/(1+iteration_counter)) * grad_sum_w
    b += (step_size/(1+iteration_counter)) * grad_sum_b

    print("Iteration:", iteration_counter, "Value of w", w, "Value of b:", b, "Loss Sum:",perceptron_loss)
    prev_perceptron_loss = perceptron_loss

    # Resetting parameters for next iteration
    perceptron_loss = 0.0
    prev_grad_sum_w = grad_sum_w
    prev_grad_sum_b = grad_sum_b
    grad_sum_w = np.array([0.0,0.0,0.0,0.0])
    grad_sum_b = 0.0
    iteration_counter += 1
