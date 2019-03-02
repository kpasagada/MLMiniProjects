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
iteration_counter = 1
random_data_counter = 0

# Iterating 2000 times
while True:
    random_data_counter = random_data_counter % 1000

    if random_data_counter % 1000 == 0 and perceptron_loss == 0.0 and iteration_counter >= 2:
        break
    elif random_data_counter % 1000 == 0:
        prev_perceptron_loss = perceptron_loss
        perceptron_loss = 0.0

    #Computing single loss
    loss = -train_data[random_data_counter,4] * (w.T.dot(train_data[random_data_counter, 0:4]) + b)

    # Checking if loss is a positive value
    if loss >= 0:
        # Summing all losses for one iteration of the data set
        perceptron_loss += loss
        # Computing derivative sums of w and b
        grad_sum_w += train_data[random_data_counter,4] * train_data[random_data_counter, 0:4]
        grad_sum_b += train_data[random_data_counter,4]
    loss = 0.0

    # Changing w and b values
    w += step_size * grad_sum_w
    b += step_size * grad_sum_b

    print("Iteration:", iteration_counter, "Value of w", w, "Value of b:", b, "Loss Sum:",perceptron_loss)

    # Resetting parameters for next iteration
    grad_sum_w = np.array([0.0,0.0,0.0,0.0])
    grad_sum_b = 0.0
    iteration_counter += 1
    # Not random, just incrementing by one
    random_data_counter += 1
