import numpy as np
import math as mt

# Function used to partition data on labels
def separate_data(data):
    data_map = {}
    feature_len = len(data[0]) - 1
    for i in range(len(data)):
        if data[i, -1] not in data_map:
            data_map[data[i, -1]] = []
        data_map[data[i, -1]].append(data[i][0:feature_len])

    return data_map

# Function returns mean and standard deviation for each feature per class
def calc_mean_std(data):
    mean_std_map = {}
    for label, dataset in data.items():
        mean_std_map[label] = [np.mean(dataset, axis=0), np.std(dataset, axis=0, ddof=1)]

    return mean_std_map

# Computes gaussian
def calculateGaussian(item, mean, std):
    exp = mt.exp(-(mt.pow(item - mean, 2) / (2 * mt.pow(std, 2))))
    return (1 / (mt.sqrt(2 * mt.pi) * std)) * exp

# Function returns predicted label for each data row
def predict(datarow, mean_std_map):
    prob_label_map = {}

    for label, item in mean_std_map.items():
        likelihood = 1.0
        for i in range(len(datarow)):
            likelihood *= calculateGaussian(datarow[i], item[0][i], item[1][i])

        prob_label_map[label] = likelihood

    best_label = 0.0
    best_prob = None
    for label, prob in prob_label_map.items():
        if best_prob is None or best_prob < prob:
            best_prob = prob
            best_label = label

    return best_label

# Accuracy calculation
def accuracy(Y1,Y2):
    counter = 0
    index = 0
    for val1 in Y1:
        if val1 == Y2[index]:
            counter += 1
        index += 1

    return counter/len(Y1)

# Computes pi
def compute_pi(vectors, length):
    return np.sum(vectors ** 2, axis=1) / length

# Opening training file and reading contents
sonar_train_file = open('sonar_train.csv', 'r')

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
        train_data[i,j] = dt
        j += 1
    j = 0
    i += 1

# Separating features and labels
train_data_length = len(train_data)
train_data_features = train_data[:, 0:len(train_data[0])-1]
train_data_labels = train_data[:, len(train_data[0])-1]


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
        test_data[i, j] = dt
        j += 1
    j = 0
    i += 1

test_data_length = len(test_data)
test_data_features = test_data[:, 0:len(test_data[0])-1]
test_data_labels = test_data[:, len(test_data[0])-1]


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

# Iterating k values
for k in range(1,11):

    # Selecting k eigen vectors
    k_eigen_vectors = eigen_vectors[:, 0:k]

    # Computing pi
    pi = compute_pi(k_eigen_vectors, k)

    # Iterating s values
    for s in range(1,21):

        # Array to store 100 accuracies
        avg_accuracy = 0.0

        for iter in range(100):
            # Getting s indices from probability distribution pi
            indices = np.random.choice(np.arange(0,60), p=pi, size=s)

            # Retrieving select features from the train data
            subset_train_data = train_data[:,np.unique(indices)]

            # Appending features column to the subset train data
            subset_train_data = np.append(subset_train_data, np.array(train_data_labels).reshape(train_data_length, 1), axis=1)

            # Separating data by labels
            separated_train_data = separate_data(subset_train_data)

            # Comupting mean and standard deviation for the label separated data
            mean_std_train = calc_mean_std(separated_train_data)

            # Getting a subset of test data
            subset_test_data = test_data[:, np.unique(indices)]

            # Getting the predictions for test data
            predict_array = []
            for i in range(test_data_length):
                predict_array.append(predict(subset_test_data[i], mean_std_train))

            # Computing accuracy for test data
            avg_accuracy += accuracy(predict_array, test_data_labels)

        print("Average accuracy for k:", k, "and s:", s, "is:", avg_accuracy/100)