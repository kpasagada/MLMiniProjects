import numpy as np
import sklearn.cluster as cluster
import matplotlib.pyplot as plt

# Opening training file and reading contents
circ_train_file = open('circs.data','r')
file_contents = ""
if circ_train_file.mode == 'r':
    file_contents = circ_train_file.read()

circ_train_file.close()

# Creating variable to parse and store train data
train_data = np.empty((102,2))

# Initialise row and column counters
i = 0
counter_1 = 0
counter_2 = 0

# Read file to parse contents and store in numpy array
for line in file_contents.split('\n'):
    for dt in line.split(" "):
        if i % 2 == 0:
            train_data[counter_1, 0] = dt
            counter_1 += 1
        else:
            train_data[counter_2, 1] = dt
            counter_2 += 1
    i += 1

train_data_length = len(train_data)

# Function to compute similarity
def compute_similarity(vector1, vector2, sigma):
    return np.exp(-(1 / (2 * (sigma ** 2))) * (np.linalg.norm(vector1 - vector2) ** 2))

# Function to compute the sum of squared loss
def calculate_loss(data, labels):
    sum_error = 0.0
    clusters = {}

    # Compute cluster centers
    for i in range(len(data)):
        if labels[i] not in clusters:
            clusters[labels[i]] = {'sum': [0.0, 0.0], 'count': 0}
        clusters[labels[i]]['sum'] += data[i]
        clusters[labels[i]]['count'] += 1

    for key in clusters:
        clusters[key]['total'] = clusters[key]['sum'] / clusters[key]['count']

    # Computing loss
    for i in range(len(data)):
        sum_error += np.linalg.norm(data[i] - clusters[labels[i]]['total']) ** 2

    return sum_error

# Setting k_value
k_value = 2

# Color labels for scatter plot
LABEL_COLOR_MAP = {0: 'green', 1: 'red', 2: 'blue', 3: 'pink', 4: 'purple', 5: 'black'}

# Alternative Kmeans clustering
kmeans_alt = cluster.KMeans(n_clusters=k_value).fit(train_data)

# Plotting for Kmeans
label_color_alt = [LABEL_COLOR_MAP[l] for l in kmeans_alt.labels_]
plt.scatter(train_data[:, 0], train_data[:, 1], c=label_color_alt)
plt.title("K-Means")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

print("Loss for K-Means:", calculate_loss(train_data, kmeans_alt.labels_))

# Spectral Clustering
# Looping for each sigma value
sigma_array = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]

for sigma in sigma_array:
    # Constructing similarity matrix A
    similarity_matrix = np.empty((102, 102))

    for i in range(train_data_length):
        for j in range(i+1, train_data_length):
            similarity_matrix[i, j] = similarity_matrix[j, i] = compute_similarity(train_data[i], train_data[j], sigma)
        similarity_matrix[i, i] = compute_similarity(train_data[i], train_data[i], sigma)

    # Construct diagonal matrix
    diagonal_matrix = np.zeros((102, 102))
    sum_matrix = similarity_matrix.sum(axis=1)
    for i in range(train_data_length):
        diagonal_matrix[i, i] = sum_matrix[i]

    # Construct laplacian matrix
    laplacian_matrix = diagonal_matrix - similarity_matrix

    # Computing eigen values and vectors
    eig_values, eig_vectors = np.linalg.eig(laplacian_matrix)

    # Sorting eigen values and vectors
    ordered_indices = eig_values.argsort()
    eig_values = eig_values[ordered_indices]
    eig_vectors = eig_vectors[:, ordered_indices]

    # Getting non-zero eigen values and vectors
    non_zero_indices = np.nonzero(eig_values)
    non_zero_indices = np.asarray(non_zero_indices)[0]
    eig_values = eig_values[non_zero_indices]
    eig_vectors = eig_vectors[:, non_zero_indices]

    # Constructing V matrix
    v_matrix = eig_vectors[:, 0:k_value]

    # Kmeans clustering
    kmeans = cluster.KMeans(n_clusters=k_value).fit(v_matrix)

    # Drawing plots for Spectral clustering
    label_color = [LABEL_COLOR_MAP[l] for l in kmeans.labels_]
    plt.scatter(train_data[:, 0], train_data[:, 1], c=label_color)
    plt.title("Spectral Clustering for sigma:" + str(sigma))
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

    print("Sigma:", sigma, "Loss for Spectral Clustering:", calculate_loss(train_data, kmeans.labels_))