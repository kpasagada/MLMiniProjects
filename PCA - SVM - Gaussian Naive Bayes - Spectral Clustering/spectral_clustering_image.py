import skimage.io as img
import numpy as np
import sklearn.cluster as cluster

# Read image
image_data = img.imread("bw.png", as_gray=True)
image_data = np.array(image_data)

# Flatten image array
image_data = np.ndarray.flatten(image_data)
image_data_length = len(image_data)

# Function to compute similarity
def compute_similarity(vector1, vector2, sigma):
    return np.exp((-1 / (2 * (sigma ** 2))) * ((vector1 - vector2) ** 2))

# Function to compute the sum of squared loss
def calculate_loss(data, labels):
    sum_error = 0.0
    clusters = {}

    # Compute cluster centers
    for i in range(len(data)):
        if labels[i] not in clusters:
            clusters[labels[i]] = {'sum': 0.0, 'count': 0}
        clusters[labels[i]]['sum'] += data[i]
        clusters[labels[i]]['count'] += 1

    for key in clusters:
        clusters[key]['total'] = clusters[key]['sum'] / clusters[key]['count']

    # Computing loss
    for i in range(len(data)):
        sum_error += np.linalg.norm(data[i] - clusters[labels[i]]['total']) ** 2

    return sum_error

# Sigma array
sigma_array = [1, 10, 100, 1000, 10000]
k_value = 2

# Reshaping image data
image_data = image_data.reshape(-1, 1)

# Alternative Kmeans clustering
kmeans_alt = cluster.KMeans(n_clusters=k_value).fit(image_data)

# Loss for Kmeans clustering
print("Loss for K-Means:", calculate_loss(image_data, kmeans_alt.labels_))

# Unflatten kmeans labels array
cluster_labels = np.reshape(np.array(kmeans_alt.labels_).astype(np.float32), (75, 100))

# Saving image
img.imsave("bw_kmeans_alt.png", cluster_labels)

# Spectral clustering
for sigma in sigma_array:
    # Constructing similarity matrix A
    similarity_matrix = np.zeros((image_data_length, image_data_length))

    for i in range(image_data_length):
        for j in range(i+1, image_data_length):
            similarity_matrix[i, j] = similarity_matrix[j, i] = compute_similarity(image_data[i], image_data[j], sigma)
        similarity_matrix[i, i] = compute_similarity(image_data[i], image_data[i], sigma)

    # Construct diagonal matrix
    diagonal_matrix = np.zeros((image_data_length, image_data_length))
    sum_matrix = similarity_matrix.sum(axis=1)
    for i in range(image_data_length):
        diagonal_matrix[i, i] = sum_matrix[i]

    # Construct laplacian matrix
    laplacian_matrix = np.subtract(diagonal_matrix, similarity_matrix)

    # Computing eigen values and vectors
    eig_values, eig_vectors = np.linalg.eig(laplacian_matrix)
    eig_values = eig_values.real
    eig_vectors = eig_vectors.real

    # Computing non zero eigen values
    idx = np.transpose(np.nonzero(eig_values)).flatten()
    eig_values = eig_values[idx]
    eig_vectors = eig_vectors[:, idx]

    # Sorting eigen values and vectors
    idy = eig_values.argsort()
    eig_vectors = eig_vectors[:, idy]

    # Constructing V matrix
    v_matrix = eig_vectors[:, 0:k_value]

    # Kmeans clustering
    kmeans = cluster.KMeans(n_clusters=k_value).fit(v_matrix)

    # Unflatten spectral labels array
    cluster_labels = np.reshape(np.array(kmeans.labels_).astype(np.float32), (75, 100))

    # Saving image
    img.imsave("bw_spectral_" + str(sigma) + ".png", cluster_labels)

    print("Sigma:", sigma, "Loss for Spectral Clustering:", calculate_loss(image_data, kmeans.labels_))

