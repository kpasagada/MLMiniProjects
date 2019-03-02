import numpy as np
import math as mt
from sklearn import preprocessing
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import silhouette_score

# Function to compute multivariate gaussian
def gaussian_function(row, mean_val, cov_value, num_of_features):
    diff_data_mean = np.array(row - mean_val).reshape(1, num_of_features)
    exp = np.exp(-0.5 * np.dot(np.dot(diff_data_mean, np.linalg.inv(cov_value)), diff_data_mean.T))
    return (1 / np.sqrt(((2 * mt.pi) ** num_of_features) * np.linalg.det(cov_value))) * exp

# Function to compute log likelihood
def compute_log_likelihood(data, mean_list, cov_list, lambda_list, k_value, num_of_features):
    log_sum = 0.0

    # Iterating all data
    for ii in range(len(data)):
        inner_sum = 0.0

        # Iterating all k
        for kk in range(k_value):
            inner_sum += lambda_list[kk] * gaussian_function(data[ii], mean_list[kk], cov_list[kk], num_of_features)

        log_sum += np.log(inner_sum)

    return log_sum

def gmm_predict(data, mean_value, covar_value, lambda_value, k_value, num_of_features):
    prediction = []

    # Iterating each data
    for pos in range(len(data)):

        best_likelihood = None
        best_cluster = None

        # Iterating each cluster k
        for k_num in range(k_value):

            # Computing likelihood value
            likelihood_value = lambda_arr[k_num] * gaussian_function(data[pos], mean_value[k_num], covar_value[k_num], num_of_features)

            # Check if best value
            if best_likelihood is None or best_likelihood <= likelihood_value:
                best_likelihood = likelihood_value
                best_cluster = k_num

        # Append to prediction array
        prediction.append(best_cluster)

    return prediction

# Generate k++ cluster centers or means
def kcluster_centers(data, k_value):

    # Choosing random center point
    cluster_centers = data[np.random.choice(np.arange(0, len(data), 1), 1)]

    # Until k cluster centers
    while len(cluster_centers) < k_value:

        distance_array = []

        # Iterating data points
        for ii in range(len(data)):

            best_distance = None

            # Iterating clusters
            for pos in range(len(cluster_centers)):

                # Compute distance
                dist = np.linalg.norm(data[ii] - cluster_centers[pos])

                # Check if best distance
                if best_distance is None or dist <= best_distance:
                    best_distance = dist

            distance_array.append(best_distance)

        # Computing distribution
        np.square(distance_array)
        p_distribution = distance_array / np.sum(distance_array)
        cluster_centers = np.append(cluster_centers, data[np.random.choice(np.arange(0, len(data), 1), p=p_distribution, size=1)], axis=0)

    return cluster_centers


############### TRAIN ####################
# Opening training file and reading contents
leaf_train_file = open('leaf.data','r')
file_contents = ""
if leaf_train_file.mode == 'r':
    file_contents = leaf_train_file.read()

leaf_train_file.close()

# Creating variable to parse and store train data
train_data = np.empty((340, 15), dtype=np.float64)

# Initialise row and column counters
i = 0
j = 0

# Read file to parse contents and store in numpy array
for line in file_contents.split('\n'):
    for dt in line.split(","):
        train_data[i, j] = dt
        j += 1
    j = 0
    i += 1

# Train data length and features length
train_data_length = len(train_data)
train_features = train_data[:, 0]
train_data = train_data[:, 1:]
features_length = len(train_data[0])

# Scaling data with mean zero and variance one
scaled_train_data = preprocessing.scale(train_data)

# K array
k_array = [12, 18, 24, 36, 42]

# Get GMM objective loss array and compute mean and variance
gmm_loss_array = []

# GMM Model stores
gmm_mean_array = []
gmm_covar_array = []
gmm_lambda_array = []

# For each cluster size k
for k in k_array:

    # For 20 iterations
    for i in range(20):

        # Initializing mean array
        mean_arr = kcluster_centers(scaled_train_data, k)

        # Initializing co-variance matrix
        cov_matrix_arr = np.empty((k, features_length, features_length))
        for j in range(k):
            cov_matrix_arr[j] = np.identity(n=features_length, dtype=np.float64)

        # Initializing lambda array
        lambda_arr = np.empty((k, 1), dtype=np.float64)
        for j in range(k):
            lambda_arr[j] = 1/k

        # Initial log likelihood value
        log_like_val = compute_log_likelihood(scaled_train_data, mean_arr, cov_matrix_arr, lambda_arr, k, features_length)
        iteration_counter = 1

        # Begin EM iterations
        while True:

            # E Step block
            q_array = np.empty((train_data_length, k), dtype=np.float64)

            # Iterating data
            for x in range(train_data_length):

                den_sum = 0.0

                # Iterating k values
                for k_val in range(k):
                    q_array[x, k_val] = lambda_arr[k_val] * gaussian_function(scaled_train_data[x], mean_arr[k_val], cov_matrix_arr[k_val], features_length)
                    den_sum += q_array[x, k_val]

                q_array[x] = q_array[x] / den_sum

            # M Step block
            # Updating mean array
            for k_val in range(k):
                num_total = 0.0
                den_total = 0.0

                for m in range(train_data_length):
                    num_total += q_array[m, k_val] * scaled_train_data[m]
                    den_total += q_array[m, k_val]

                mean_arr[k_val] = num_total / den_total

            # Updating covariance array
            for k_val in range(k):
                num_total = 0.0
                den_total = 0.0

                for m in range(train_data_length):
                    diff_vector = scaled_train_data[m] - mean_arr[k_val]
                    diff_vector = np.array(diff_vector).reshape((1, features_length))
                    num_total += q_array[m, k_val] * np.dot(diff_vector.T, diff_vector)
                    den_total += q_array[m, k_val]

                cov_matrix_arr[k_val] = num_total / den_total
                cov_matrix_arr[k_val] += np.identity(n=features_length)

            # Updating lambda array
            for k_val in range(k):
                num_total = 0.0

                for m in range(train_data_length):
                    num_total += q_array[m, k_val]

            lambda_arr[k_val] = num_total / train_data_length

            # Compute log likelihood value
            prev_log_like_val = log_like_val
            log_like_val = compute_log_likelihood(scaled_train_data, mean_arr, cov_matrix_arr, lambda_arr, k, features_length)

            # Status
            print("K value:", k, "Iteration:", i, "Counter:", iteration_counter)

            # Increment iteration
            iteration_counter += 1

            # Checking for convergence
            if prev_log_like_val >= log_like_val:
                gmm_loss_array.append(log_like_val)
                gmm_mean_array.append(mean_arr)
                gmm_covar_array.append(cov_matrix_arr)
                gmm_lambda_array.append(lambda_arr)
                break

# Mean and variance of converged log likelihood for each k
print("GMM objective for k: 12 - Mean:", np.mean(gmm_loss_array[0:20]), "Variance:", np.var(gmm_loss_array[0:20]))
print("GMM objective for k: 18 - Mean:", np.mean(gmm_loss_array[20:40]), "Variance:", np.var(gmm_loss_array[20:40]))
print("GMM objective for k: 24 - Mean:", np.mean(gmm_loss_array[40:60]), "Variance:", np.var(gmm_loss_array[40:60]))
print("GMM objective for k: 36 - Mean:", np.mean(gmm_loss_array[60:80]), "Variance:", np.var(gmm_loss_array[60:80]))
print("GMM objective for k: 42 - Mean:", np.mean(gmm_loss_array[80:100]), "Variance:", np.var(gmm_loss_array[80:100]))

# Predict clusters with k = 36
acc = 0.0
sil_acc = 0.0
temp_data = np.append(scaled_train_data, np.array(train_features - 1).reshape((train_data_length, 1)), axis=1)
for i in range(20):
    predict_array = gmm_predict(scaled_train_data, gmm_mean_array[60+i], gmm_covar_array[60+i], gmm_lambda_array[60+i], 36, features_length)
    acc += adjusted_rand_score(train_features-1, predict_array)
    sil_acc += silhouette_score(temp_data, predict_array, sample_size=20)

print("Average Similarity Measure (Adjusted Rand Index) of the GMM model with k: 36 is", acc/20)
print("Average Silhouette Coefficient for all samples of the GMM model with k: 36 is", sil_acc/20)
