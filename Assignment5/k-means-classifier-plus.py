import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import silhouette_score

# Accuracy calculation
def accuracy(Y1,Y2):
    counter = 0
    index = 0
    for val1 in Y1:
        if val1 == Y2[index]:
            counter += 1
        index += 1

    return counter/len(Y1)

############### TRAIN ####################
# Opening training file and reading contents
leaf_train_file = open('leaf.data','r')
file_contents = ""
if leaf_train_file.mode == 'r':
    file_contents = leaf_train_file.read()

leaf_train_file.close()

# Creating variable to parse and store train data
train_data = np.empty((340, 15), dtype=float)

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

# K Means model
kmeans_model_array = []

# For each cluster size k
for k in k_array:

    # For 20 iterations
    for i in range(20):

        # K Means model
        kmeans_model_array.append(KMeans(n_clusters=k).fit(scaled_train_data))

# Get KMeans objective loss array and compute mean and variance
kmeans_loss_array = []

for i in range(len(kmeans_model_array)):
    kmeans_loss_array.append(kmeans_model_array[i].inertia_)

print("K-means objective for k: 12 - Mean:", np.mean(kmeans_loss_array[0:20]), "Variance:", np.var(kmeans_loss_array[0:20]))
print("K-means objective for k: 18 - Mean:", np.mean(kmeans_loss_array[20:40]), "Variance:", np.var(kmeans_loss_array[20:40]))
print("K-means objective for k: 24 - Mean:", np.mean(kmeans_loss_array[40:60]), "Variance:", np.var(kmeans_loss_array[40:60]))
print("K-means objective for k: 36 - Mean:", np.mean(kmeans_loss_array[60:80]), "Variance:", np.var(kmeans_loss_array[60:80]))
print("K-means objective for k: 42 - Mean:", np.mean(kmeans_loss_array[80:100]), "Variance:", np.var(kmeans_loss_array[80:100]))

# Predict clusters with k = 36
acc = 0.0
sil_acc = 0.0
temp_data = np.append(scaled_train_data, np.array(train_features - 1).reshape((train_data_length, 1)), axis=1)
for i in range(20):
    predict_array = kmeans_model_array[60+i].predict(scaled_train_data)
    acc += adjusted_rand_score(train_features-1, predict_array)
    sil_acc += silhouette_score(temp_data, predict_array, sample_size=20)

print("Average similarity measure (adjusted rand index) of the K-Means model with k: 36 is", acc/20)
print("Average Silhouette Coefficient for all samples of the K-Means model with k: 36 is", sil_acc/20)