import numpy as np
import matplotlib.pyplot as plt

# This array is obtained from the adaboost.py file which contains iterations and their train and test accuracies
arr = np.array([[1, 0.8125, 0.8021390374331551], [2, 0.8125, 0.8021390374331551], [3, 0.8, 0.839572192513369], [4, 0.825, 0.8074866310160428], [5, 0.8375, 0.7967914438502673], [6, 0.8875, 0.8074866310160428], [7, 0.8875, 0.7914438502673797], [8, 0.9125, 0.786096256684492], [9, 0.9, 0.8074866310160428], [10, 0.925, 0.7914438502673797]])

# Extracting iteration numbers, train and test accuracies
iterations = arr[:, 0, None].reshape(1,10)[0]
train_accs = arr[:, 1, None].reshape(1,10)[0]
test_accs = arr[:, 2, None].reshape(1,10)[0]

# Displaying both test and train plot lines and setting labels and titles
plt.plot(iterations, train_accs, label = "Training accuracies per iteration")
plt.plot(iterations, test_accs, label = "Testing accuracies per iteration")
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('Iteration Vs Accuracy')

# Displays legend
plt.legend()

# Shows plot
plt.show()
