import numpy as np
import quadprog as qp

# Opening file and reading contents
mystery_file = open('mystery.data','r')
file_contents = ""
if mystery_file.mode == 'r':
    file_contents = mystery_file.read()

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

train_data_length = len(train_data)

# Quad prog wrapper (Reference: https://scaron.info/blog/quadratic-programming-in-python.html)
def quadprog_solve_qp(P, q, G=None, h=None):
    quad_G = .5 * P   # make sure P is symmetric
    quad_a = q
    quad_C = -G.T
    quad_b = -h
    meq = 0
    return qp.solve_qp(quad_G, quad_a, quad_C, quad_b, meq)[0]

def add_features(train_data, train_data_length):
    new_train_data = np.empty((1000,15), dtype=float)

    # Feature vector of (x0,x1,x2,x3) is (x02,x12,x22,x32,x0*x1,x0*x2,x0*x3,x1*x2,x1*x3,x2*x3,x0,x1,x2,x3)
    for k in range(train_data_length):
        new_train_data[k, 0] = train_data[k, 0] * train_data[k, 0]
        new_train_data[k, 1] = train_data[k, 1] * train_data[k, 1]
        new_train_data[k, 2] = train_data[k, 2] * train_data[k, 2]
        new_train_data[k, 3] = train_data[k, 3] * train_data[k, 3]
        new_train_data[k, 4] = train_data[k, 0] * train_data[k, 1]
        new_train_data[k, 5] = train_data[k, 0] * train_data[k, 2]
        new_train_data[k, 6] = train_data[k, 0] * train_data[k, 3]
        new_train_data[k, 7] = train_data[k, 1] * train_data[k, 2]
        new_train_data[k, 8] = train_data[k, 1] * train_data[k, 3]
        new_train_data[k, 9] = train_data[k, 2] * train_data[k, 3]
        new_train_data[k, 10] = train_data[k, 0]
        new_train_data[k, 11] = train_data[k, 1]
        new_train_data[k, 12] = train_data[k, 2]
        new_train_data[k, 13] = train_data[k, 3]

        # Y values from old training data to new training data
        new_train_data[k, 14] = train_data[k, 4]

    return new_train_data

train_data = add_features(train_data, train_data_length)

#Iterate data and define matrices for quad prog minimize (1/2)xTPx+qTx subject to Gx≤h and Ax=b
P = np.zeros((15,15))
P[0,0] = P[1,1] = P[2,2] = P[3,3] = P[4,4] = P[5,5] = P[6,6] = P[7,7] = P[8,8] = P[9,9] = P[10,10] = P[11,11] = P[12,12] = P[13,13] = 1
# Since quad prog function expects a positive definite matrix, this value represents b
P[14,14]= 1

q = np.zeros((15,1)).reshape((15,))
h = -np.ones((train_data_length,1)).reshape((train_data_length,))
G = np.zeros((train_data_length,15))

for l in range(train_data_length):
    y = train_data[l, 14]
    G[l, 0] = -train_data[l, 0] * y
    G[l, 1] = -train_data[l, 1] * y
    G[l, 2] = -train_data[l, 2] * y
    G[l, 3] = -train_data[l, 3] * y
    G[l, 4] = -train_data[l, 4] * y
    G[l, 5] = -train_data[l, 5] * y
    G[l, 6] = -train_data[l, 6] * y
    G[l, 7] = -train_data[l, 7] * y
    G[l, 8] = -train_data[l, 8] * y
    G[l, 9] = -train_data[l, 9] * y
    G[l, 10] = -train_data[l, 10] * y
    G[l, 11] = -train_data[l, 11] * y
    G[l, 12] = -train_data[l, 12] * y
    G[l, 13] = -train_data[l, 13] * y
    G[l, 14] = -y

# Calling quad prog function to find w and b for minimum cost
wb_vector = quadprog_solve_qp(P,q,G,h)

print("Vector w:", wb_vector[0:14], " b:", wb_vector[14])

# Calculating margin
print("Margin: ",  (1 / np.sqrt((wb_vector[0:14] ** 2).sum())))