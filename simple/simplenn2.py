'''
    Simple neural network example for non-linear problems with 1 hidden layer
'''

import numpy as np

# Define activation function
def sigmoid(x, deriv=False):
    if (deriv == True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

X = np.array([
    [1,0,1],
    [0,1,1],
    [0,0,1],
    [1,1,1]
])

Y = np.array([[0,0,1,1]]).T

np.random.seed(1)

# Initialize weights, zero centered
w0 = 2 * np.random.random((3,4)) - 1
w1 = 2 * np.random.random((4,1)) - 1

for i in xrange(1000):

    # Forward propagation
    l0 = X
    l1 = sigmoid(np.dot(l0, w0))
    l2 = sigmoid(np.dot(l1, w1))

    # Calculate output error
    l2_error = Y - l2
    print(l2_error)

    l2_delta = l2_error * sigmoid(l2, True)

    # Calculate layer 1 error
    l1_error = np.dot(l2_delta, w1.T)

    l1_delta = l1_error * sigmoid(l1, True)

    # Update weights
    w0 += np.dot(l0.T, l1_delta)
    w1 += np.dot(l1.T, l2_delta)

print "Output after trainig"
print(l2)
