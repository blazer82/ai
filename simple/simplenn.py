'''
    Simple linear neural network example
'''

import numpy as np

# Define activation function
def sigmoid(x, deriv=False):
    if (deriv == True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

X = np.array([
    [0,0,1],
    [0,1,1],
    [1,0,1],
    [1,1,1]
])

Y = np.array([[0,0,1,1]]).T

np.random.seed(1)

# Initialize weights, zero centered
w0 = 2 * np.random.random((3,1)) - 1

for i in xrange(1000):

    # Forward propagation
    l0 = X
    l1 = sigmoid(np.dot(l0, w0))

    # Calculate output error
    l1_error = Y - l1
    print(l1_error)

    l1_delta = l1_error * sigmoid(l1, True)

    # Update weights
    w0 += np.dot(l0.T, l1_delta)

print "Output after trainig"
print(l1)
