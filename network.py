import sys

# data storage
import numpy as np

# for math needed
import math

# for plotting data during testing
import matplotlib.pyplot as plt

# print(f'python version: {sys.version}')
# print(f'numpy version: {np.__version__}')
# print(f'matplot version: {matplotlib.__version__}')

np.random.seed(0)

# generate a data set to "test" our NN
# sourced from:
#   https://cs231n.github.io/neural-networks-case-study/

N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
X = np.zeros((N*K,D)) # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8') # class labels
for j in range(K):
    ix = range(N*j,N*(j+1))
    r = np.linspace(0.0,1,N) # radius
    t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
    X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
    y[ix] = j
# lets visualize the data:
'''plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.show()'''
# end of sourced block

class Layer:
    def __init__(self, n_inputs, n_neurons):
        # creates a "gaussian distribution" for weights
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    # we need to implement an activation function to add some linearity to the processing
    def relu(self, input):
        self.output = np.maximum(0, input)

    # forward feed data into neural network
    def forwardProp(self, inputs):
        # simply outputs a linear value
        self.output = np.dot(inputs, self.weights) + self.biases

        self.relu(self.output)

        return self.output
    
# test    
layer1 = Layer(2, 5)
layer2 = Layer(5, 2)
print(f'layer one: \n{layer1.forwardProp(X)}')
print(f'layer two: \n{layer2.forwardProp(layer1.output)}')

# display graph
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.show()

'''# class definition of a neuron
class Neuron:
    def __init__(self, weight, bias, inputs, outputs):
        self.weight = weight
        self.bias = bias
        self.input = inputs
        self.output = outputs

# class definition of a neural network
class NeuralNetwork:
    def __init__(self):
        pass
    
    def forward_prop(input, weight, bias):
        output = np.array()
        return output
    
    def back_prop(output, weight, bias):
        # calc partial derivatives
        return weight, bias

# end class definition'''