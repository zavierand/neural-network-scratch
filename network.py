import sys

# data storage
import numpy as np

# for math needed
import math

# for plotting data during testing
import matplotlib

# print(f'python version: {sys.version}')
# print(f'numpy version: {np.__version__}')
# print(f'matplot version: {matplotlib.__version__}')

np.random.seed(0)

# hard code one input -> output layer 
X = np.array([[1, 2, 3, 2.5],
              [2.0, 5.0, -1.0, 2.0],
              [-1.5, 2.7, 3.3, -0.8]])

class Layer:
    def __init__(self, n_inputs, n_neurons):
        # creates a gaussian distribution for weights
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    # forward feed data into neural network
    def forwardProp(self, inputs):
        # simply outputs a linear value
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output
        
layer1 = Layer(4, 5)
layer2 = Layer(5, 2)
print(f'layer one: \n{layer1.forwardProp(X)}')
print(f'layer two: \n{layer2.forwardProp(layer1.output)}')

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