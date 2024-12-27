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

# hard code one input -> output layer 
inputs = np.array([[1, 2, 3, 2.5],
                  [2.0, 5.0, -1.0, 2.0],
                  [-1.5, 2.7, 3.3, -0.8]])

weights = np.array([[0.2, 0.8, -0.5, 1.0], 
                    [0.5, -0.91, 0.26, -0.5], 
                    [-0.26, -0.27, 0.17, 0.87]])

biases = np.array([2, 3, 0.5])

weights_2 = np.array([[0.1, -0.14, 0.5], 
                    [-0.5, -0.12, -0.33],
                    [-0.44, 0.73, -0.13]])

biases_2 = np.array([-1, 2, -0.5])

layer_one_output = np.dot(inputs, weights.T) + biases

layer_two_output = np.dot(layer_one_output, weights_2.T) + biases_2


print(f'layer 1 output:\n {layer_one_output}')
print()
print(f'layer 2 output:\n {layer_two_output}')



# class definition of a neuron
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

# end class definition