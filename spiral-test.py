import numpy as np
import matplotlib.pyplot as plt
from network import NeuralNetwork  # Importing the NeuralNetwork class

# data generation sourced from:
#   https://cs231n.github.io/neural-networks-case-study/
# Generate the spiral dataset
N = 100  # number of points per class
D = 2    # dimensionality
K = 3    # number of classes

X = np.zeros((N*K, D))  # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8')  # class labels

for j in range(K):
    ix = range(N*j, N*(j+1))
    r = np.linspace(0.0, 1, N)  # radius
    t = np.linspace(j*4, (j+1)*4, N) + np.random.randn(N) * 0.2  # theta with noise
    X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]  # create the spiral points
    y[ix] = j  # label for the class

# Visualize the data
'''plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.title("Spiral Dataset")
plt.show()'''

# initialize the neural network with 2 input neurons (for 2D data) and 3 output neurons (for 3 classes)
nn = NeuralNetwork(input_size=D, output_size=K)

# encode the labels
y_one_hot = np.eye(K)[y]  # Convert labels to one-hot encoding

# train network
nn.train(X, y_one_hot, epochs=10000)

# plot decision boundary after training
def plot_decision_boundary(X, y, model):
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.forward(np.c_[xx.ravel(), yy.ravel()])
    Z = np.argmax(Z, axis=1)  # Choose the class with the highest probability
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.title("Decision Boundary after Training")
    plt.show()

# plot decision boundary after training
plot_decision_boundary(X, y, nn)
