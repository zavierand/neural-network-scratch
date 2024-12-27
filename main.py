import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor

# Neural Network Components (same as previous)
class DenseLayer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backprop(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

class ActivationReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backprop(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class ActivationSoftmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backprop(self, dvalues, y_true):
        n_samples = dvalues.shape[0]
        if len(y_true.shape) == 1:
            y_true = np.eye(dvalues.shape[1])[y_true]
        self.dinputs = (dvalues - y_true) / n_samples

class CrossEntropyLoss:
    def loss(self, y_pred, y_true):
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred[range(len(y_pred)), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred * y_true, axis=1)
        losses = -np.log(correct_confidences)
        return np.mean(losses)

class NeuralNetwork:
    def __init__(self, input_size, output_size):
        self.learning_rate = 0.1
        self.dense1 = DenseLayer(input_size, 5)
        self.activation1 = ActivationReLU()
        self.dense2 = DenseLayer(5, 5)
        self.activation2 = ActivationReLU()
        self.dense3 = DenseLayer(5, 5)
        self.activation3 = ActivationReLU()
        self.dense4 = DenseLayer(5, output_size)
        self.activation4 = ActivationSoftmax()
        self.loss_function = CrossEntropyLoss()

    def forward(self, X):
        self.dense1.forward(X)
        self.activation1.forward(self.dense1.output)
        self.dense2.forward(self.activation1.output)
        self.activation2.forward(self.dense2.output)
        self.dense3.forward(self.activation2.output)
        self.activation3.forward(self.dense3.output)
        self.dense4.forward(self.activation3.output)
        self.activation4.forward(self.dense4.output)
        return self.activation4.output

    def compute_loss(self, y_pred, y_true):
        return self.loss_function.loss(y_pred, y_true)

    def backprop(self, X, y_true):
        self.activation4.backprop(self.activation4.output, y_true)
        self.dense4.backprop(self.activation4.dinputs)
        self.activation3.backprop(self.dense4.dinputs)
        self.dense3.backprop(self.activation3.dinputs)
        self.activation2.backprop(self.dense3.dinputs)
        self.dense2.backprop(self.activation2.dinputs)
        self.activation1.backprop(self.dense2.dinputs)
        self.dense1.backprop(self.activation1.dinputs)

    def update_weights(self):
        self.dense1.weights -= self.learning_rate * self.dense1.dweights
        self.dense1.biases -= self.learning_rate * self.dense1.dbiases
        self.dense2.weights -= self.learning_rate * self.dense2.dweights
        self.dense2.biases -= self.learning_rate * self.dense2.dbiases
        self.dense3.weights -= self.learning_rate * self.dense3.dweights
        self.dense3.biases -= self.learning_rate * self.dense3.dbiases
        self.dense4.weights -= self.learning_rate * self.dense4.dweights
        self.dense4.biases -= self.learning_rate * self.dense4.dbiases

    def train(self, X, y_true, epochs=500):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = self.compute_loss(y_pred, y_true)
            self.backprop(X, y_true)
            self.update_weights()

    def predict(self, X):
        y_pred = self.forward(X)
        return np.argmax(y_pred, axis=1)

    def plot_data(self, X, y, ax):
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=50, alpha=0.7)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_title('Neural Network Clusters')

    def plot_decision_boundaries(self, X, y, ax):
        # If there are no points in X, skip plotting boundaries
        if len(X) == 0:
            return

        # Generate grid for decision boundaries
        h = .02  # Step size for the mesh
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # Get predictions for each point in the mesh grid
        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Plot decision boundary
        ax.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())

# Interactive plot
def on_click(event, X, y, nn, ax, epochs=500):
    new_point = np.array([[event.xdata, event.ydata]])
    new_label = np.random.choice([0, 1, 2])  # Random label for demo

    # Add the new point and label
    X = np.vstack([X, new_point])
    y = np.append(y, new_label)

    y_one_hot = np.eye(3)[y]

    # Retrain the model with updated data
    nn.train(X, y_one_hot, epochs)

    # Predict with the updated model
    predictions = nn.predict(X)

    # Clear the previous plot and plot the updated data and decision boundaries
    ax.clear()
    nn.plot_decision_boundaries(X, predictions, ax)
    nn.plot_data(X, predictions, ax)
    plt.draw()

    return X, y  # Return updated X and y

# Initialize variables
X = np.empty((0, 2))  # Start with no points
y = np.empty(0, dtype=int)  # Start with no labels

# Initialize neural network
nn = NeuralNetwork(input_size=2, output_size=3)

# Set up the plot
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the decision boundaries and initial empty state
nn.plot_decision_boundaries(X, y, ax)
plt.title('Click to Add Points')

cursor = Cursor(ax, useblit=True, color='red', linewidth=1)

# Connect the click event and update X, y across multiple clicks
def update_on_click(event):
    global X, y
    X, y = on_click(event, X, y, nn, ax)

fig.canvas.mpl_connect('button_press_event', update_on_click)

# Show the plot
plt.show()
