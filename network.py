import numpy as np
import matplotlib.pyplot as plt

# class handling functionality of each layer
class DenseLayer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backprop(self, dvalues):
        # Calculate gradients for weights, biases, and inputs
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

# class that holds forward prop and backprop algos
class ActivationReLU:
    def forward(self, inputs):
        self.inputs = inputs  # store inputs for backpropagation
        self.output = np.maximum(0, inputs)

    def backprop(self, dvalues):
        # gradient of ReLU: 0 where input was <= 0, 1 otherwise
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


# class that holds forward prop and backprop algos
class ActivationSoftmax:
    def forward(self, inputs):
        # subtract max value for numerical stability
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backprop(self, dvalues, y_true):
        # Number of samples
        n_samples = dvalues.shape[0]

        # ff labels are sparse (integers), convert to one-hot encoding
        if len(y_true.shape) == 1:
            y_true = np.eye(dvalues.shape[1])[y_true]

        # gradient of the combined softmax and cross-entropy
        self.dinputs = (dvalues - y_true) / n_samples

# class to calculate the error from the model prediction
class CrossEntropyLoss:
    def loss(self, y_pred, y_true):
        # clip values to prevent division by zero
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # handle case where y_true is one-hot or scalar
        if len(y_true.shape) == 1:
            correct_confidences = y_pred[range(len(y_pred)), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred * y_true, axis=1)

        # Negative log-likelihood
        losses = -np.log(correct_confidences)
        return np.mean(losses)

# class that contains architecture of neural network
class NeuralNetwork:
    def __init__(self, input_size, output_size):
        self.learning_rate = 0.1
        # Initialize the network with layers
        self.dense1 = DenseLayer(input_size, 5)  # First layer: input_size inputs, 5 neurons
        self.activation1 = ActivationReLU()

        self.dense2 = DenseLayer(5, 5)  # Second layer: 5 inputs, 5 neurons
        self.activation2 = ActivationReLU()

        self.dense3 = DenseLayer(5, 5)  # Third layer: 5 inputs, 5 neurons
        self.activation3 = ActivationReLU()

        self.dense4 = DenseLayer(5, output_size)  # Output layer: 5 inputs, output_size neurons (classes)
        self.activation4 = ActivationSoftmax()

        # Loss Function
        self.loss_function = CrossEntropyLoss()

    def forward(self, X):
        # Forward pass through the layers
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
        # Backpropagate through the layers (starting from output layer)
        self.activation4.backprop(self.activation4.output, y_true)  # Softmax & Cross-Entropy
        self.dense4.backprop(self.activation4.dinputs)

        self.activation3.backprop(self.dense4.dinputs)
        self.dense3.backprop(self.activation3.dinputs)

        self.activation2.backprop(self.dense3.dinputs)
        self.dense2.backprop(self.activation2.dinputs)

        self.activation1.backprop(self.dense2.dinputs)
        self.dense1.backprop(self.activation1.dinputs)

    def update_weights(self):
        # Update weights and biases using the gradients calculated during backpropagation
        self.dense1.weights -= self.learning_rate * self.dense1.dweights
        self.dense1.biases -= self.learning_rate * self.dense1.dbiases

        self.dense2.weights -= self.learning_rate * self.dense2.dweights
        self.dense2.biases -= self.learning_rate * self.dense2.dbiases

        self.dense3.weights -= self.learning_rate * self.dense3.dweights
        self.dense3.biases -= self.learning_rate * self.dense3.dbiases

        self.dense4.weights -= self.learning_rate * self.dense4.dweights
        self.dense4.biases -= self.learning_rate * self.dense4.dbiases

    def train(self, X, y_true, epochs=1000):
        accuracies = []
        # train nn
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.forward(X)

            # Compute loss
            loss = self.compute_loss(y_pred, y_true)

            # Compute accuracy
            acc = self.accuracy(y_pred, y_true)
            accuracies.append(acc)

            # Backpropagation
            self.backprop(X, y_true)

            # Update weights
            self.update_weights()

            if epoch % 1000 == 0:  # Print loss every 100 epochs
                print(f'Epoch {epoch} - Loss: {loss}, Accuracy: {acc}')

    def accuracy(self, y_pred, y_true):
        # convert prediction probabilities to class labels
        predictions = np.argmax(y_pred, axis=1)
        
        # convert one-hot encoded labels to class labels
        y_true_labels = np.argmax(y_true, axis=1)
        
        # ompare predictions to true labels and compute accuracy
        accuracy = np.mean(predictions == y_true_labels)
        return accuracy

    def plot_layer_output(self, layer_output, layer_name):
        """
        Plot the output of a given layer.
        """
        plt.figure(figsize=(6, 4))
        plt.imshow(layer_output, cmap='viridis', aspect='auto')
        plt.colorbar()
        plt.title(f'Output of {layer_name}')
        plt.xlabel('Neurons')
        plt.ylabel('Samples')
        plt.show()

    def plot_final_output(self):
        """
        Plot the final output (softmax probabilities).
        """
        plt.figure(figsize=(6, 4))
        plt.bar(range(len(self.activation4.output[0])), self.activation4.output[0], tick_label=[f'Class {i}' for i in range(3)])
        plt.title('Final Output: Softmax Probabilities')
        plt.xlabel('Class')
        plt.ylabel('Probability')
        plt.show()

    def plot_all_layers(self, X, y_true):
        """
        Plot the output of each layer in the network.
        """
        self.forward(X)  # Perform the forward pass

        # Plot outputs of each layer
        self.plot_layer_output(self.dense1.output, "Dense Layer 1")
        self.plot_layer_output(self.activation1.output, "ReLU Activation 1")

        self.plot_layer_output(self.dense2.output, "Dense Layer 2")
        self.plot_layer_output(self.activation2.output, "ReLU Activation 2")

        self.plot_layer_output(self.dense3.output, "Dense Layer 3")
        self.plot_layer_output(self.activation3.output, "ReLU Activation 3")

        self.plot_layer_output(self.dense4.output, "Dense Layer 4")
        self.plot_layer_output(self.activation4.output, "Softmax Activation")

        # Loss calculation
        loss = self.compute_loss(self.activation4.output, y_true)
        print("\nLoss:", loss)
        
        # Plot the final output (Softmax probabilities)
        self.plot_final_output()
    
# test neural network now
X = np.random.randn(3, 3)  # 3 samples, 3 features (changed to 3 features)
y = np.array([0, 0, 1])  # Class labels

# Initialize the neural network with 3 input neurons instead of 5
nn = NeuralNetwork(input_size=3, output_size=3)

# One-hot encode labels
y_one_hot = np.eye(3)[y]  # Convert to one-hot encoding

# Train the network
accuracies = nn.train(X, y_one_hot, epochs=10000)