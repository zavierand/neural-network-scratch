# modules
from network import NeuralNetwork

# dependencies
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor

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
