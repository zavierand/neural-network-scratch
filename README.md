# Project
This is a simple neural network from scratch. The goal of neural network is to cluster points as best as possible as they are being added.

# Overview
As many of us know, a neural network is a machine learning technique that consists of three types of layers - the input layer, hidden layers, and output layers. 

# Algorithms
The algorithms used in this are all rely on simple linear algebra, proability theory, and simple calculus. Different from supervised learning - where we have our input features AND target labels, we have no target labels, $\hat{y}$ ,to use while training in a neural network. Hence, if trying to predict the point ${x, \hat{y}}$, we would need to approach it through an unsupervised learning approach.

In a neural network, given the 3 types of layers: input, hidden, output - we can predict the points through a set of functions. With neural networks, the architecture is simple to understand on a high level, but practically much more complex.

A neural network works very simply - the data is forward fed where activation functions process the data from the input layer all the way through to the output layer. This is also known as **forward propagation**. On error calculation, we propagate backwards and essentially just see how far off our computations were. This is more commonly known as **back propagation**.

## Activation Functions in Forward Propagation
An activation fucntion is From the input layer to the first hidden layer, we process our data with the following activation function, ReLu - an acronym for "Rectified Linear Unit". If given an input matrix, $A$, ReLu can be expressed as the following function:

$$ \text{relu}(x) = \text{max}(0, x) $$

where we simply return the max of 0 and $x$, the output from the the layer before. We can denote it as a piecewise function as well:

$$\[
\text(relu)(x) =
    \begin{cases} 
    0 & \text{if } x < 0, \\
    x & \text{if } x \geq 0.
    \end{cases}
\]$$

As the neural network traverses the layers from the input -> hidden layers -> output our activation functions do change. Upon reach the output - 1 layer, the activation function used is called the "Softmax Activation function". Softmax can be expressed as the following function:
$$
p_i = \frac{exp(z_i)}{\sum_{j=1}^{n}z_j}
$$

Essentially, we just compute the expected values produced from the layer before and generate a probability distribution for the output layer, where we then return the max value of the distribution as our predicted guess, $\hat{y}$.

## Backpropagation Algorithm
Backpropagation is the algorithm used for training neural networks. We simply calculate the cost of the neural network given the following error function:

$$J(w) = ||c - (f(w_1, w_2, ..., n_n))||^2$$

Not going too much into the math, but to calculate the cost of the neural network, we see that we have a recurrence relation with the number of weights. Therefore, we can simply take the partial derivative of the error function using chain rule, with respect to each weight, $w$, and we get the following:

$$
\frac{\partial J}{\partial W_n} = 2(c - X_p) \cdot \prod_{n=1}^{p} \frac{\partial f(W_n, X_{n-1})}{\partial X_{n-1}}.
$$

This, on a very high level, is backpropagation.

## Gradient Descent
To compute the best possible values for our weights and biases in our neural network, we can use gradient descent (of course). Gradient descent is obviously one of - if not, the most popular algorithm used for updating the parameters to be as optimal as possible. Given an input vector of values, we can compute the optimal values for our weights and biases with the following two equations:

$$
w = w - \alpha \cdot \frac{\partial J}{\partial w}
$$
$$b = b - \alpha \cdot \frac{\partial J}{\partial b}$$

where $w$ and $b$ are the weights and biases associated with each edge interlinking the neurons in different layers.

# And so...
This is the math behind how a neural network generates output and trains itself. It may seem daunting, but in practice, the math is not as bad as it seems (though it can still be not fun at times).