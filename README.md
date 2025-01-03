# Project
This is a simple neural network from scratch. The goal of neural network is to cluster points as best as possible as they are being added.

**please note the backpropagation section needs a rewrite**

# Overview
As many of us know, a neural network is a machine learning technique that consists of three types of layers - the input layer, hidden layers, and output layers. 

# Algorithms
The algorithms used in this are all rely on simple linear algebra, probability theory, and simple calculus. Different from supervised learning - where we have our input features AND target labels, we have no target labels, $\hat{y}$ ,to use while training in a neural network. Hence, if trying to predict the point ${x, \hat{y}}$, we would need to approach it through an unsupervised learning approach.

In a neural network, given the 3 types of layers: input, hidden, output - we can predict the points through a set of functions. With neural networks, the architecture is simple to understand on a high level, but practically much more complex.

A neural network works very simply - the data is forward fed where activation functions process the data from the input layer all the way through to the output layer. This is also known as **forward propagation**. On error calculation, we propagate backwards and essentially just see how far off our computations were. This is more commonly known as **back propagation**.

## Activation Functions in Forward Propagation
### ReLU:
An activation fucntion is a function used in forward propagation to help process data and add non-linearity to the output from the input neuron from the input layer all the way to $n - 1$ layers - where $n$ is the number of layers in the network including both the input and output layer. There are many activation functions we can use to process our data, but for this network, we will use the following activation function, ReLu - an acronym for "Rectified Linear Unit". If given an input matrix, $A$, ReLu can be expressed as the following function:

$$ \text{relu}(x) = \text{max}(0, x) $$

where we simply return the max of 0 and $x$, the output from the the layer before. We can denote it as a piecewise function as well:

$$
\text{relu}(x) =
    \begin{cases} 
    0 & \text{if } x < 0, \\
    x & \text{if } x \geq 0
    \end{cases}
$$

Here is a visual of ReLU:
![ReLU activation function](./images/relu.png)

### Softmax:
As the neural network traverses the layers from the input -> hidden layers -> output our activation functions do change. Upon reach the $n - 1$ layer, the activation function used is called the "Softmax Activation function". The Softmax activation functions converts logits into probabilities, making it ideal for classification tasks. Softmax can be expressed as the following function:

$$
p_i = \frac{\text{exp}(z_i)}{\sum_{j=1}^{n}\text{exp}(z_j)}
$$

Essentially, we just compute the expected values produced from the layer before and generate a probability distribution for the output layer, where we then return the max value of the distribution as our predicted guess, $\hat{y}$.

### Activation Function Goal
The goal of the an activation function is to add non-linearity to processing. Thinking about this intuitively, we use the following equation to propagate forward:

$$
f_{w,b}(x) = w_1x_1 + w_2x_2 + ... + w_nx_n + b
$$

where $w$, are the weights and $b$ is the bias per neuron. We can see this as a glorified regression model ($\hat{y} = wx^{i} + b$). To add some non-linearity to forward processing, we apply an activation function. This function essentially lets the model recognize complex patterns in data, that otherwise would have been glossed over.

## Calculating Loss with our Error Function
Thinking back to a linear regression model, we can calculate the loss of a neural network using *Cross Entropy Loss*. Cross entropy takes the true probability of an input, multiplying it by the log of it and takes the negative sum over all inputs. This is expressed as this formula:

$$
H(P^\*|P) = -\sum_{i}^{} P^{*} \log(P(i))
$$

This is a dominant loss function in calculating error in machine learning, specifically with neural networks and classification problems - which is what this project is!

## Backpropagation
Backpropagation is the algorithm used for training neural networks. To update the weights and biases of the network, we take the partial derivatives, using chain rule, of each layer and then we update the weights and biases accordingly.

Not going too much into the math, but to calculate the cost of the neural network, we see that we have a recurrence relation with the number of weights. Therefore, we can simply take the partial derivative of the error function using chain rule, with respect to each weight, $w$, and we get the following:

$$
\frac{\partial J}{\partial W_n} = 2(c - X_p) \cdot \prod_{n=1}^{p} \frac{\partial f(W_n, X_{n-1})}{\partial X_{n-1}}.
$$

This, on a very high level, is backpropagation.

### Derivatives of the Activation Functions
Seeing how backprop works, we can assume that the derivatives of the activation functions can help train the network in a surprisingly effective way. When looking at the derivative of ReLU, $\text{relu}(x) = \text{max}(0, x)$, notice that we get this:

$$
f'(x) =
    \begin{cases} 
        1 & \text{if } x > 0, \\
        0 & \text{if } x \leq 0
    \end{cases}
$$

Essentially, this "deactivates" a neuron on training, influencing which neurons should be considered during forward pass. This works in conjunction with gradient descent to improve accuracy rates by updating the weights and biases of the neurons while also influencing paths a network can take to reach a desired output.

## Gradient Descent
To compute the best possible values for our weights and biases in our neural network, we can use gradient descent (of course). Gradient descent is obviously one of - if not, the most popular algorithm used for updating the parameters to be as optimal as possible. Given an input vector of values, we can compute the optimal values for our weights and biases with the following two equations:

$$
w = w - \alpha \cdot \frac{\partial J}{\partial w}
$$
$$b = b - \alpha \cdot \frac{\partial J}{\partial b}$$

where $w$ and $b$ are the weights and biases associated with each edge interlinking the neurons in different layers.

# Architecture
I wanted to be able to cluster data points into three categories, similar to a k-means or k-nearest neighbors - both clustering algorithms used in different situations. There are other clustering algorithms that work, but these were the first that popped into my head.

Arbitrarily, I have set k = 3 - meaning I want the neural network to cluster each point that's added to the data set into three clusters.

# Program
There are multiple files in this repo, [the model itself](network.py), the [interactive "game"](main.py), and a [test](spiral-test.py) on the model.

### network.py
This is the source code for the network. All of the math and theory mentioned earlier in the file is incorporated here. Decided to approach this with an object-oriented approach to allow for modularity across the different files.

### main.py
The interactive ui lets users add a point to the network and watch the model create clusters as it learns. Here is one output of the network 

### spiral-test.py
The spiral test data set was sourced from the following case-study: [Case Study Link](https://cs231n.github.io/neural-networks-case-study/). On most runs, after 10000 epochs, we get an accuracy range of 98.6 to 99.3 percent.

# Running the program
You can clone this repo:
```
git clone https://github.com/zavierand/neural-network-scratch
```

Since this was built in python and NumPy, you don't need to install a lot of dependencies. Run this to install any of the dependencies you need:
```
pip install numpy
pip install matplotlib
```

You can run the following command to run the model.:
```
python3 network.py
```

or these commands to run the interactive ui or the spiral-test
```
python spiral-test.py
python main.py
```

Here is one output of the test:
![spiral test](./.outputs/spiral-test-output2.jpg)
Here, we can see the model has an accuracy of 99.3% at 9000 epochs.

# And so...
This is the math behind how a neural network generates output and trains itself. It may seem daunting, but in practice, the math is not as bad as it seems (though it can still be not fun at times).

# Resources:
[Neural Network from Scratch](https://nnfs.io) - [Sentdex](https://www.youtube.com/@sentdex)

[Stanford CS231: Neural Networks for Visual Recognition](https://cs231n.github.io)

Some of my notes from Professor [Ernest Davis](https://cs.nyu.edu/~davise/) from Fall 2024 Semester - CSCI-UA 472

Russell, S. J., & Norvig, P. (2016). [Artificial Intelligence: a Modern Approach (3rd ed.)](https://aima.cs.berkeley.edu). Pearson.

Bishop, C. M. (2006). [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf). Springer

[Building a neural network from scratch](https://www.youtube.com/watch?v=w8yWXqWQYmU&t=834s&pp=ygUcbmV1cmFsIG5ldHdvcmtzIGZyb20gc2NyYXRjaA%3D%3D)

‌