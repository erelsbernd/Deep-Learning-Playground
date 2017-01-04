#Using stochastic gradient descent with a feedforward neural network.

import random
import numpy as np


#class to represent a neural network
class Network(object):


    """ sizes is the number of neurons in each layer of the network. Biases and weights are randomly initialized using a Gaussian distribution with mean 0, and variance 1. """
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.size = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1])]



    #return output of network if 'a' is input    
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w,a) + b)
        return a


    """train the neural network using mini-batch stochastic gradient descent
    if test_data is provided then the network will be evaluated against it after each epoch and print out partial progress.  epochs is number of epochs to train for. eta is the learning rate. """
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data: n_test = len(test_data)
        n = len(training_data)
        
        #randomly shuffle the training data
        for j in xrange(epochs):
            random.shuffle(training_data)

            #partition shuffled training data into mini-batches of the appropriate size
            mini_batches = [training_data[k:k+mini_batch_size]
                            for k in xrange(0, n, mini_batch_size)]

            #for each mini_batch apply a single step of gradient descent
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print "Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)


    """ update weights and biases with gradient descent using backpropagation to a single mini batch/for every  training example in the mini_batch."""
    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            #call the backpropagation algorithm (compute gradient of cost function)
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb_dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta/len(mini_batch)) * nw for w, nw in zip(self.weights.nabla_w)]
        self.biases = [b - (eta/len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]


    """ return (nabla_b, nabla_w) representing the gradient for the cost function C_x.
    nabla_b and nalba_w are layer-by-layer lists of numpy arrays, like self.biases, self.weights """
    def backprop(self, x y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        #feedforward
        activation = x
        #list to store all the activations
        activations = [x]
        #list to store all the z vectors
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        #backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for 1 in xrange(2, self.num_layers):
            z = zs[-1]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-1+1].transpose(), delta) * sp
            nabla_b[-1] = delta
            nabla_w[-1] = np.dot(delta, activations[-1-1].transpose())
        return (nabla_b, nabla_w)


    """ return the number of test inputs for which the neural network outputs the correct result. The network's output is assumed to be the index of whichever neuron in the final layer has the highest activation. """
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)


    #return the vecor of partial derivates, partial C_x partial a for the output activations
    def cost_derivative(self, output_activations, y):
        return (output_activations - y)


    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    #computes the derivative of the sigmoid function
    def sigmoid_prime(z):
        return sigmoid(z) * (1 - sigmoid(z))
