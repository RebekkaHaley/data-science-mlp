"""
Python class housing an n-layer neural network.
"""

import numpy as np


# Define Neuron class:
class NeuralNetwork():
    """An n-layer neural network (i.e. a multi-layer perceptron).

    CAUTION: Please make sure the matrix dimensions are correct when setting own 'weights' param.

    Args:
        x_input (np.array): Initial input vector.
        y_input (np.array): Initial y_true.
        weights (np.array): List of weights. Can be set to None.
        linear (bool): Use linear activation function when true, else use signmoid.
    """
    def __init__(self, bias=1, eta=0.1, n_nodes=2, n_layers=2, **kwargs):
        x_input = kwargs['x_input']
        y_input = kwargs['y_input']
        weights = kwargs['weights']  # can be set to None
        linear = kwargs['linear']  # set to False for sigmoid

        # Create list of LAYERS:
        self.layers = []
        self.layers.append(x_input)  # append input layer
        for i in range(n_layers-1):
            self.layers.append(np.zeros((n_nodes, 1)))  # append hidden layers
        self.layers.append(np.zeros(y_input.shape))  # append output layer

        # Create list of WEIGHTS:
        if weights is None:
            self.weights = []
            for i in range(n_layers):
                weight = np.random.rand(self.layers[i+1].shape[0], self.layers[i].shape[0])
                self.weights.append(weight)
        else:
            self.weights = weights

        # Create list of BIASES:
        self.biases = []
        for i in range(n_layers):
            # Multiply bias:
            self.biases.append(np.ones((self.weights[i].shape[0], self.layers[i].shape[1]))*bias)

        # Set the other parameters:
        self.y_true = y_input
        self.eta = eta
        self.linear = linear
        self.n_layers = n_layers


    def activ_func(self, x_input):
        """Calculates activation function used during forward pass.
        """
        # For linear:
        if self.linear is True:
            return x_input
        # # For ReLU:
        # elif self.function == 'relu':
        # 	return x_input * (x_input > 0)
        # # For tanh:
        # elif self.function == 'tanh':
        # 	return np.tanh(x_input)
        # For sigmoid:
        return 1.0/(1.0 + np.exp(-x_input))


    def forwardpass(self):
        """Runs the forward pass algorithm using the internal state (via self).
        """
        for i in range(self.n_layers):
            layer = np.dot(self.weights[i], self.layers[i]) + self.biases[i]
            self.layers[i+1] = self.activ_func(layer)


    def activ_deriv(self, x_input):
        """Calculates derivative of the activation function used during backpropagation.
        """
        # For linear:
        if self.linear is True:
            return 1.0
        # # For ReLU:
        # elif self.function == 'relu':
        # 	return 1.0 * (x_input > 0)
        # # For tanh:
        # elif self.function == 'tanh':
        # 	return 1.0 - np.tanh(x_input)**2
        # For sigmoid:
        return self.activ_func(x_input)*(1-self.activ_func(x_input))


    def error_deriv(self):
        """Calculates derivative of the error function used during backpropagation.
        """
        return -(self.y_true-self.layers[-1])


    def error(self):
        """Calculates error function.
        """
        return ((self.y_true-self.layers[-1])**2)*0.5


    def backprop(self):
        """Runs backpropagation algorithm using the internal state (via self).

        Steps:
        (1) applies chain rule to find derivative of loss function;
        (2) updates the weights and biases with the gradient of the loss function.
        """
        # Initialise lists to contain deltas:
        deltas = []

        # Iterate over n number of layers and calculate delta:
        for i in reversed(range(self.n_layers)):  # reversed for backpropagation
            # Calculate the deriv wrt. activation:
            d_activ = self.activ_deriv(x_input=np.dot(self.weights[i], self.layers[i]))

            # Delta for output layer:
            if i == self.n_layers-1:
                delta = self.error_deriv() * d_activ

            # Delta for subsequent layers:
            else:
                # Use prev delta and prev layer
                delta = np.dot(deltas[0].T, self.weights[i+1]).T * d_activ

            # Save delta to list:
            deltas.insert(0, delta)  # undo reversed order

        # Iterate over deltas and apply both kinds of updates:
        for i in range(self.n_layers):
            # Update weight:
            self.weights[i] += -self.eta * np.dot(deltas[i], self.layers[i].T)

            # Update bias:
            self.biases[i] += -self.eta * deltas[i] * self.biases[i]


    def fit(self, x_inputs, y_inputs, iterations=1):
        """Applies forward pass and backpropagation algorithms in sequence to fit training data.

        Args:
            x_inputs (np.array): List of training data vectors.
            y_inputs (np.array): List of training target vectors.
            iterations (int): Number of times to repeat the sequence over whole dataset.
        """
        y_preds = []
        errors = []
        for iteration in range(iterations):
            for i, x_input in enumerate(x_inputs):
                # Reset inputs:
                self.layers[0] = x_input  # x assigned to input layer
                self.y_true = y_inputs[i]  # y assigned to y_true

                self.forwardpass()
                self.backprop()

                # Save the final interation of output layer:
                if iteration == iterations-1:
                    y_preds.append(self.layers[-1])
                    errors.append(self.error())
        return np.array(y_preds), np.array(errors)


    def predict(self, x_inputs):
        """Applies forward pass using the internal state to the given input data.

        Args:
            x_inputs (np.array): Input data.
        """
        y_preds = []
        for x_input in x_inputs:  # per data point
            self.layers[0] = x_input  # x assigned to input layer
            self.forwardpass()
            y_preds.append(self.layers[-1])
        return np.array(y_preds)
