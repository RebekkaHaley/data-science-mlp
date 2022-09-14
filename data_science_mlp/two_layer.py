"""
Python class housing a two-layer neural network.
"""

import numpy as np


# Define Neuron class:
class NeuralNetwork():
    """A 2-layer neural network (i.e. a multi-layer perceptron).

    Args:
        x_input (np.array): Initial input vector.
        y_input (np.array): Initial y_true vector.
        bias (float): ...
        weights_1 (np.array): List of weights.
        weights_2 (np.array): List of weights.
        num_nodes (int): ...
        eta (float): Adjusts step size for gradient descent.
        linear (bool): Use linear activation function when true, else use signmoid.
    """
    def __init__(self, weights_1=None, weights_2=None, num_nodes=2, bias=1, **kwargs):
        # Initialise internal state of network.
        self.x_input = kwargs['x_input']
        self.y_input = kwargs['y_input']
        self.linear = kwargs['linear']  # set to False for sigmoid
        self.eta = kwargs['eta']  # default: eta=0.1
        if weights_1 is None:
            self.weights_1 = np.random.rand(num_nodes, self.x_input.shape[0])
        else:
            self.weights_1 = weights_1
        if weights_2 is None:
            self.weights_2 = np.random.rand(self.y_input.shape[0], num_nodes)
        else:
            self.weights_2 = weights_2
        self.bias_1 = np.ones((self.weights_1.shape[0], self.x_input.shape[1]), dtype=float)*bias
        self.bias_2 = np.ones(self.y_input.shape, dtype=float)*bias
        self.output = np.zeros(self.y_input.shape)
        self.layer1 = None


    def activ_func(self, num):
        """Calculates activation function used during forward pass.
        """
        if self.linear is True: # For linear:
            return num
        # For sigmoid:
        return 1.0/(1.0 + np.exp(-num))


    def forwardpass(self):
        """Runs the forward pass algorithm using the internal state (via self).
        """
        self.layer1 = self.activ_func(np.dot(self.weights_1, self.x_input) + self.bias_1)
        self.output = self.activ_func(np.dot(self.weights_2, self.layer1) + self.bias_2)


    def activ_deriv(self, num):
        """Calculates derivative of the activation function used during backpropagation.
        """
        if self.linear is True: # For linear:
            return 1
        # For sigmoid:
        return self.activ_func(num)*(1-self.activ_func(num))


    def error_deriv(self):
        """Calculates derivative of the error function used during backpropagation.
        """
        return -(self.y_input-self.output)


    def backprop(self):
        """Runs backpropagation algorithm using the internal state (via self).

        Steps:
        (1) applies chain rule to find derivative of loss function;
        (2) updates the weights and biases with the gradient of the loss function.
        """
        # Output layer:
        active_derivative_1 = self.activ_deriv(num=np.dot(self.weights_2, self.layer1))
        big_delta = self.error_deriv() * active_derivative_1
        output_unit = -self.eta * np.dot(big_delta, self.layer1.T)

        # Hidden layer:
        active_derivative_2 = self.activ_deriv(num=np.dot(self.weights_1, self.x_input))
        sml_delta = np.dot(big_delta.T, self.weights_2).T * active_derivative_2
        hidden_unit = -self.eta * np.dot(sml_delta, self.x_input.T)

        # Update the weights and biases with the derivative (slope) of the loss function.
        # Weights:
        self.weights_2 += output_unit
        self.weights_1 += hidden_unit
        # Biases:
        self.bias_2 += -self.eta * big_delta * self.bias_2
        self.bias_1 += -self.eta * sml_delta * self.bias_1


    def fit(self, x_inputs, y_inputs, iterations=20):
        """Applies forward pass and backpropagation algorithms in sequence to fit training data.
        """
        y_preds = []
        for i, x_input in enumerate(x_inputs): # Per data point:
            self.x_input = x_input
            self.y_input = y_inputs[i]
            for i in range(iterations): # Per iteration:
                self.forwardpass()
                self.backprop()
            y_preds.append(self.output)
        return np.array(y_preds)


    def predict(self, x_inputs):
        """Applies forward pass using the internal state to the given input data.
        """
        y_preds = []
        for x_input in x_inputs:
            self.x_input = x_input
            self.forwardpass()
            y_preds.append(self.output)
        return np.array(y_preds)
