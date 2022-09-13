import numpy as np
from rgb import display_RGB_colour


# Define Neuron class:
class NeuralNetwork():
    """An n-layer neural network (i.e. a multi-layer perceptron).

    CAUTION: Please make sure the matrix dimensions are correct when setting own 'Ws' param.

    Args:
        X (np.array): Initial input vector.
        y (np.array): Initial y_true.
        Ws (np.array): Optional. List of weights.
    """
    def __init__(self, X, y, bias=1, eta=0.1, n_nodes=2, n_layers=2, Ws=None, linear=False):
        # Create list of LAYERS:
        self.layers = []
        self.layers.append(X) # Append input layer.
        for i in range(n_layers-1):
            self.layers.append(np.zeros((n_nodes, 1))) # Append hidden layers.
        self.layers.append(np.zeros(y.shape)) # Append output layer.

        # Create list of WEIGHTS:
        if Ws is None:
            self.Ws = []
            for i in range(n_layers):
                self.Ws.append(np.random.rand(self.layers[i+1].shape[0], self.layers[i].shape[0]))
        else:
            self.Ws = Ws

        # Create list of BIASES:
        self.biases = []
        for i in range(n_layers):
            self.biases.append(np.ones((self.Ws[i].shape[0], self.layers[i].shape[1]))*bias) # Multiply bias.

        # Set the other parameters:
        self.y_true = y
        self.eta = eta
        self.linear = linear
        self.n_layers = n_layers


    def activ_func(self, x):
        """Activation function used during forward pass.
        """
        # For linear:
        if self.linear is True:
            return x
        # # For ReLU:
        # elif self.function == 'relu':
        # 	return x * (x > 0)
        # # For tanh:
        # elif self.function == 'tanh':
        # 	return np.tanh(x)
        # For sigmoid:
        else:
            return 1.0/(1.0 + np.exp(-x))


    def forwardpass(self):
        """Runs the forward pass algorithm using the internal state (via self).
        """
        for i in range(self.n_layers):
            self.layers[i+1] = self.activ_func(np.dot(self.Ws[i], self.layers[i]) + self.biases[i])


    def activ_deriv(self, x):
        """Derivative of the activation function used during backpropagation.
        """
        # For linear:
        if self.linear is True:
            return 1.0
        # # For ReLU:
        # elif self.function == 'relu':
        # 	return 1.0 * (x > 0)
        # # For tanh:
        # elif self.function == 'tanh':
        # 	return 1.0 - np.tanh(x)**2
        # For sigmoid:
        else:
            return self.activ_func(x)*(1-self.activ_func(x))


    def error_deriv(self):
        """Derivative of the error function used during backpropagation.
        """
        return -(self.y_true-self.layers[-1])


    def error(self):
        """Error function.
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
        for i in reversed(range(self.n_layers)): # NB: reversed for backpropagation.
            # Calculate the deriv wrt. activation:
            d_activ = self.activ_deriv(x=np.dot(self.Ws[i], self.layers[i]))
            
            # Delta for output layer:
            if i == self.n_layers-1:
                delta = self.error_deriv() * d_activ
                
            # Delta for subsequent layers:
            else:
                delta = np.dot(deltas[0].T, self.Ws[i+1]).T * d_activ # NB: uses the prev delta and prev layer.
                
            # Save delta to list:
            deltas.insert(0, delta) # NB: undo reversed order.

        # Iterate over deltas and apply both kinds of updates:
        for i in range(self.n_layers):
            # Update weight:
            self.Ws[i] += -self.eta * np.dot(deltas[i], self.layers[i].T)
            
            # Update bias:
            self.biases[i] += -self.eta * deltas[i] * self.biases[i]


    def fit(self, Xs, ys, iterations=1):
        """Applies the forward pass and backpropagation algorithms in sequence to fit given training data.

        Args:
            Xs (np.array): List of training data vectors.
            ys (np.array): List of training target vectors.
            iterations (int): Number of times to repeat the sequence over whole dataset.
        """
        y_preds = []
        errors = []
        for iteration in range(iterations):
            for i, X in enumerate(Xs):
                # Reset inputs:
                self.layers[0] = X  # X assigned to input layer.
                self.y_true = ys[i] # y assigned to y_true.
                
                self.forwardpass()
                self.backprop()
                
                # Save the final interation of output layer:
                if iteration == iterations-1:
                    y_preds.append(self.layers[-1])
                    errors.append(self.error())
        return np.array(y_preds), np.array(errors)


    def predict(self, Xs):
        """Applies forward pass using the internal state to the given input data (Xs).

        Args:
            Xs (np.array): Input data.
        """
        y_preds = []
        for X in Xs: # Per data point.
            self.layers[0] = X # X assigned to input layer.
            self.forwardpass()
            y_preds.append(self.layers[-1])
        return np.array(y_preds)


    def display_test_results(self, Xs, y_preds):
        """Will plot a figure of a given colour (via Xs) and its predicted text colour (via y_preds).

        NB: specific to the "text predictor" scenario.
        
        Args:
            Xs (np.array): Input data.
            y_preds (np.array): Predicted colours.
        """
        for i, y in enumerate(y_preds):
            if y == 0:
                print('\n--->\t{}:\tlight text'.format(y))
                display_RGB_colour(colour=tuple(Xs[i, :]), font_col='#fff')
            else:
                print('\n--->\t{}:\dark text'.format(y))
                display_RGB_colour(colour=tuple(Xs[i, :]), font_col='#000')