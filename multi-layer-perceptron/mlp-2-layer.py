# Implementation of a 2-layer neural network (i.e. a multi-layer perceptron).

# Import libraries:
import numpy as np

# Define Neuron class:
class NeuralNetwork():
    def __init__(self, X, y, bias=1, eta=0.1, w1=None, w2=None, num_nodes=2, num_hidden=2, linear=False):
        self.X = X
        self.y = y
        self.eta = eta
        self.w1 = np.random.rand(num_nodes, self.X.shape[0]) if w1 is None else w1
        self.w2 = np.random.rand(self.y.shape[0], num_nodes) if w2 is None else w2
        self.b1 = np.ones((self.w1.shape[0], self.X.shape[1]), dtype=float)*bias
        self.b2 = np.ones(self.y.shape, dtype=float)*bias
        self.output = np.zeros(self.y.shape)
        self.linear = linear
    
    def activ_func(self, x):
        if self.linear is True: # For linear:
            return x
        else: # For sigmoid:
            return 1.0/(1.0 + np.exp(-x))
    
    def forwardpass(self):
        self.layer1 = self.activ_func(np.dot(self.w1, self.X) + self.b1)
        self.output = self.activ_func(np.dot(self.w2, self.layer1) + self.b2)
        
    def activ_deriv(self, x):
        if self.linear is True: # For linear:
            return 1
        else: # For sigmoid:
            return self.activ_func(x)*(1-self.activ_func(x))
    
    def error_deriv(self):
        return -(self.y-self.output)
    
    def backprop(self):
        '''Apply chain rule to find derivative of loss function.'''
        # Output layer:
        big_delta = self.error_deriv() * self.activ_deriv(x=np.dot(self.w2, self.layer1))
        output_unit = -self.eta * np.dot(big_delta, self.layer1.T)

        # Hidden layer:
        sml_delta = np.dot(big_delta.T, self.w2).T * self.activ_deriv(x=np.dot(self.w1, self.X))        
        hidden_unit = -self.eta * np.dot(sml_delta, self.X.T)
        
        '''Update the weights and biases with the derivative (slope) of the loss function.'''
        # Weights:
        self.w2 += output_unit
        self.w1 += hidden_unit
        
        # Biases:
        self.b2 += -self.eta * big_delta * self.b2
        self.b1 += -self.eta * sml_delta * self.b1
    
    def fit(self, Xs, ys, iterations=20):
        y_preds = []
        for i, X in enumerate(Xs): # Per data point:
            self.X = X
            self.y = ys[i]
            
            for i in range(iterations): # Per iteration:
                self.forwardpass()
                self.backprop()
                
            y_preds.append(self.output)
            
        return np.array(y_preds)
    
    def predict(self, Xs):
        y_preds = []
        for X in Xs:
            self.X = X
            self.forwardpass()
            
            y_preds.append(self.output)
            
        return np.array(y_preds)