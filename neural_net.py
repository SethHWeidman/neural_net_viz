import numpy as np

class NeuralNetwork(object):
    def __init__(self, layers, random_seed):
        self.layers = layers
        self.random_seed = random_seed
        for i, layer in enumerate(self.layers):
            setattr(layer, 'random_seed', self.random_seed+i)
            layer.initialize_weights()

    def forwardpass(self, X):
        """ Calculate an output Y for the given input X. """
        X_next = X
        for layer in self.layers:
            X_next = layer.fprop(X_next)
        prediction = X_next
        return prediction

    def loss(self, prediction, Y):
        """ Calculate error on the given data. """
        loss = 0.5 * (Y - prediction) ** 2
        return -1.0 * (Y - prediction)

    def backpropogate(self, loss):
        """ Calculate an output Y for the given input X. """
        loss_next = loss
        for layer in reversed(self.layers):
            loss_next = layer.bprop(loss_next)
        return loss

class Layer(object):
    def _setup(self, input_shape):
        """ Setup layer with parameters that are unknown at __init__(). """
        pass

    def fprop(self, input):
        """ Calculate layer output for given input (forward propagation). """
        raise NotImplementedError()

    def bprop(self, output_grad):
        """ Calculate input gradient. """
        raise NotImplementedError()

class Linear(Layer):

    random_seed = None

    def __init__(self, n_in, n_out,
                 activation_function):
        self.n_in = n_in
        self.n_out = n_out
        self.iteration = 0
        self.activation_function = activation_function

    def initialize_weights(self):
        np.random.seed(seed=self.random_seed)
        self.W = np.random.normal(size=(self.n_in, self.n_out))

    def fprop(self, layer_input):
        self.layer_input = layer_input
        self.activation_input = np.dot(layer_input, self.W)
        return self.activation_function(self.activation_input, bprop=False)

    def bprop(self, layer_gradient):
        dOutdActivationInput = self.activation_function(self.activation_input, bprop=True)
        dLayerInputdActivationInput = layer_gradient * dOutdActivationInput
        dActivationOutputdActivationInput = self.layer_input.T
        output_grad = np.dot(dLayerInputdActivationInput, self.W.T)
        weight_update = np.dot(dActivationOutputdActivationInput, dLayerInputdActivationInput)
        W_new = self.W - weight_update
        self.W = W_new
        self.iteration += 1
        return output_grad

def sigmoid(x, bprop=False):
    if bprop:
        s = sigmoid(x)
        return s*(1-s)
    else:
        return 1.0/(1.0+np.exp(-x))
