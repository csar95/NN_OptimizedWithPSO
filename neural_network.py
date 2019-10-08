import numpy as np


class NeuralNetwork:

    activation_functions = ('null', 'sigmoid', 'hyperbolic_tangent', 'cosine', 'gaussian')

    def __init__(self):
        self.layers_weights = []
        self.layers_biases = []
        self.layers_activation = []

    # activation can be: {null, sigmoid, hyperbolic_tangent, cosine, gaussian}
    def add(self, neurons, input_shape=None):
        if not self.layers_weights:
            weights = np.random.uniform(-1, 1, size=(neurons, input_shape))
        else:
            #                                               Number of rows in the previous layer
            weights = np.random.uniform(-1, 1, size=(neurons, self.layers_weights[-1][:, 0].size))
        bias = np.random.uniform(-1, 1, size=(neurons, 1))
        activation = np.random.choice(self.activation_functions, size=(neurons, 1))
        self.layers_weights.append(weights)
        self.layers_biases.append(bias)
        self.layers_activation.append(activation)

    def set_parameters(self, pso_array, pso_activation):
        # Sets the weights for each neuron
        temp = list(pso_array.copy())
        for layer in range(len(self.layers_weights)):
            for weights in range(len(self.layers_weights[layer])):
                for weight in range(len(self.layers_weights[layer][weights])):
                    self.layers_weights[layer][weights][weight] = temp[0]
                    temp.pop(0)
        # Sets the bias for each neuron
        for layer in range(len(self.layers_biases)):
            for biases in range(len(self.layers_biases[layer])):
                for bias in range(len(self.layers_biases[layer][biases])):
                    self.layers_biases[layer][biases][bias] = temp[0]
                    temp.pop(0)
        # Sets the activation function for each neuron
        temp = list(pso_activation.copy())
        for layer in range(len(self.layers_activation)):
            for activations in range(len(self.layers_activation[layer])):
                for activation in range(len(self.layers_activation[layer][activations])):
                    self.layers_activation[layer][activations][activation] = temp[0]
                    temp.pop(0)

    def feed_forward(self, inputs):
        layer_inputs = np.reshape(inputs, newshape=(-1, 1))
        for i in range(len(self.layers_weights)):
            # Calculating the layer outputs
            layer_outputs = np.dot(self.layers_weights[i], layer_inputs)
            layer_outputs = np.add(layer_outputs, self.layers_biases[i])
            # Activation function
            if self.layers_activation[i] == 'null':
                self.map(self.null, layer_outputs)
            elif self.layers_activation[i] == 'sigmoid':
                self.map(self.sigmoid, layer_outputs)
            elif self.layers_activation[i] == 'hyperbolic_tangent':
                self.map(self.hyperbolic_tangent, layer_outputs)
            elif self.layers_activation[i] == 'cosine':
                self.map(self.cosine, layer_outputs)
            elif self.layers_activation[i] == 'gaussian':
                self.map(self.gaussian, layer_outputs)
            # This layer's outputs will be the inputs for the next layer
            layer_inputs = layer_outputs
        return layer_outputs

    @staticmethod
    def map(func, inputs):
        for i in range(inputs.size):
            for j in range(inputs[i].size):
                inputs[i][j] = func(inputs[i][j])

    @staticmethod
    def null(x):
        return 0

    @staticmethod
    def derivative_null(x):
        return 0

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def derivative_sigmoid(x):
        return x * (1 - x)

    @staticmethod
    def hyperbolic_tangent(x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    @staticmethod
    def derivative_hyperbolic_tangent(x):
        return 1 - (x**2)

    @staticmethod
    def cosine(x):
        return np.cos(x)

    @staticmethod
    def derivative_cosine(x):
        return -np.sin(x)

    @staticmethod
    def gaussian(x):
        return np.exp(-(x**2))

    @staticmethod
    def derivative_gaussian(x):
        return -2 * x * np.exp(-(x**2))
