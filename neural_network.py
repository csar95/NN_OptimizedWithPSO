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

    def set_parameters(self, pso_array, pso_activation=None):
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
        if pso_activation:
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
            self.map(self.layers_activation[i], layer_outputs)
            # This layer's outputs will be the inputs for the next layer
            layer_inputs = layer_outputs
        return layer_outputs

    def map(self, activations, inputs):
        for i in range(len(activations)):
            if activations[i] == 'null':
                inputs[i] = self.null(inputs[i])
            elif activations[i] == 'sigmoid':
                inputs[i] = self.sigmoid(inputs[i])
            elif activations[i] == 'hyperbolic_tangent':
                inputs[i] = self.hyperbolic_tangent(inputs[i])
            elif activations[i] == 'cosine':
                inputs[i] = self.cosine(inputs[i])
            elif activations[i] == 'gaussian':
                inputs[i] = self.gaussian(inputs[i])

    @staticmethod
    def null(x):
        return 0

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def hyperbolic_tangent(x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    @staticmethod
    def cosine(x):
        return np.cos(x)

    @staticmethod
    def gaussian(x):
        return np.exp(-(x**2))
