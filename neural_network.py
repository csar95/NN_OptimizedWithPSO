import numpy as np


class NeuralNetwork:

    # activation_functions = {1: 'null', 2: 'sigmoid', 3: 'hyperbolic_tangent', 4: 'cosine', 5: 'gaussian'}

    # This arrays will contain the arrays for each layer, thus they represents matrices
    def __init__(self):
        self.layers_weights = []
        self.layers_biases = []
        self.layers_activation = []

    # Weights and biases are initialized random in range (-1, 1), activation functions are integers in range (1, 5)
    def add(self, neurons, input_shape=None):
        # If this is the first layer added
        if not self.layers_weights:
            weights = np.random.uniform(-1, 1, size=(neurons, input_shape))
        # For every other layer
        else:
            #                                                 Number of rows in the previous layer
            weights = np.random.uniform(-1, 1, size=(neurons, self.layers_weights[-1][:, 0].size))
        bias = np.random.uniform(-1, 1, size=(neurons, 1))
        activation = np.random.randint(1, 6, size=(neurons, 1))
        self.layers_weights.append(weights)
        self.layers_biases.append(bias)
        self.layers_activation.append(activation)

    # It transposes an array of values into the parameters of the neural network
    def set_parameters(self, pso_array):
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
        for layer in range(len(self.layers_activation)):
            for activations in range(len(self.layers_activation[layer])):
                for activation in range(len(self.layers_activation[layer][activations])):
                    self.layers_activation[layer][activations][activation] = int(temp[0])
                    temp.pop(0)

    def feed_forward(self, inputs):
        layer_inputs = np.reshape(inputs, newshape=(-1, 1))
        for i in range(len(self.layers_weights)):
            # Calculating the layer outputs
            layer_outputs = np.dot(self.layers_weights[i], layer_inputs)
            layer_outputs = np.add(layer_outputs, self.layers_biases[i])
            # Applying an activation function
            self.map(self.layers_activation[i], layer_outputs)
            # This layer's outputs will be the inputs for the next layer
            layer_inputs = layer_outputs
        return layer_outputs

    # This method maps activation functions numbers into the actual functions
    def map(self, activations, inputs):
        for i in range(len(activations)):
            if activations[i] == 1:
                inputs[i] = self.null(inputs[i])
            elif activations[i] == 2:
                inputs[i] = self.sigmoid(inputs[i])
            elif activations[i] == 3:
                inputs[i] = self.hyperbolic_tangent(inputs[i])
            elif activations[i] == 4:
                inputs[i] = self.cosine(inputs[i])
            elif activations[i] == 5:
                inputs[i] = self.gaussian(inputs[i])
            else:
                print('ERROR: Activation function not in range from 1 to 5!')

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
        return np.exp(-(x**2)/2)
