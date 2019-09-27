import numpy as np
import math


class NeuralNetwork:

    learningRate = .1

    def __init__(self, layers):
        self.layersSize = np.array(layers)
        self.weightMatrices = []
        self.biases = []

        for i in range(1, self.layersSize.size):
            weights = np.random.uniform(-1, 1, size=(self.layersSize[i], self.layersSize[i-1]))
            self.weightMatrices.append(weights)
            bias = np.random.uniform(-1, 1, size=(self.layersSize[i], 1))
            self.biases.append(bias)

    def feedforward(self, inputs):
        layerInputs = np.reshape(inputs, newshape=(-1, 1))

        for i in range(1, self.layersSize.size):
            # Calculate layer outputs
            layerOutputs = np.dot(self.weightMatrices[i-1], layerInputs)
            layerOutputs = np.add(layerOutputs, self.biases[i-1])
            # Activation function
            self.__map_layer(layerOutputs, self.layersSize[i])
            layerInputs = layerOutputs  # This layer's outputs will be the inputs for the next layer

        return layerOutputs

    def feedforward_for_training(self, inputs):
        layersInputs = []
        layerInputs = np.reshape(inputs, newshape=(-1, 1))
        layersInputs.append(layerInputs)

        for i in range(1, self.layersSize.size):
            # Calculate layer outputs
            layerOutputs = np.dot(self.weightMatrices[i-1], layersInputs[-1])
            layerOutputs = np.add(layerOutputs, self.biases[i-1])
            # Activation function
            self.__map_layer(layerOutputs, self.layersSize[i])
            layersInputs.append(layerOutputs)  # This layer's outputs will be the inputs for the next layer

        return layersInputs

    def train(self, inputs, targets):
        layersOutputs = self.feedforward_for_training(inputs)
        errors = np.array([])

        for i in range(1, len(layersOutputs)):
            if i == 1:
                # Output errors
                targets = np.reshape(targets, newshape=(-1, 1))
                errors = np.subtract(targets, layersOutputs[-i])
            else:
                # Errors of a hidden layer
                w = np.transpose(self.weightMatrices[-(i-1)])
                errors = np.dot(w, errors)

            ########### Calculate gradient: Layer(i) -> Layer(i+1) ###########
            self.__map_layer_derivative(layersOutputs[-i], self.layersSize[-i])
            gradients = np.dot(self.learningRate,
                               np.multiply(layersOutputs[-i], errors))  # np.multiply -> Hadamard product

            ########### Update weights: Layer(i) -> Layer(i+1) ###########
            if i == len(layersOutputs) - 1:
                layer_T = np.reshape(inputs, newshape=(1, -1))
            else:
                layer_T = np.transpose(layersOutputs[-(i+1)])

            w_variation = np.dot(gradients, layer_T)
            self.weightMatrices[-i] = np.add(self.weightMatrices[-i], w_variation)

            ########### Adjust bias. The variation is just the gradients ###########
            self.biases[-i] = np.add(self.biases[-i], gradients)

    def __map_layer(self, layer, size):
        for i in range(size):
            layer[i][0] = self.sigmoid(layer[i][0])

    def __map_layer_derivative(self, layer, size):
        for i in range(size):
            layer[i][0] = self.derivative_sigmoid(layer[i][0])

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    @staticmethod
    def derivative_sigmoid(y):
        return y * (1 - y)
