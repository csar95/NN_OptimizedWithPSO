from neural_network import *

nn = NeuralNetwork()
nn.add(4, input_shape=2)
nn.add(1)
# print(nn.layers_biases)
print(nn.layers_activation)
# pso_array = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
# pso_activation = ['null', 'sigmoid', 'hyperbolic_tangent', 'cosine', 'gaussian']
# nn.set_parameters(pso_array, pso_activation)
# print(nn.layers_biases)
# print(nn.layers_activation)
nn.feed_forward([1, 2])
