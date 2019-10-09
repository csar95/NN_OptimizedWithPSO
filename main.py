from neural_network import *
import random


nn = NeuralNetwork()
nn.add(4, input_shape=2, activation='sigmoid')
nn.add(1, activation='gaussian')
# pso_array = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
# print(nn.layers_weights)
# print(nn.layers_biases)
# print(pso_array)
# nn.set_weights(pso_array)
# print(pso_array)
# print(nn.layers_weights)
# print(nn.layers_biases)
nn.feed_forward([1,2])


s = np.random.uniform(-1, 1, size=17)
print(s)
s = np.append(s, [float(random.randint(1, 5))])
print(s)

a = np.ones(shape=18)
print(a)

b = np.subtract(a, s)
print(b)
