from particle_swarm_optimization import *
from neural_network import *


dataFile = 'Data/1in_linear.txt'
# dataFile = 'Data/1in_sine.txt'

file = open(dataFile, 'rt')

X_train, y_train = np.array([]), np.array([])

for line in file:
    coordinates = line.split()
    X_train = np.append(X_train, [float(coordinates[0])], axis=0)
    y_train = np.append(y_train, [float(coordinates[1])], axis=0)


nn = NeuralNetwork()
nn.add(4, input_shape=2, activation='sigmoid')
nn.add(1, activation='gaussian')

layerStructure = [2, 4, 1]
pso = PSO(layerStructure)

pso.generate_initial_population()

iter = 0

while iter < 500 and not pso.stop():
    # Calculate fitness of each particle
    for particle in pso.population:
        particle.calculate_fitness(nn, X_train)

    # Choose the particle with the best fitness value of all as gBest for each particle
    for particle in pso.population:
        particle.choose_neighborhood_best()

    # Update velocity and position for each particle
    for particle in pso.population:
        particle.update_velocity(pso.importancePBest, pso.importanceGBest)
        particle.update_position()

    iter += 1
