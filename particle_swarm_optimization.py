from nn_particle import *
import numpy as np


class PSO:

    populationSize = 20
    neighborhood = 3
    # Sum should be equal 4 and not critical for PSO’s convergence and alleviation of local minimum
    importancePBest = 1.5  # Self confidence factor
    importanceGBest = 2.5  # Swarm confidence factor

    def __init__(self, nn):
        self.particleSize = 0

        for matrix in nn.layers_weights:
            self.particleSize += (matrix.shape[0] * (matrix.shape[1] + 1))

        self.population = np.array([], dtype=object)  # Initialize empty population

    def generate_initial_population(self):
        # Initialize particles randomly
        for _ in range(self.populationSize):
            particle = NN_Solution(self.particleSize)
            self.population = np.append(self.population, [particle], axis=0)

    def stop(self):
        # TODO: Compare all gBest and consider whether the solution is good enough or not
        return False
