from nn_solution import *
import numpy as np


class PSO:

    populationSize = 20
    neighborhood = 3
    # Sum should be equal 4 and not critical for PSO’s convergence and alleviation of local minimum
    importancePBest = 1.5  # Self confidence factor
    importanceGBest = 2.5  # Swarm confidence factor

    def __init__(self, layers):
        self.particleSize = 0
        for i in range(1, len(layers)):
            self.particleSize += (layers[i] * layers[i-1])  # (layers[i] * (layers[i-1] + 1)) If we consider biases

        self.population = np.array([], dtype=object)  # Initialize empty population

    def generate_initial_population(self):
        # Initialize particles randomly
        for _ in range(self.populationSize):
            particle = NN_Solution(self.particleSize)
            self.population = np.append(self.population, [particle], axis=0)

    def stop(self):
        # TODO: Compare all gBest and consider whether the solution is good enough or not
        return False
