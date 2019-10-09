from nn_particle import *
import numpy as np


class PSO:

    populationSize = 40
    neighborhood = 5
    # Sum should be equal 4 and not critical for PSO’s convergence and alleviation of local minimum
    importancePBest = 3  # Self confidence factor
    importanceGBest = 1  # Swarm confidence factor

    def __init__(self, nn):
        self.numWeightsBiases = 0
        self.numNeurons = 0

        for matrix in nn.layers_weights:
            self.numWeightsBiases += (matrix.shape[0] * (matrix.shape[1] + 1))
            self.numNeurons += matrix.shape[0]

        self.population = np.array([], dtype=object)  # Initialize empty population

    def generate_initial_population(self):
        # Initialize particles randomly
        for _ in range(self.populationSize):
            particle = NN_Solution(self.numWeightsBiases, self.numNeurons)
            self.population = np.append(self.population, [particle], axis=0)

    def update_neighborhood_best(self):
        for particle_idx in range(self.populationSize):
            # Choose neighbors based on euclidean distance
            neighborsIndeces = [(index, 999999999.9) for index in range(self.neighborhood)]

            # Calculate distance from one particle to the others
            for j in range(self.populationSize):
                if particle_idx != j:
                    distance = self.population[particle_idx].euclidean_distance(self.population[j])
                    for n in range(len(neighborsIndeces)):

                        # If distance is less than any of the particles in neighborsIndeces replace index and continue
                        if distance < neighborsIndeces[n][1]:
                            neighborsIndeces.insert(n, (j, distance))  # FILO
                            neighborsIndeces.pop()
                            break

            # Among those neighbor particles update our particle's gBest with the pBest of the fittest
            fittestNeighbor = (-1, 999999999.9)  # (index, pbest)
            for neighbor in neighborsIndeces:
                if self.population[neighbor[0]].pBestFitness < fittestNeighbor[1]:
                    fittestNeighbor = (neighbor[0], self.population[neighbor[0]].pBestFitness)
            self.population[particle_idx].gBest = self.population[fittestNeighbor[0]].pBest
            self.population[particle_idx].gBestFitness = self.population[fittestNeighbor[0]].pBestFitness

    def find_global_best(self):
        gBest_idx = -1
        gBest_fitness = 999999999.9
        for particle_idx in range(self.populationSize):
            if self.population[particle_idx].pBestFitness < gBest_fitness:
                gBest_idx = particle_idx
                gBest_fitness = self.population[particle_idx].pBestFitness
        print(gBest_fitness)
        return self.population[gBest_idx].pBest

    def stop(self):
        # TODO: Compare all gBest and consider whether the solution is good enough or not
        return False
