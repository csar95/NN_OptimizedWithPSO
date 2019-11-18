from nn_particle import *
import numpy as np


class PSO:

    # TODO: try from 10 to 60
    populationSize = 30
    # TODO: try from 2 to 10
    neighborhood = 4
    # Sum should be equal 4 and not critical for PSO’s convergence and alleviation of local minimum
    importancePBest = 2.5  # Self confidence factor
    importanceGBest = 1.5  # Swarm confidence factor

    def __init__(self, nn, populationSize=30, neighborhood=4, importancePBest=2.5):
        self.populationSize = populationSize
        self.neighborhood = neighborhood

        self.importancePBest = importancePBest
        self.importanceGBest = 4 - self.importancePBest

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

    # Based on a random selection of neighbors
    def update_neighborhood_best_random(self):
        for particleIdx in range(self.populationSize):
            # neighborsIndeces = [(index % self.populationSize, float('inf'))
            #                     for index in range(particleIdx + 1, particleIdx + 1 + self.neighborhood)]

            particleIndeces = [i for i in range(self.populationSize)]
            particleIndeces.remove(particleIdx)

            neighborsIndeces = [(particleIndeces.pop(random.randint(0, len(particleIndeces)-1)), float('inf'))
                                for _ in range(self.neighborhood)]

            # Among those neighbor particles update our particle's gBest with the pBest of the fittest
            fittestNeighbor = (-1, float('inf'))  # (index, pbest)
            for neighbor in neighborsIndeces:
                if self.population[neighbor[0]].pBestFitness < fittestNeighbor[1]:
                    fittestNeighbor = (neighbor[0], self.population[neighbor[0]].pBestFitness)
            self.population[particleIdx].gBest = self.population[fittestNeighbor[0]].pBest
            self.population[particleIdx].gBestFitness = self.population[fittestNeighbor[0]].pBestFitness

    # Based on a local neighborhood
    def update_neighborhood_best_local(self):
        for particleIdx in range(self.populationSize):
            # Choose neighbors based on euclidean distance
            neighborsIndeces = [(index, float('inf')) for index in range(self.neighborhood)]

            # Calculate distance from one particle to the others
            for j in range(self.populationSize):
                if particleIdx != j:
                    distance = self.population[particleIdx].euclidean_distance(self.population[j])
                    for n in range(len(neighborsIndeces)):

                        # If distance is less than any of the particles in neighborsIndeces replace index and continue
                        if distance < neighborsIndeces[n][1]:
                            neighborsIndeces.insert(n, (j, distance))  # FILO
                            neighborsIndeces.pop()
                            break

            # Among those neighbor particles update our particle's gBest with the pBest of the fittest
            fittestNeighbor = (-1, float('inf'))  # (index, pbest)
            for neighbor in neighborsIndeces:
                if self.population[neighbor[0]].pBestFitness < fittestNeighbor[1]:
                    fittestNeighbor = (neighbor[0], self.population[neighbor[0]].pBestFitness)
            self.population[particleIdx].gBest = self.population[fittestNeighbor[0]].pBest
            self.population[particleIdx].gBestFitness = self.population[fittestNeighbor[0]].pBestFitness

    def find_global_best(self):
        gBest_idx = -1
        gBest_fitness = float('inf')

        for particle_idx in range(self.populationSize):
            if self.population[particle_idx].pBestFitness < gBest_fitness:
                gBest_idx = particle_idx
                gBest_fitness = self.population[particle_idx].pBestFitness

        return self.population[gBest_idx]
