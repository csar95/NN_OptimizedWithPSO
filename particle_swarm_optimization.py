from nn_particle import *
import numpy as np


class PSO:

    def __init__(self, nn, populationSize=30, neighborhood=4, importancePBest=2.5):
        self.populationSize = populationSize
        self.neighborhood = neighborhood  # Number of informants of a particle

        # Sum the cognitive (pBest) and social (gBest) components should be equal 4
        self.importancePBest = importancePBest  # Self confidence factor
        self.importanceGBest = 4 - self.importancePBest  # Neighbourhood confidence factor

        self.numWeightsBiases = 0  # Number of weights components represented in a particle's position
        self.numNeurons = 0  # Number of neurons in the NN

        for matrix in nn.layers_weights:
            self.numWeightsBiases += (matrix.shape[0] * (matrix.shape[1] + 1))
            self.numNeurons += matrix.shape[0]

        self.population = np.array([], dtype=object)  # Initialize empty population (swarm)

    # Initialize particles in the swarm at random
    def generate_initial_population(self):
        for _ in range(self.populationSize):
            particle = NN_Solution(self.numWeightsBiases, self.numNeurons)
            self.population = np.append(self.population, [particle], axis=0)

    # For each particle in the swarm find its informants based on a random selection.
    # Then, calculates the best position ever found by those informants (gBest)
    def update_neighborhood_best_random(self):
        for particleIdx in range(self.populationSize):
            
            particleIndexes = [i for i in range(self.populationSize)]
            particleIndexes.remove(particleIdx)
            
            # Select the index of particles in the swarm at random. These'll be the informants of particle 'particleIdx'
            neighboursIndexes = [(particleIndexes.pop(random.randint(0, len(particleIndexes)-1)), float('inf'))
                                for _ in range(self.neighborhood)]

            # Among those neighbour particles update the particle's gBest with the pBest of the fittest neighbour
            fittestNeighbor = (-1, float('inf'))  # Format: (neighbourIndex, pBestFitness)
            for neighbor in neighboursIndexes:
                if self.population[neighbor[0]].pBestFitness < fittestNeighbor[1]:
                    fittestNeighbor = (neighbor[0], self.population[neighbor[0]].pBestFitness)
            
            self.population[particleIdx].gBest = self.population[fittestNeighbor[0]].pBest
            self.population[particleIdx].gBestFitness = self.population[fittestNeighbor[0]].pBestFitness

    # For each particle in the swarm find its informants based on the euclidean distance (Local neighborhood).
    # Then, calculates the best position ever found by those informants (gBest)
    def update_neighborhood_best_local(self):
        for particleIdx in range(self.populationSize):

            neighboursIndexes = [(index, float('inf')) for index in range(self.neighborhood)]

            # Calculate euclidean distance from one particle to the others
            for j in range(self.populationSize):
                if particleIdx != j:
                    distance = self.population[particleIdx].euclidean_distance(self.population[j])
                    for n in range(len(neighboursIndexes)):

                        # If distance is less than any of the particles in neighboursIndexes replace index and continue
                        if distance < neighboursIndexes[n][1]:
                            neighboursIndexes.insert(n, (j, distance))  # FILO
                            neighboursIndexes.pop()
                            break

            # Among those neighbour particles update the particle's gBest with the pBest of the fittest neighbour
            fittestNeighbor = (-1, float('inf'))  # Format: (neighbourIndex, pBestFitness)
            for neighbor in neighboursIndexes:
                if self.population[neighbor[0]].pBestFitness < fittestNeighbor[1]:
                    fittestNeighbor = (neighbor[0], self.population[neighbor[0]].pBestFitness)

            self.population[particleIdx].gBest = self.population[fittestNeighbor[0]].pBest
            self.population[particleIdx].gBestFitness = self.population[fittestNeighbor[0]].pBestFitness

    # Find the position found by all the particles in the swarm. Used for plotting the graph
    def find_global_best(self):
        gBestIdx = -1
        gBestFitness = float('inf')

        for particleIdx in range(self.populationSize):
            if self.population[particleIdx].pBestFitness < gBestFitness:
                gBestIdx = particleIdx
                gBestFitness = self.population[particleIdx].pBestFitness

        return self.population[gBestIdx]
