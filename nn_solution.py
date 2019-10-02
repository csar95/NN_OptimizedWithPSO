import numpy as np


class NN_Solution:

    alfa = 1

    def __init__(self, particleSize):
        self.length = particleSize
        self.position = np.random.uniform(-1, 1, size=(1, particleSize))
        self.velocity = np.random.uniform(-1, 1, size=(1, particleSize))  # TODO: To be defined
        # TODO: self.velMax
        self.fitness = .0
        self.pBest = np.random.uniform(-1, 1, size=(1, particleSize))  # Personal best position
        self.pBestFitness = .0
        self.gBest = np.random.uniform(-1, 1, size=(1, particleSize))  # Neighborhood best position
        self.gBestFitness = .0

    def calculate_fitness(self):
        # TODO: Calculate fitness value based on Neural Network feed-forward alg. and cost error
        # TODO: If the fitness value is better than its personal best set current value as the new pBest
        pass

    def update_velocity(self, c1, c2):
        # TODO: Make sure this is the correct formula
        # TODO: Consider max velocity. If exceeded, limit to velMax
        # c1 * u1 * (pBest - position)
        personalInfluence = np.multiply(c1, np.multiply(np.random.uniform(0, self.alfa, size=(1, self.length)),
                                                        np.subtract(self.pBest, self.position)))
        # c2 * u2 * (gBest - position)
        neighborhoodInfluence = np.multiply(c2, np.multiply(np.random.uniform(0, self.alfa, size=(1, self.length)),
                                                            np.subtract(self.gBest, self.position)))

        velVariation = np.add(personalInfluence, neighborhoodInfluence)

        self.velocity = np.add(self.velocity, velVariation)

    def update_position(self):
        self.position = np.add(self.position, self.velocity)

    def choose_neighborhood_best(self):
        # TODO: A particle knows the fitness of those in its neighborhood, and uses the position of the fittest
        pass
