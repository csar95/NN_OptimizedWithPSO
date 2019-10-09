import numpy as np
import math
import random
from neural_network import *


class NN_Solution:

    alfa = 1
    vMax_ActFunc = 2
    vMax_WB = 4
    # TODO: Inertia factor for velocity

    def __init__(self, numWeightsBiases, numNeurons):
        self.length = numWeightsBiases + numNeurons
        self.fitness = .0
        self.position = np.random.uniform(-1, 1, size=numWeightsBiases)
        # Add the activation func. component correspondent to each neuron
        for _ in range(numNeurons):
            self.position = np.append(self.position, [float(random.randint(1, 5))])

        # As particles gain velocity throughout a run, they tend to leave the search space, initializing the velocities
        # to 0 lowers the chances of this happening,
        self.velocity = np.zeros(shape=self.length, dtype=float)

        # One mechanism for preserving solution feasibility is one in which particles going outside the search space are
        # not allowed to improve their pBest position so that they are attracted back to the feasible space in
        # subsequent iterations.
        self.pBest = np.zeros(shape=self.length, dtype=float)  # Personal best position
        self.pBestFitness = 999999999.9
        self.gBest = np.zeros(shape=self.length, dtype=float)  # Neighborhood best position
        self.gBestFitness = 999999999.9

    def calculate_fitness(self, nn, x_train, y_train):
        # Updates fitness value based on Neural Network feed-forward alg. and cost error
        nn.set_weights(self.position)

        mse = .0
        for i, x in enumerate(x_train):
            y_pred = nn.feed_forward(x)[0]
            mse += ((y_train[i] - y_pred[0]) ** 2)

        mse /= len(x_train)
        self.fitness = mse

        # If the fitness value is better than its personal best set current value as the new pBest
        if self.fitness < self.pBestFitness:
            self.pBest = self.position
            self.pBestFitness = self.fitness

    def update_velocity(self, c1, c2):
        # TODO: Add global best
        # c1 * u1 * (pBest - position)
        personalInfluence = np.multiply(c1, np.multiply(np.random.uniform(0, self.alfa, size=self.length),
                                                        np.subtract(self.pBest, self.position)))
        # c2 * u2 * (gBest - position)
        neighborhoodInfluence = np.multiply(c2, np.multiply(np.random.uniform(0, self.alfa, size=self.length),
                                                            np.subtract(self.gBest, self.position)))

        velVariation = np.add(personalInfluence, neighborhoodInfluence)

        # TODO: Consider inertia
        self.velocity = np.add(self.velocity, velVariation)
        # TODO: Consider max velocity. If exceeded, limit to velMax

    # TODO: See https://www.researchgate.net/publication/224613555_A_Hybrid_Boundary_Condition_for_Robust_Particle_Swarm_Optimization for solution outside the search space
    def update_position(self):
        self.position = np.add(self.position, self.velocity)

    def euclidean_distance(self, particle):
        distance = .0
        for i in range(self.position.size):
            distance += (self.position[i] - particle.position[i])**2
        return math.sqrt(distance)
