import numpy as np
import math
import random
from neural_network import *


class NN_Solution:

    vMax_ActFunc = 2  # Maximum velocity for weights components
    vMax_WB = 4  # Maximum velocity for activation functions components

    def __init__(self, numWeightsBiases, numNeurons):
        self.numNeurons = numNeurons  # Number of neurons in the NN
        self.length = numWeightsBiases + numNeurons  # Particle's position length
        self.fitness = .0
        self.position = np.random.uniform(-1, 1, size=numWeightsBiases)

        # Add the activation func. component correspondent to each neuron
        for _ in range(numNeurons):
            self.position = np.append(self.position, [float(np.random.randint(1, 6))])

        # As particles gain velocity throughout a run, they tend to leave the search space, initializing the velocities
        # to 0 lowers the chances of this happening
        self.velocity = np.zeros(shape=self.length, dtype=float)

        self.pBest = np.zeros(shape=self.length, dtype=float)  # Personal best position
        self.pBestFitness = 999999999.9
        self.gBest = np.zeros(shape=self.length, dtype=float)  # Neighborhood best position
        self.gBestFitness = 999999999.9

    # Updates particle's fitness value based on Neural Network feed-forward alg. and the mean square error (MSE)
    def calculate_fitness(self, nn, x_train, y_train):
        nn.set_parameters(self.position)

        # Calculate MSE error which will determine the fitness of the particle.
        mse = .0
        for i, x in enumerate(x_train):
            y_pred = nn.feed_forward(x)[0]
            mse += ((y_train[i] - y_pred[0]) ** 2)

        mse /= len(x_train)
        self.fitness = mse

        # If the new fitness value is better than its personal best set current value as the new pBest
        if self.fitness < self.pBestFitness:
            self.pBest = self.position
            self.pBestFitness = self.fitness

    # Update particle's velocity based on the cognitive (pBest) and social (gBest) components
    def update_velocity(self, alpha, beta, gamma):
        b = random.uniform(0, beta)
        c = random.uniform(0, gamma)
        # b * (pBest - position)
        personalInfluence = np.multiply(b, np.subtract(self.pBest, self.position))
        # c * (gBest - position)
        neighborhoodInfluence = np.multiply(c, np.subtract(self.gBest, self.position))

        velVariation = np.add(personalInfluence, neighborhoodInfluence)

        # Update velocity taking into account the inertia (alpha)
        self.velocity = np.add(np.multiply(alpha, self.velocity), velVariation)

        # Limit velocity if it exceeds vMax (positive or negative)
        for i in range(len(self.velocity)):
            # Weights and biases components
            if i < self.length - self.numNeurons:
                if self.velocity[i] > self.vMax_WB:
                    self.velocity[i] = self.vMax_WB
                elif self.velocity[i] < -self.vMax_WB:
                    self.velocity[i] = -self.vMax_WB
            # Activation functions components
            else:
                if self.velocity[i] > self.vMax_ActFunc:
                    self.velocity[i] = self.vMax_ActFunc
                elif self.velocity[i] < -self.vMax_ActFunc:
                    self.velocity[i] = -self.vMax_ActFunc

    # Update particle's position by adding the velocity vector to the current position
    def update_position(self):
        self.position = np.add(self.position, self.velocity)
        for i in range(len(self.position)):
            # Absorbing walls for the activation functions components. The value needs to be in range [1, 6)
            if i >= self.length - self.numNeurons:
                if self.position[i] >= 6:
                    self.position[i] = 5. - (self.position[i] - 5.)
                    self.velocity[i] = -self.velocity[i]
                elif self.position[i] < 1:
                    self.position[i] = 1. + (1. - self.position[i])
                    self.velocity[i] = -self.velocity[i]

    # Calculates the euclidean distance between this particle (self) and another passed as a parameter
    # Used to find the closest neighbours of a particle
    def euclidean_distance(self, particle):
        distance = .0
        for i in range(self.position.size):
            distance += (self.position[i] - particle.position[i])**2
        return math.sqrt(distance)
