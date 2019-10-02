from particle_swarm_optimization import *


layerStructure = [1, 3, 4, 1]
pso = PSO(layerStructure)

pso.generate_initial_population()

iter = 0

while iter < 500 and not pso.stop():
    # Calculate fitness of each particle
    for particle in pso.population:
        particle.calculate_fitness()

    # Choose the particle with the best fitness value of all as gBest for each particle
    for particle in pso.population:
        particle.choose_neighborhood_best()

    # Update velocity and position for each particle
    for particle in pso.population:
        particle.update_velocity(pso.importancePBest, pso.importanceGBest)
        particle.update_position()

    iter += 1
