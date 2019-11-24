from particle_swarm_optimization import *
from neural_network import *

datasets = ['Data/1in_cubic.txt', 'Data/1in_linear.txt', 'Data/1in_sine.txt', 'Data/1in_tanh.txt', 'Data/2in_complex.txt', 'Data/2in_xor.txt']
populationSize = [10, 20, 30, 40, 50, 60, 100]
neighborhood = [2, 3, 4, 5, 6, 7, 10]
layers = [0, 1, 2, 3, 4, 10, 20]
importancePBest = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]

for dataFile in datasets:

    print('############################' + dataFile + '################################')

    popSizeError = []
    neighborhoodError = []
    layersError = []
    pBestError = []

    ########################## ALG. INITIALIZATION ##########################

    file = open(dataFile, 'rt')

    dim = len(file.readline().split())
    file.seek(0)

    X_train, y_train = np.empty((0, (dim-1)), dtype=float), np.array([])

    for line in file:
        coordinates = line.split()
        X_train = np.append(X_train, [np.array(coordinates[:(dim-1)], dtype=float)], axis=0)
        y_train = np.append(y_train, [float(coordinates[dim-1])])

    max_steps = 750
    alpha1 = .9
    alpha2 = .4

    #########################################################################

    print('-------------------------------------------------------------')

    for popSize in populationSize:
        avgError = .0

        for i in range(10):

            aux = np.column_stack((X_train, y_train))

            nn = NeuralNetwork()
            nn.add(4, input_shape=(dim-1))
            nn.add(3)
            nn.add(1)

            pso = PSO(nn, populationSize=popSize)
            pso.generate_initial_population()

            minError = 1.

            for step in range(max_steps):
                inertia_weight = (alpha1 - alpha2) * ((max_steps - step) / max_steps) + alpha2

                # Testing how good the parameters are with part of the data after randomizing it
                # This helps to discover new configurations for the NN
                np.random.shuffle(aux)
                X = np.array([np.array(elem[:(dim - 1)]) for elem in aux[:30]])
                y = np.array([elem[dim - 1] for elem in aux[:30]])

                # Calculate fitness of each particle
                for particle in pso.population:
                    particle.calculate_fitness(nn, X, y)

                # Choose the particle with the best fitness value of all as gBest for each particle
                # (Reference slides or idea taken from https://www.sciencedirect.com/science/article/pii/S0020025517306485)
                if step < max_steps / 3:
                    pso.update_neighborhood_best_random()  # Exploration
                else:
                    pso.update_neighborhood_best_local()  # Exploitation

                # Update velocity and position for each particle
                for particle in pso.population:
                    particle.update_velocity(inertia_weight, pso.importancePBest, pso.importanceGBest)
                    particle.update_position()

                global_best = pso.find_global_best()

                best_error = global_best.pBestFitness
                minError = best_error if best_error < minError else minError

            print(f'Episode: {i} | Minimum error: {minError}')
            avgError += minError

        avgError /= 10
        popSizeError.append(avgError)
        print(f'\nPopulation size: {popSize} | Average error: {avgError}')

    #########################################################################

    print('-------------------------------------------------------------')

    for n in neighborhood:
        avgError = .0

        for i in range(10):

            aux = np.column_stack((X_train, y_train))

            nn = NeuralNetwork()
            nn.add(4, input_shape=(dim-1))
            nn.add(3)
            nn.add(1)

            pso = PSO(nn, neighborhood=n)
            pso.generate_initial_population()

            minError = 1.

            for step in range(max_steps):
                inertia_weight = (alpha1 - alpha2) * ((max_steps - step) / max_steps) + alpha2

                # Testing how good the parameters are with part of the data after randomizing it
                # This helps to discover new configurations for the NN
                np.random.shuffle(aux)
                X = np.array([np.array(elem[:(dim - 1)]) for elem in aux[:30]])
                y = np.array([elem[dim - 1] for elem in aux[:30]])

                # Calculate fitness of each particle
                for particle in pso.population:
                    particle.calculate_fitness(nn, X, y)

                # Choose the particle with the best fitness value of all as gBest for each particle
                # (Reference slides or idea taken from https://www.sciencedirect.com/science/article/pii/S0020025517306485)
                if step < max_steps / 3:
                    pso.update_neighborhood_best_random()  # Exploration
                else:
                    pso.update_neighborhood_best_local()  # Exploitation

                # Update velocity and position for each particle
                for particle in pso.population:
                    particle.update_velocity(inertia_weight, pso.importancePBest, pso.importanceGBest)
                    particle.update_position()

                global_best = pso.find_global_best()

                best_error = global_best.pBestFitness
                minError = best_error if best_error < minError else minError

            print(f'Episode: {i} | Minimum error: {minError}')
            avgError += minError

        avgError /= 10
        neighborhoodError.append(avgError)
        print(f'\nNum. of neighbors: {n} | Average error: {avgError}')

    #########################################################################

    print('-------------------------------------------------------------')

    for nLayers in layers:
        avgError = .0

        for i in range(10):

            aux = np.column_stack((X_train, y_train))

            nn = NeuralNetwork()
            nn.add(4, input_shape=(dim-1))
            for _ in range(nLayers):
                nn.add(3)
            nn.add(1)

            pso = PSO(nn)
            pso.generate_initial_population()

            minError = 1.

            for step in range(max_steps):
                inertia_weight = (alpha1 - alpha2) * ((max_steps - step) / max_steps) + alpha2

                # Testing how good the parameters are with part of the data after randomizing it
                # This helps to discover new configurations for the NN
                np.random.shuffle(aux)
                X = np.array([np.array(elem[:(dim - 1)]) for elem in aux[:30]])
                y = np.array([elem[dim - 1] for elem in aux[:30]])

                # Calculate fitness of each particle
                for particle in pso.population:
                    particle.calculate_fitness(nn, X, y)

                # Choose the particle with the best fitness value of all as gBest for each particle
                # (Reference slides or idea taken from https://www.sciencedirect.com/science/article/pii/S0020025517306485)
                if step < max_steps / 3:
                    pso.update_neighborhood_best_random()  # Exploration
                else:
                    pso.update_neighborhood_best_local()  # Exploitation

                # Update velocity and position for each particle
                for particle in pso.population:
                    particle.update_velocity(inertia_weight, pso.importancePBest, pso.importanceGBest)
                    particle.update_position()

                global_best = pso.find_global_best()

                best_error = global_best.pBestFitness
                minError = best_error if best_error < minError else minError

            print(f'Episode: {i} | Minimum error: {minError}')
            avgError += minError

        avgError /= 10
        layersError.append(avgError)
        print(f'\nNum. of layers: {nLayers} | Average error: {avgError}')

    print('------------------------------------------------------------- ')

    for c1 in importancePBest:
        avgError = .0

        for i in range(10):

            aux = np.column_stack((X_train, y_train))

            nn = NeuralNetwork()
            nn.add(4, input_shape=(dim-1))
            nn.add(3)
            nn.add(1)

            pso = PSO(nn, importancePBest=c1)
            pso.generate_initial_population()

            minError = 1.

            for step in range(max_steps):
                inertia_weight = (alpha1 - alpha2) * ((max_steps - step) / max_steps) + alpha2

                # Testing how good the parameters are with part of the data after randomizing it
                # This helps to discover new configurations for the NN
                np.random.shuffle(aux)
                X = np.array([np.array(elem[:(dim - 1)]) for elem in aux[:30]])
                y = np.array([elem[dim - 1] for elem in aux[:30]])

                # Calculate fitness of each particle
                for particle in pso.population:
                    particle.calculate_fitness(nn, X, y)

                # Choose the particle with the best fitness value of all as gBest for each particle
                # (Reference slides or idea taken from https://www.sciencedirect.com/science/article/pii/S0020025517306485)
                if step < max_steps / 3:
                    pso.update_neighborhood_best_random()  # Exploration
                else:
                    pso.update_neighborhood_best_local()  # Exploitation

                # Update velocity and position for each particle
                for particle in pso.population:
                    particle.update_velocity(inertia_weight, pso.importancePBest, pso.importanceGBest)
                    particle.update_position()

                global_best = pso.find_global_best()

                best_error = global_best.pBestFitness
                minError = best_error if best_error < minError else minError

            print(f'Episode: {i} | Minimum error: {minError}')
            avgError += minError

        avgError /= 10
        pBestError.append(avgError)
        print(f'\nInfluence of pBest: {c1} | Average error: {avgError}')

    print('------------------------------------------------------------- ')
    print(f'Population size error: {popSizeError}')
    print(f'Number of informants error: {neighborhoodError}')
    print(f'Number of layers error: {layersError}')
    print(f'Influence of pBest error: {pBestError}')
