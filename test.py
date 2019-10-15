from particle_swarm_optimization import *
from neural_network import *
import matplotlib.pyplot as plt
import matplotlib.animation as ani


def draw_graph(j):

    global step, nn, pso

    if step % 25 == 0:
        print(step)

    if step > 0:
        # Calculate fitness of each particle
        for particle in pso.population:
            particle.calculate_fitness(nn, X_train, y_train)

        # Choose the particle with the best fitness value of all as gBest for each particle
        pso.update_neighborhood_best()

        # Update velocity and position for each particle
        for particle in pso.population:
            particle.update_velocity(pso.importancePBest, pso.importanceGBest)
            particle.update_position()

    global_best = pso.find_global_best()
    nn.set_parameters(global_best)
    y_preds = np.array([])
    for x in X_train:
        y_pred = nn.feed_forward(np.array(x))[0]
        y_preds = np.append(y_preds, y_pred)

    training_data.cla()
    training_data.plot(X_train, y_train, color='blue')
    predicted_data.plot(X_train, y_preds, color='green')

    if step == 500:
        animation.event_source.stop()
    # plt.savefig(str(step) + '.png')

    step += 1


# dataFile = 'Data/1in_linear.txt'
dataFile = 'Data/1in_sine.txt'
# dataFile = 'Data/2in_complex.txt'

file = open(dataFile, 'rt')

X_train, y_train = np.array([]), np.array([])

for line in file:
    coordinates = line.split()
    X_train = np.append(X_train, [float(coordinates[0])], axis=0)
    y_train = np.append(y_train, [float(coordinates[1])], axis=0)

nn = NeuralNetwork()
nn.add(4, input_shape=1)
nn.add(3)
nn.add(1)

pso = PSO(nn)

pso.generate_initial_population()

# step = 0
#
# while step < 500 and not pso.stop():
#     # Calculate fitness of each particle
#     for particle in pso.population:
#         particle.calculate_fitness(nn, X_train, y_train)
#
#     # Choose the particle with the best fitness value of all as gBest for each particle
#     pso.update_neighborhood_best()
#
#     # Update velocity and position for each particle
#     for particle in pso.population:
#         particle.update_velocity(pso.importancePBest, pso.importanceGBest)
#         particle.update_position()
#
#     step += 1

####### Generate figure

fig = plt.figure()
training_data = fig.add_subplot(1, 1, 1)
predicted_data = fig.add_subplot(1, 1, 1)

step = 0

animation = ani.FuncAnimation(fig, draw_graph, interval=2)

plt.xticks([])
plt.yticks([])
plt.show()