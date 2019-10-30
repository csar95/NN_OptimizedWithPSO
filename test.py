from particle_swarm_optimization import *
from neural_network import *
import matplotlib.pyplot as plt
import matplotlib.animation as ani


def draw_graph(j):

    global step, nn, pso, aux, min_error, alfa1, alfa2

    time_step.append([step])

    if step % 50 == 0:
        print(f'Step: {step} - Minimum error found so far: {min_error}')

    if step > 0:

        inertia_weight = (alfa1 - alfa2) * ((max_steps - step)/max_steps) + alfa2

        # TODO: Check whether to keep this or not
        np.random.shuffle(aux)
        X = [elem[0] for elem in aux[:30]]
        y = [elem[1] for elem in aux[:30]]

        # Calculate fitness of each particle
        for particle in pso.population:
            particle.calculate_fitness(nn, X, y)

        # Choose the particle with the best fitness value of all as gBest for each particle
        pso.update_neighborhood_best()

        # Update velocity and position for each particle
        for particle in pso.population:
            particle.update_velocity(inertia_weight, pso.importancePBest, pso.importanceGBest)
            particle.update_position()

    global_best = pso.find_global_best()

    best_error = global_best.pBestFitness
    error_historical.append(best_error)
    min_error = best_error if best_error < min_error else min_error

    nn.set_parameters(global_best.pBest)
    y_preds = np.array([])
    for x in X_train:
        y_pred = nn.feed_forward(np.array(x))[0]
        y_preds = np.append(y_preds, y_pred)

    function.cla()
    function.title.set_text('Function approximation')
    function.set_xlim([-1, 1])
    function.set_ylim([-1, 1])
    function.plot(X_train, y_train, color='blue', label='Desired output')
    function.plot(X_train, y_preds, color='green', label='ANN output')
    function.legend(loc='upper left')
    function.grid(True)

    error.cla()
    error.title.set_text('Mean squared error')
    error.set_xlabel('Num. of iterations')
    error.set_ylabel('Error')
    error.set_ylim([0, .225])
    error.plot(time_step, error_historical, color='red', linewidth=.7)

    if step == max_steps:
        animation.event_source.stop()
    # plt.savefig(str(step) + '.png')

    step += 1


##########################################################################

dataFile = 'Data/1in_linear.txt'
# dataFile = 'Data/1in_sine.txt'
# dataFile = 'Data/2in_complex.txt'

file = open(dataFile, 'rt')

X_train, y_train = np.array([]), np.array([])

for line in file:
    coordinates = line.split()
    X_train = np.append(X_train, [float(coordinates[0])], axis=0)
    y_train = np.append(y_train, [float(coordinates[1])], axis=0)

aux = np.column_stack((X_train, y_train))

nn = NeuralNetwork()
nn.add(4, input_shape=1)
nn.add(3)
nn.add(1)

pso = PSO(nn)

pso.generate_initial_population()

############################ GENERATE FIGURE ############################

fig = plt.figure()
function = fig.add_subplot(1, 2, 1)
error = fig.add_subplot(1, 2, 2)

step = 0
max_steps = 750
min_error = 1.
time_step = []
error_historical = []

alfa1 = .9
alfa2 = .4

animation = ani.FuncAnimation(fig, draw_graph, interval=2)

plt.xticks([])
plt.yticks([])
plt.show()
