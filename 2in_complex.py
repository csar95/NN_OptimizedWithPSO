from particle_swarm_optimization import *
from neural_network import *
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from mpl_toolkits.mplot3d import Axes3D


def draw_graph(j):

    global step, nn, pso, dim, aux, min_error, alfa1, alfa2

    time_step.append([step])

    if step % 50 == 0:
        print(f'Step: {step} - Minimum error found so far: {min_error}')

    inertia_weight = (alfa1 - alfa2) * ((max_steps - step)/max_steps) + alfa2

    # Testing how good the parameters are with part of the data after randomizing it
    # This helps to discover new configurations for the NN
    np.random.shuffle(aux)
    X = np.array([np.array(elem[:(dim-1)]) for elem in aux[:30]])
    y = np.array([elem[dim-1] for elem in aux[:30]])

    # Calculate fitness of each particle
    for particle in pso.population:
        particle.calculate_fitness(nn, X, y)

    # Choose the particle with the best fitness value of all as gBest for each particle
    # (Reference slides or idea taken from https://www.sciencedirect.com/science/article/pii/S0020025517306485)
    if step < max_steps/3:
        pso.update_neighborhood_best_random()  # Exploration
    else:
        pso.update_neighborhood_best_local()  # Exploitation

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
    if dim == 2:
        function.set_xlim([-1, 1])
        function.set_ylim([-1, 1])
        function.plot(X_train, y_train, color='blue', label='Desired output')
        function.plot(X_train, y_preds, color='green', label='ANN output')
    else:
        X = [elem[0] for elem in X_train]
        Y = [elem[1] for elem in X_train]
        function.plot(X, Y, y_train, color='blue', label='Desired output')
        function.plot(X, Y, y_preds, color='green', label='ANN output')
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


########################## ALG. INITIALIZATION ##########################

dataFile = 'Data/2in_complex.txt'

file = open(dataFile, 'rt')

dim = len(file.readline().split())
file.seek(0)

X_train, y_train = np.empty((0, (dim-1)), dtype=float), np.array([])

for line in file:
    coordinates = line.split()
    X_train = np.append(X_train, [np.array(coordinates[:(dim-1)], dtype=float)], axis=0)
    y_train = np.append(y_train, [float(coordinates[dim-1])])

aux = np.column_stack((X_train, y_train))

nn = NeuralNetwork()
nn.add(4, input_shape=(dim-1))
for _ in range(20):
    nn.add(3)
nn.add(1)

pso = PSO(nn, populationSize=40, importancePBest=3)

pso.generate_initial_population()

############################ GENERATE FIGURE ############################

fig = plt.figure()
function = fig.add_subplot(1, 2, 1, projection='3d') if dim == 3 else fig.add_subplot(1, 2, 1)
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

