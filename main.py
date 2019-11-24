from particle_swarm_optimization import *
from neural_network import *
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from mpl_toolkits.mplot3d import Axes3D


def draw_graph(j):

    global step, nn, pso, dim, aux, minError, alpha1, alpha2
    time_step.append([step])

    if step % 50 == 0:
        print(f'Step: {step} - Minimum error found so far: {minError}')

    # Inertia decreases linearly between alpha1 and alpha2
    inertiaWeight = (alpha1 - alpha2) * ((max_steps - step) / max_steps) + alpha2

    # Randomize data set and choose a group of 30 pairs for training.
    # This helps to discover new configurations for the NN
    np.random.shuffle(aux)
    X = np.array([np.array(elem[:(dim-1)]) for elem in aux[:30]])
    y = np.array([elem[dim-1] for elem in aux[:30]])

    # Calculate fitness of each particle
    for particle in pso.population:
        particle.calculate_fitness(nn, X, y)

    # Choose the particle with the best fitness value of all as gBest for each particle
    if step < max_steps/3:
        pso.update_neighborhood_best_random()  # Exploration
    else:
        pso.update_neighborhood_best_local()  # Exploitation

    # Update velocity and position for each particle
    for particle in pso.population:
        particle.update_velocity(inertiaWeight, pso.importancePBest, pso.importanceGBest)
        particle.update_position()

    globalBest = pso.find_global_best()

    bestError = globalBest.pBestFitness
    errorHistorical.append(bestError)
    minError = bestError if bestError < minError else minError

    nn.set_parameters(globalBest.pBest)
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
    error.plot(time_step, errorHistorical, color='red', linewidth=.7)

    if step == max_steps:
        animation.event_source.stop()
        # plt.savefig(str(step) + '.png')

    step += 1


# ------------------------------ ALG. INITIALIZATION ------------------------------ #

# dataFile = 'Data/1in_cubic.txt'
dataFile = 'Data/1in_linear.txt'
# dataFile = 'Data/1in_sine.txt'
# dataFile = 'Data/1in_tanh.txt'
# dataFile = 'Data/2in_complex.txt'
# dataFile = 'Data/2in_xor.txt'

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
nn.add(3)
nn.add(1)

pso = PSO(nn)
pso.generate_initial_population()

# -------------------------------- GENERATE FIGURE -------------------------------- #

fig = plt.figure()
function = fig.add_subplot(1, 2, 1, projection='3d') if dim == 3 else fig.add_subplot(1, 2, 1)
error = fig.add_subplot(1, 2, 2)

step = 0
max_steps = 750
minError = 1.
time_step = []
errorHistorical = []

# Maximum and minimum values for the inertia (decreases linearly)
alpha1 = .9
alpha2 = .4

# The PSO algorithm is run inside the draw_graph function which plots two graphs.
# One with the desired output function and the best solution ever found by the swarm, and another showing the evolution
# of the mean square error (MSE).
animation = ani.FuncAnimation(fig, draw_graph, interval=2)

plt.xticks([])
plt.yticks([])
plt.show()
