from particle_swarm_optimization import *
from neural_network import *
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from mpl_toolkits.mplot3d import Axes3D
import time

dataset = ['Data/1in_cubic.txt', 'Data/1in_linear.txt', 'Data/1in_sine.txt', 'Data/1in_tanh.txt', 'Data/2in_complex.txt', 'Data/2in_xor.txt']
populationSize = [10, 20, 30, 40, 50, 60]
neighborhood = [2, 3, 4, 5, 6, 7]
layers = [1, 2, 3, 4]


for data in dataset:
    dataFile = data




########################## ALG. INITIALIZATION ##########################

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

# TODO: try from 1 to 5
nn = NeuralNetwork()
nn.add(4, input_shape=(dim-1))
nn.add(3)
nn.add(1)

pso = PSO(nn)

pso.generate_initial_population()

############################ GENERATE FIGURE ############################

step = 0
max_steps = 750
min_error = 1.
time_step = []
error_historical = []

alfa1 = .9
alfa2 = .4

time1 = .0

time_step.append([step])

if step % 50 == 0:
    print(f'Step: {step} - Minimum error found so far: {min_error}')

inertia_weight = (alfa1 - alfa2) * ((max_steps - step) / max_steps) + alfa2

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
error_historical.append(best_error)
min_error = best_error if best_error < min_error else min_error

nn.set_parameters(global_best.pBest)
y_preds = np.array([])
for x in X_train:
    y_pred = nn.feed_forward(np.array(x))[0]
    y_preds = np.append(y_preds, y_pred)

if step == 0:
    time1 = time.time()

if step == max_steps:
    time2 = time.time()
    print(time2 - time1)
# plt.savefig(str(step) + '.png')

step += 1