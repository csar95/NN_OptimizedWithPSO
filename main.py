from neural_network import *
import matplotlib.pyplot as plt
import matplotlib.animation as ani


def draw_graph(j):

    global step, nn

    if step % 50 == 0: print(step)

    if step > 0:
        picks = np.arange(len(X_train))
        np.random.shuffle(picks)
        for i in picks:
            nn.train(X_train[i], y_train[i])

    y_preds = np.array([])

    for x in X_train:
        y_pred = nn.feedforward(np.array(x))[0]
        y_preds = np.append(y_preds, y_pred)

    training_data.cla()
    training_data.plot(X_train, y_train, color='blue')
    predicted_data.plot(X_train, y_preds, color='green')

    if step == 500:
        animation.event_source.stop()
    # plt.savefig(str(step) + '.png')

    step += 1


##################################################################

nn = NeuralNetwork()
nn.add(4, input_shape=2, activation='sigmoid')
nn.add(1, activation='gaussian')
pso_array = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
print(nn.layers_weights)
print(nn.layers_biases)
print(pso_array)
nn.set_weights(pso_array)
print(pso_array)
print(nn.layers_weights)
print(nn.layers_biases)



'''
nn = NeuralNetwork([1, 3, 4, 1])

dataFile = 'Data/1in_linear.txt'
# dataFile = '1in_sine.txt'

file = open(dataFile, 'rt')

X_train, y_train = np.array([]), np.array([])

for line in file:
    coordinates = line.split()
    X_train = np.append(X_train, [float(coordinates[0])], axis=0)
    y_train = np.append(y_train, [float(coordinates[1])], axis=0)

# y_preds = np.array([])
#
# for x in X_train:
#     y_pred = nn.feedforward(np.array(x))
#     y_preds = np.append(y_preds, y_pred)
#
# plt.plot(X_train, y_train, color='b')
# plt.plot(X_train, y_preds, color='r')
# plt.show()

####### Generate figure

fig = plt.figure()
training_data = fig.add_subplot(1, 1, 1)
predicted_data = fig.add_subplot(1, 1, 1)

step = 0

animation = ani.FuncAnimation(fig, draw_graph, interval=2)

plt.xticks([])
plt.yticks([])
plt.show()
'''
