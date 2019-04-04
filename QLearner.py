import random
import numpy as np


def getStockDataVec(key):
    vec = []
    lines = open("data/" + key + ".csv", "r").read().splitlines()
    for line in lines[1:]:
        vec.append(float(line.split(",")[4]))
    return vec


actions = ['buy', 'sell', 'hold']
#Hyparameters
gamma = 0.9
alpha = 0.1 #learning rate
epsilon = 0.1
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.01
all_epochs = []
all_penalities = []
penalities = 0


q_table = None
balance = 1000





for i in range(1, 100001):
    #state = env.reset()
    epochs, penalites, reward = 0, 0, 0
    done = False
    while not done:
        if random.uniform(0, 1) < epsilon:
            #action = env.action_space.sample()
            action = actions.sample()
        else:
            action = np.argmax(q_table[state])
        #next_state, reward, done, info = env.step(action)
        next_max = np.max(q_table[next_state])
        q_table[state, action] = q_table[state, action] + alpha * (reward + gamma * next_max - q_table[state, action])
        if reward == -10:
            penalities += 1
        state = next_state
        epochs += 1
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-0.1 * epsilon)
    if i % 100 == 0:
        #clear_output(wait=True)
        print('Episode: {}'.format(i))

print('Training Finished..')