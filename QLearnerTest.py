import numpy as np
import matplotlib.pyplot as plt


sample_data = [100, 500, 700, 900, 1000, 1100, 1200, 300, 1400, 1500, 100, 100, 100, 100, 100, 100, 0]
sample_choices = [1, 1, 1, 1, 2, 2, 2, 1, 2, 2, 3, 3, 3, 3, 3, 3]
#1 = buy, 2=sell, 3=hold


def updateWeights(difference, features, weights, alpha):
    weights = [x+y for x,y in zip(weights, [i*(alpha * difference) for i in features])]
    denom = sum(weights)
    if denom != 0:
        weights[:] = [x / denom for x in weights]
    return weights


def qState(features, weights):
    return np.dot(features, weights)


def calcReward(curState, nextState):
    return (nextState[0][0] * nextState[0][len(nextState[0]) - 1]) \
           - (curState[0][0] * curState[0][len(curState[0]) - 1])


def difference(gamma, actions, curState, action, v, weights):
    x, index = curState
    next_state = newState(v, curState, action)
    max = -2.2250738585072014e308
    num_stocks = x[0]
    q = qState(curState[0], weights)
    for a in actions:
        if num_stocks != 0 and a == "s":
            next_next_state = newState(v, next_state, a)
            q_prime = qState(next_next_state[0], weights)
            if q_prime > max:
                max = q_prime
        elif a == 'b':
            next_next_state = newState(v, next_state, a)
            q_prime = qState(next_next_state[0], weights)
            if q_prime > max:
                max = q_prime
        else:
            next_next_state = newState(v, next_state, a)
            q_prime = qState(next_next_state[0], weights)
            if q_prime > max:
                max = q_prime
    reward = calcReward(curState, next_state)
    return (reward + (gamma * max)) - q


def sortSecond(val):
    return val[1]


def chooseBestAction(cur_state, weights, actions, v):
    best_actions = []
    max = -2.2250738585072014e308
    for a in actions:
        next_state = newState(v, cur_state, a)
        if ((cur_state[0][0] > 0 and a == "s") or a != 's'):
            q_prime = qState(next_state[0], weights)
            best_actions.append([a, q_prime])
    best_actions.sort(key = sortSecond)
    return best_actions


def newState(v, cur_state, action):
    x, ind = cur_state
    account = x[1]
    num_stocks = x[0]
    if (action == "s"):
        account += x[-1]
        num_stocks -= 1
    if (action == "b"):
        account -= x[-1]
        num_stocks += 1
    if (len(sample_choices) > ind + 1):
        return ([num_stocks, account, sample_choices[ind + 1]], ind + 1)
    return ([num_stocks, account, sample_choices[ind]], ind + 1)


def plotBalances(states):
    x = []
    y = []
    count = 1
    for s in states:
        y.append(s[0][1])
        x.append(count)
        count += 1
    plt.plot(x, y)
    plt.xlabel('episode')
    plt.ylabel('balance')
    plt.title('Learning Curve')
    plt.show()


def plotWeights(weights, episodes):
    all_weights = []
    x = []
    for i in range(len(weights[0])):
        temp = []
        for j in range(len(weights)):
            temp.append(weights[j][i])
        all_weights.append(temp)
    for i in range(episodes):
        x.append(i)
    count = 0
    for w in all_weights:
        plt.plot(x, w, label=count)
        count += 1
    plt.legend(loc="lower left")
    plt.xlabel('episode')
    plt.ylabel('weighting')
    plt.title('Weights')
    plt.show()


def plotChoices(v, choices):
    plt.plot(v)
    count = 0
    for i in v:
        if count >= len(choices):
            print('this is dumb')
        elif choices[count] == 'b':
            plt.scatter(count, i, c='green')
        elif choices[count] == 's':
            plt.scatter(count, i, c='red')
        elif choices[count] == 'h':
            plt.scatter(count, i, c='blue')
        count += 1
    plt.show()


def qLearn(alpha, gamma, epsilon, episodes):
    np.random.seed(50)

    v = sample_data

    actions = ['b', 's', 'h']
    weights = np.zeros(3)
    weights = weights.tolist()

    end_states = []
    end_weights = []
    end_actions = []

    for i in range(1, episodes+1):
        list = [0, 10000]
        betterList = list + [sample_choices[0]]
        cur_state = (betterList, 1)
        done = False
        while not done:
            best_actions = chooseBestAction(cur_state, weights, actions, v)
            axs = []
            for a in best_actions:
                axs.append(a[0])
            if len(axs) == 3:
                #action = np.random.choice(axs, len(axs), p=[epsilon, (1 - epsilon)/2, (1 - epsilon)/2])[0]
                action = np.random.choice(axs, len(axs), p=[(1 - epsilon)/2, (1 - epsilon)/2, epsilon])[0]
            elif len(axs) == 2:
                #action = np.random.choice(axs, len(axs), p=[epsilon, 1-epsilon])[0]
                action = np.random.choice(axs, len(axs), p=[1-epsilon, epsilon])[0]

            new_state = newState(v, cur_state, action)

            if i == episodes:
                end_actions.append(action)

            d = difference(gamma, actions, cur_state, action, v, weights)
            weights = updateWeights(d, cur_state[0], weights, alpha)

            cur_state = new_state
            if new_state[1] == len(v) - 2:
                done = True
        if i % 10 == 0:
            print("Episode: ", i, cur_state[0][1])
        end_states.append(cur_state)
        end_weights.append(weights)
    plotChoices(v, end_actions)
    plotBalances(end_states)
    plotWeights(end_weights, episodes)
    return cur_state

print(qLearn(.6, .1, .9, 1000))