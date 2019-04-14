import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def getLastNPrices(numDays, row, v):
    lastNPrices = np.zeros(numDays)
    if row >= numDays:
        for i in range(numDays):
            lastNPrices[i] = v[row - numDays + i + 1]
    else:
        avg = 0
        for val in v:
            avg = avg + val
        avg = avg / len(v)
        for i in range(numDays):
            lastNPrices[i] = avg
    lastNPrices[-1] = v[row]
    return lastNPrices


# def pastPrices(v, numDays):
#     newV = np.zeros((len(v), numDays))
#     for row in range(len(v)):
#         newV[row] = getLastNPrices(numDays, row, v)
#     return newV


# def updateWeights(difference, features, weights, alpha):
#     weights = [x+y for x,y in zip(weights, [i*(alpha * difference) for i in features])]
#     denom = sum(weights)
#     if denom != 0:
#         weights[:] = [x / denom for x in weights]
#     return weights

# def updateWeights(difference, features, weights, alpha):
#     weights = [x + y for x, y in zip(weights, [i * (alpha * difference) for i in features])]
#     max = -2.2250738585072014e308
#     min = 2.2250738585072014e308
#     for w in weights:
#         if w > max:
#             max = w
#         elif w < min:
#             min = w
#     count = 0
#     for w in weights:
#         if w != 0:
#             weights[count] = (w - min) / (max - min)
#         count += 1
#     return weights

def updateWeights(difference, features, weights, alpha):
    weights = [x + y for x, y in zip(weights, [i * (alpha * difference) for i in features])]
    max = -2.2250738585072014e308
    min = 2.2250738585072014e308
    for w in weights:
        if w > max:
            max = w
        elif w < min:
            min = w
    count = 0
    for w in weights:
        if w != 0:
            weights[count] = (2 * ((w - min) / (max - min))) - 1
        count += 1
    return weights


def qState(features, weights):
    return np.dot(features, weights)


# state: [(numStocks, account, prices), index]
#def calcReward(curState, nextState):
#    return ((nextState[0][0] * nextState[0][len(nextState[0]) - 1]) + nextState[0][1]) \
#           - ((curState[0][0] * curState[0][len(curState[0]) - 1]) + curState[0][1])


# def calcReward(curState, nextState):
#     return (nextState[0][0] * nextState[0][len(nextState[0]) - 1]) \
#            - (curState[0][0] * curState[0][len(curState[0]) - 1])


def calcReward(action, nextState, bought_prices):
    if action == 'h':
        return 0
    elif action == 's':
        if len(bought_prices) == 0:
            return -1 # this should never happen
        return (nextState[0][-1] - bought_prices[-1])
    else: #action is buying
        if len(bought_prices) == 0:
            return 1
        return nextState[0][-1] - bought_prices[-1]


def difference(gamma, actions, curState, numDays, action, v, weights, bought_prices):
    x, index = curState
    next_state = newState(v, curState, action, numDays)
    max = -2.2250738585072014e308
    num_stocks = x[0]
    q = qState(curState[0], weights)
    for a in actions:
        if num_stocks != 0 and a == "s":
            next_next_state = newState(v, next_state, a, numDays)
            q_prime = qState(next_next_state[0], weights)
            if q_prime > max:
                max = q_prime
        elif a == 'b':
            next_next_state = newState(v, next_state, a, numDays)
            q_prime = qState(next_next_state[0], weights)
            if q_prime > max:
                max = q_prime
        else:
            next_next_state = newState(v, next_state, a, numDays)
            q_prime = qState(next_next_state[0], weights)
            if q_prime > max:
                max = q_prime
    reward = calcReward(action, next_state, bought_prices)
    return (reward + (gamma * max)) - q


def newState(v, cur_state, action, numDays):
    x,ind = cur_state
    account = x[1]
    num_stocks = x[0]
    if(action == "s"):
        account += x[-1]
        num_stocks -= 1
    if(action == "b"):
        account -= x[-1]
        num_stocks += 1
    return ([num_stocks,account] + (getLastNPrices(numDays,ind+1,v).tolist()) ,ind+1)


def sortSecond(val):
    return val[1]


def getActionsQStates(cur_state, weights, actions, v, numDays):
    best_actions = []
    for a in actions:
        next_state = newState(v, cur_state, a, numDays)
        if ((cur_state[0][0] > 0 and a == "s") or a != 's'):
            q_prime = qState(next_state[0], weights)
            best_actions.append([a, q_prime])
    best_actions.sort(key = sortSecond)
    return best_actions


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


def updateBoughtPrices(new_state, bought_prices, action):
    if action == 'b':
        bought_prices.append(new_state[0][-1])
    elif action == 's':
        bought_prices.pop()


def chooseBestAction(epsilon, best_actions):
    axs = []
    for a in best_actions:
        #print(a)
        axs.append(a[0])
    #print('----------')
    if len(axs) == 3:
        #return np.random.choice(axs, len(axs), p=[epsilon, (1 - epsilon) / 2, (1 - epsilon) / 2])[0]
        return np.random.choice(axs, len(axs), p=[(1 - epsilon)/2, (1 - epsilon)/2, epsilon])[0]
    elif len(axs) == 2:
        #return np.random.choice(axs, len(axs), p=[epsilon, 1 - epsilon])[0]
        return np.random.choice(axs, len(axs), p=[1-epsilon, epsilon])[0]


def qLearn(alpha, gamma, epsilon, numDays, episodes):
    np.random.seed(50)

    data = pd.read_csv("data/GSPC_2011.csv")
    closing_prices = data.loc[:, "Close"]
    v = closing_prices.values

    actions = ['b', 's', 'h']
    weights = np.zeros(numDays + 2)
    weights = weights.tolist()

    epsilon_change = (1 - epsilon) / episodes

    end_states = []
    end_weights = []
    end_actions = []

    for i in range(1, episodes+1):
        list = [0, 10000]
        betterList = list + (getLastNPrices(numDays, 0, v)).tolist()
        cur_state = (betterList, 1)
        bought_prices = []
        epsilon = epsilon + epsilon_change
        done = False
        while not done:
            best_actions = getActionsQStates(cur_state, weights, actions, v, numDays)
            action = chooseBestAction(epsilon, best_actions)
            new_state = newState(v, cur_state, action, numDays)
            updateBoughtPrices(new_state, bought_prices, action)

            if i == episodes:
                end_actions.append(action)

            d = difference(gamma, actions, cur_state, numDays, action, v, weights, bought_prices)
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


print(qLearn(.6, .1, .9, 6, 1000))


# state: ([numStocks, account_balance, lastNprices], index)
# state1 = ([0, 1000, 1, 1, 1], 1)
# state2 = ([1, 999, 1, 1, 5], 2)
