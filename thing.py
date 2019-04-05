import numpy as np
import pandas as pd

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


#print(getLastNPrices(3, 5, [2, 3, 4, 5, 6, 1]))

def pastPrices(v, numDays):
    newV = np.zeros((len(v), numDays))
    for row in range(len(v)):
        newV[row] = getLastNPrices(numDays, row, v)
    return newV


def updateWeights(difference, features, weights, alpha):
    # print(difference)
    weights = [x+y for x,y in zip(weights, [i*(alpha * difference) for i in features])]
    denom = sum(weights)
    if denom != 0:
        weights[:] = [x / denom for x in weights]
    return weights


def qState(features, weights):
    return np.dot(features, weights)



#state: [(numStocks, account, prices), index]
def calcReward(curState, nextState):
    return ((nextState[0][0] * nextState[0][len(nextState[0]) - 1]) + nextState[0][1]) \
           - ((curState[0][0] * curState[0][len(curState[0]) - 1]) + curState[0][1])


def difference(gamma, actions, curState, numDays, action, v, weights):
    x, index = curState
    next_state = newState(v, curState, action, numDays)
    max = -2.2250738585072014e308
    num_stocks = x[0]
    next_next_state = None
    q = qState(curState[0], weights)
    for a in actions:
        if num_stocks != 0 and a == "s":
            next_next_state = newState(v, next_state, a, numDays)
            q_prime = qState(next_next_state[0], weights)
            #print("Q'=", q_prime)
            if q_prime > max:
                max = q_prime
        elif a == 'b':
            next_next_state = newState(v, next_state, a, numDays)
            q_prime = qState(next_next_state[0], weights)
            #print("Q'=", q_prime)
            if q_prime > max:
                max = q_prime
        else:
            next_next_state = newState(v, next_state, a, numDays)
            q_prime = qState(next_next_state[0], weights)
            #print("Q'=", q_prime)
            if q_prime > max:
                max = q_prime
    reward = calcReward(curState, next_state)
    #print([reward, gamma, max, q]) q bad
    #print(gamma, actions, curState, numDays, action, weights) weights bad
    #print('MAX=', max)
    #print (reward + (gamma * max))
    #print ("Q:  ", q)
    # print ("total:  ", (reward + (gamma * max)) - q)
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


def chooseBestAction(cur_state, weights, actions, v, numDays):
    best_action = None
    best_actions = []
    max = -2.2250738585072014e308
    for a in actions:
        next_state = newState(v, cur_state, a, numDays)
        if ((cur_state[0][0] > 0 and a == "s") or a != 's'):
            q_prime = qState(next_state[0], weights)
            best_actions.append([a, q_prime])
    best_actions.sort(key = sortSecond)
    return best_actions


def qLearn(alpha, gamma, epsilon, numDays):
    data = pd.read_csv("data/GSPC_2011.csv")
    closing_prices = data.loc[:, "Close"]
    v = closing_prices.values
    actions = ['b', 's', 'h']
    weights = np.zeros(numDays + 2)
    weights = weights.tolist()

    for i in range(1, 1001):
        list = [0, 1000]
        betterList = list + (getLastNPrices(numDays, 0, v)).tolist()
        cur_state = (betterList, 1)
        done = False
        while not done:
            #action = random.choice(chooseBestAction(cur_state, weights, actions, v, numDays))[0]

            best_actions = chooseBestAction(cur_state, weights, actions, v, numDays)
            axs = []
            for a in best_actions:
                axs.append(a[0])
            if len(axs) == 3:
                action = np.random.choice(axs, len(axs), p=[.15, .15, .7])[0]
            elif len(axs) == 2:
                action = np.random.choice(axs, len(axs), p=[.3, .7])[0]

            new_state = newState(v, cur_state, action, numDays)

            d = difference(gamma, actions, cur_state, numDays, action, v, weights)
            weights = updateWeights(d, cur_state[0], weights, alpha)

            cur_state = new_state
            if new_state[1] == len(v) - 2:
                done = True
        if i % 10 == 0:
            print("Episode: ", i, cur_state[0][1])
    return cur_state

print(qLearn(.01, .9, .1, 10))



#print(new_state([2,3,4,5,6,1],([0, 100, 3, 4, 5], 3),"b",3))
#print(new_state([2,3,4,5,6,1],([0, 100, 3, 4, 5], 3),"h",3))
#print(new_state([2,3,4,5,6,1],([0, 100, 3, 4, 5], 3),"s",3))

# state: ([numStocks, account_balance, lastNprices], index)
#state1 = ([0, 1000, 1, 1, 1], 1)
#state2 = ([1, 999, 1, 1, 5], 2)
#print(calcReward(state1, state2))