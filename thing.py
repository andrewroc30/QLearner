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



data_2011 = pd.read_csv("data/GSPC_2011.csv")
closing_prices = data_2011.loc[:,"Close"]
closing_prices = closing_prices.values


def updateWeights(difference, features, weights, alpha):
    return  [x+y for x,y in zip(weights,[i*(alpha * difference) for i in features])]


def qState(features, weights):
    return np.dot(features, weights)


#state: [numStocks, account, prices]
def calcReward(curState, nextState):
    return ((nextState[0] * nextState[len(nextState) - 1]) + nextState[1]) - ((curState[0] * curState[len(curState) - 1]) + curState[1])


 def difference(reward, gamma, actions, curState, action, numDays, v):
     x, index = curState
     max = 2.2250738585072014e-308
     account = x[1]
     num_stocks = x[0]
     next_state = None
     for a in actions:
         if x[0] != 0 and a == "s":
             next_state = newState(v, curState, a, numDays)
     return reward

def newState(v, cur_state, action, numDays):
    x,ind = cur_state
    account = x[1]
    num_stocks = x[0]
    if(action == "s"):
        account  += x[-1]
        num_stocks -= 1
    if(action == "b"):
        account -= x[-1]
        num_stocks += 1
    return ([num_stocks,account] + (getLastNPrices(numDays,ind+1,v).tolist()) ,ind+1)

#print(new_state([2,3,4,5,6,1],([0, 100, 3, 4, 5], 3),"b",3))
#print(new_state([2,3,4,5,6,1],([0, 100, 3, 4, 5], 3),"h",3))
#print(new_state([2,3,4,5,6,1],([0, 100, 3, 4, 5], 3),"s",3))





#state1 = ([0, 1000, 1, 1, 1], 1)
#state2 = ([1, 999, 1, 1, 5], 2)
#print(calcReward(state1, state2))