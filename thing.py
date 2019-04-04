import numpy as np
import pandas as pd

def getLastNPrices(numDays, row, v):
    lastNPrices = np.zeros(numDays)
    if row >= numDays:
        for i in range(numDays):
            lastNPrices[i] = v[row - numDays + i]
    else:
        avg = 0
        for val in v:
            avg = avg + val
        avg = avg / len(v)
        for i in range(numDays):
            lastNPrices[i] = avg
    return lastNPrices

def pastPrices(v, numDays):
    newV = np.zeros((len(v), numDays))
    for row in range(len(v)):
        newV[row] = getLastNPrices(numDays, row, v)
    return newV



data_2011 = pd.read_csv("data/GSPC_2011.csv")
closing_prices = data_2011.loc[:,"Close"]
closing_prices = closing_prices.values


#vec = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
#print(pastPrices(vec, 2))
#print(pastPrices(vec, 1))
print(pastPrices(closing_prices, 10))