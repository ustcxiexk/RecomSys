#!/bin/python
'''
Date:20180306
@author: xiexk
'''
import numpy as np
import random 

## download the user-item data
def load_data(path):
    f = open(path)
    data = []
    for line in f.readlines():
        arr = []
        lines = line.strip().split(",")
        for x in lines:
            if x != "-":
                arr.append(float(x))
            else:
                arr.append(float(0))
        data.append(arr)
    return data

## stochastic gradient ascent based method
def gradAscent(data, K):
    dataMat = np.mat(data)
    print (dataMat)
    m, n = np.shape(dataMat)
    p = np.mat(np.random.random((m, K)))
    q = np.mat(np.random.random((K, n)))

    alpha = 0.0002
    beta = 0.02
    maxCycles = 10000
    e = []

    for step in range(maxCycles):
        for i in range(m):
            for j in range(n):
                #print dataMat[i,j]
                if dataMat[i,j] != 0:
                    error = dataMat[i,j]
                    for k in range(K):
                        error = error - p[i,k]*q[k,j]
                    for k in range(K):
                        p[i,k] = p[i,k] + alpha * (2 * error * q[k,j] - beta * p[i,k])
                        q[k,j] = q[k,j] + alpha * (2 * error * p[i,k] - beta * q[k,j])

        loss = 0.0
        for i in range(m):
            for j in range(n):
                if dataMat[i,j] != 0:
                    error = 0.0
                    for k in range(K):
                        error = error + p[i,k]*q[k,j]
                    loss = (dataMat[i,j] - error) * (dataMat[i,j] - error)
                    for k in range(K):
                        loss = loss + beta * (p[i,k] * p[i,k] + q[k,j] * q[k,j]) / 2
                    
        e.append(loss)
        if loss < 0.001:
            break
        #print step
        if step % 1000 == 0:
            print (loss)

    return p, q, e


if __name__ == "__main__":
    dataMatrix = load_data("./IntelligenceRecommendation/data")
    p, q, loss = gradAscent(dataMatrix, 5)
    result = p * q
    print (p)
    print (q)
    print (result)
    np.save('error.npy',loss)
   