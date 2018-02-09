import numpy as np
import pickle as pkl
import random

"""
This program preprocesses the data as follows:

For each point (x, y) convert a frequency ct z to z*log(len(s_z))*rand(0.95, 1.05) 5 times to augment data

split 1:3 test:train into tsX, tsY, trX, trY pickle files
"""

def process(fstr, mult=True, hy=True):
    f = open(fstr, "rb")
    
    words = f.readline()[1:].split()
    for i in range(len(words)):
        words[i] = len(words[i])

    trainX = []
    trainY = []
    tot = []
    for line in f:
        tot.append(line)
    random.shuffle(tot)
    for line in tot:
        ls = list(map(int, line.split()))
        for iter in range(5 if mult else 1):
            if(hy):
                x = ls[1:]
                y = ls[0]
            else:
                x = ls
                y = []
            for i in range(len(x)):
                x[i] += (random.random()*0.1-0.05)
                x[i] *= (words[i]**(1/2))*(random.random()*0.1+0.95)
            if(sum(x) == 0):
                continue
            x = np.dot(1./(sum(x)*0.5), x) #normalize to sum to 1 if different length review
            trainX.append(x)
            trainY.append(y)
    return trainX, trainY

print("process training data")
trainX, trainY = process("data/train.in", mult=False)

pkl.dump(trainX, open("pkl/totalX.pkl", "w"))
pkl.dump(trainY, open("pkl/totalY.pkl", "w"))

testX = trainX[:int(0.25*len(trainX))]
testY = trainY[:int(0.25*len(trainY))]
trainX = trainX[int(0.25*len(trainX)):]
trainY = trainY[int(0.25*len(trainY)):]

pkl.dump(testX, open("pkl/tsX.pkl", "w"))
pkl.dump(testY, open("pkl/tsY.pkl", "w"))
pkl.dump(trainX, open("pkl/trX.pkl", "w"))
pkl.dump(trainY, open("pkl/trY.pkl", "w"))

print("process predictions")
predX, predY = process("data/pred.in", False, False)

pkl.dump(predX, open("pkl/prX.pkl", "w"))


       


