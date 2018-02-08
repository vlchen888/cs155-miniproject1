import numpy as np 
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
print("done importing")

#load data

xTrain = pickle.load(open("pkl/trX.pkl", "r"))
xTest = pickle.load(open("pkl/tsX.pkl", "r"))
yTrain = pickle.load(open("pkl/trY.pkl", "r"))
yTest = pickle.load(open("pkl/tsY.pkl", "r"))
xPred = pickle.load(open("pkl/prX.pkl", "r"))

print("done loading")
#model

rf = RandomForestClassifier(n_estimators = 200, max_leaf_nodes = 200, n_jobs=12)
rf.fit(xTrain, yTrain)

print(accuracy_score(rf.predict(xTrain), yTrain))
print(accuracy_score(rf.predict(xTest), yTest))

"""
p = model.predict(np.array(xPred))
print(p)
po = open("predout.csv", "wb")
po.write("ID,Prediction\n")
for i in range(len(p)):
    po.write(str(i+1) + ",")
    po.write("0\n" if p[i][0] > p[i][1] else "1\n")
"""
