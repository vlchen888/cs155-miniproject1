import numpy as np 
import tensorflow as tf 
import keras
import pickle
import os
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout

xTrain = pickle.load(open("pkl/totalX.pkl", "r"))
yTrain = pickle.load(open("pkl/totalY.pkl", "r"))
xPred = pickle.load(open("pkl/prX.pkl", "r"))

#process data function

def process(x, y):
    yoh = []
    xfl = []
    
    #onehot
    for i in range(len(y)):
        yoh.append(keras.utils.np_utils.to_categorical(y[i], 2))
        xfl.append(np.ndarray.flatten(x[i]))
        
    yoh = np.array(yoh)
    xfl = np.array(xfl)

    #normalize training x
    return xfl, yoh

xTrain, yTrain = process(xTrain, yTrain)

#model

model = Sequential()

model.add(Dense(250, input_shape=(len(xTrain[0]),)))
model.add(Activation('relu'))
model.add(Dropout(0.9))

model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dropout(0.3))

model.add(Dense(20))
model.add(Activation('softmax'))
model.add(Dropout(0.3))

#output
model.add(Dense(2))
model.add(Activation('softmax'))

model.summary()

adam = keras.optimizers.Adam(lr = 1e-3)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

fit = model.fit(xTrain, yTrain, batch_size=50, nb_epoch=4, verbose=1)

score = model.evaluate(xTrain, yTrain, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])


p = model.predict(np.array(xPred))
print(p)
po = open("predout.csv", "wb")
po.write("ID,Prediction\n")
for i in range(len(p)):
    po.write(str(i+1) + ",")
    po.write("0\n" if p[i][0] > p[i][1] else "1\n")
