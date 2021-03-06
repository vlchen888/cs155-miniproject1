import numpy as np
import tensorflow as tf
import keras
import pickle
import os
import sys
from keras.models import Sequential
from keras.layers import *

xTrain = pickle.load(open("pkl/trX.pkl", "r"))
xTest = pickle.load(open("pkl/tsX.pkl", "r"))
yTrain = pickle.load(open("pkl/trY.pkl", "r"))
yTest = pickle.load(open("pkl/tsY.pkl", "r"))
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
xTest, yTest = process(xTest, yTest)

#model

model = Sequential()

model.add(Reshape((40, 25, 1), input_shape=(1000,)))

model.add(Conv2D(25, (5,5), strides=1, padding='same', activation='relu'))
model.add(MaxPooling2D((5,5), strides=None, padding='same'))
model.add(Conv2D(50, (5,5), strides=1, padding='same', activation='relu'))
model.add(Reshape((8*5*50,)))
model.add(Dense(30))
model.add(BatchNormalization())
model.add(Dropout(0.6))
model.add(Activation('relu'))
#output
model.add(Dense(2))
model.add(Activation('softmax'))

model.summary()

adam = keras.optimizers.Adam(lr = 5e-2)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

for i in range(20):
    fit = model.fit(xTrain, yTrain, batch_size=100, nb_epoch=1, verbose=1)
    
    ## Printing the accuracy of our model, according to the loss function specified in model.compile above
    score = model.evaluate(xTest, yTest, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

"""
p = model.predict(np.array(xPred))
print(p)
po = open("predout.csv", "wb")
po.write("ID,Prediction\n")
for i in range(len(p)):
    po.write(str(i+1) + ",")
    po.write("0\n" if p[i][0] > p[i][1] else "1\n")
"""
