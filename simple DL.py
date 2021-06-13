import tensorflow as tf
print(tf.__version__)

import numpy as np
from numpy import loadtxt

data=loadtxt("diabetesclean.csv",delimiter=',')

x=data[:,0:8]
y=data[:,-1]

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2)

from keras.models import Sequential
from keras.layers import Dense

model=Sequential()
model.add(Dense(8,input_dim=8,activation="relu")) #INPUT N
model.add(Dense(8,activation="relu")) #HIDDEN LAYER
model.add(Dense(1,activation="sigmoid")) #Output layer

model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])

model.fit(xtrain,ytrain,epochs=200,batch_size=43)

_,acc=model.evaluate(xtest,ytest)
print("Accuracy score for test data:",acc)

