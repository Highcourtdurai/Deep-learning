import pandas as pd
import numpy as np

data=pd.read_csv("kc_house_data.csv")

x=data.iloc[:,3:].values
y=data.iloc[:,2].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

from sklearn.preprocessing import StandardScaler
sd=StandardScaler()
x_train=sd.fit_transform(x_train)
x_test=sd.transform(x_test)

import keras
from keras.models import Sequential
from keras.layers import Dense

classifier=Sequential()

classifier.add(Dense(units=9,input_dim=18,kernel_initializer="uniform",activation='relu'))

classifier.add(Dense(units=9,kernel_initializer="uniform",activation='relu'))

classifier.add(Dense(units=1))

classifier.compile(loss='mean_squared_error',optimizer='adam',metrics=["accuracy"])

classifier.fit(x_train,y_train,batch_size=500,epochs=100)

