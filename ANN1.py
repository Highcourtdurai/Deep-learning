import pandas as pd
import numpy as np

data=pd.read_csv("Churn_Modelling.csv")
data=pd.get_dummies(data,columns=["Geography"])

from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
data["Gender"]=lb.fit_transform(data["Gender"])


x=data.drop(columns=["RowNumber","CustomerId","Surname","Exited"]).values
y=data.loc[:,"Exited"].values

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

classifier.add(Dense(units=6,input_dim=12,kernel_initializer="uniform",activation='relu'))

classifier.add(Dense(units=6,kernel_initializer="uniform",activation='relu'))

classifier.add(Dense(units=1,activation="sigmoid"))

classifier.compile(loss='binary_crossentropy',optimizer='adam',metrics=["accuracy"])

classifier.fit(x_train,y_train,batch_size=200,epochs=100)


y_pred=classifier.predict(x_test)
y_pred=(y_pred>0.6)

from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test,y_pred)
acc=accuracy_score(y_test,y_pred)
print(acc)

