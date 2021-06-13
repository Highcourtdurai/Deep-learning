import tensorflow as tf

import numpy as np
# a=np.array(([1,2,3,4,5],[6,7,8,9,10]),dtype=np.float32)
# b=tf.constant(0.3)
# c=tf.Variable(a,dtype=tf.int64)

# #Eiger tensor 2.x
# a=tf.constant(5)
# b=tf.constant(4)
# print(a-b)
# print(a+b)
# print(a/b)
# print(a*b)

# print(tf.__version__)


#We can use version 1.x also in google colab by using below code and set as restart runtime again 

#%tensorflow_version_ 1.x

# a=tf.constant(5)
# b=tf.constant(4)

# with tf.Session() as sess:
#     result=sess.run(a+b)
#     print(result)
    
# x=tf.constant(np.array([1,2,3,4,5]),dtype=np.float32)
# w=tf.Variable(0.3)#slope
# b=tf.Variable(0.9)#intercept

# y=tf.add(tf.multiply(w,x),b)
# print(y)


#AI model


import pandas as pd
dataset=pd.read_csv("Churn_Modelling.csv")
dataset.head(5)

from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
dataset["Gender"]=lb.fit_transform(dataset["Gender"])

dataset=pd.get_dummies(dataset,columns=["Geography"])

dataset.head(5)


y=dataset.loc[:,'Exited'].values
x=dataset.drop(columns=["RowNumber","CustomerId","Exited","Surname"]).values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

from sklearn.preprocessing import StandardScaler
sd=StandardScaler()
x_train=sd.fit_transform(x_train)
x_test=sd.transform(x_test)

# print(x_train.dtype)
# print(x_test.dtype)
# print(y_train.dtype)
# print(y_test.dtype)
# print(x_train.shape)

import numpy as np

x_train,x_test=np.array(x_train,dtype=np.float32),np.array(x_test,dtype=np.float32)


training_data=tf.data.Dataset.from_tensor_slices((x_train,y_train))
training_data=training_data.repeat().batch(32).prefetch(1)


n_dim=12
n_hidden1=60
n_hidden2=60
out=1
learning_rate=0.005

initializer=tf.initializers.glorot_uniform()

weights=[tf.Variable(initializer(shape=[n_dim,n_hidden1])),tf.Variable(initializer(shape=[n_hidden1,n_hidden2])),tf.Variable(initializer(shape=[n_hidden2,out]))]

biases=[tf.Variable(initializer(shape=[n_hidden1])),tf.Variable(initializer(shape=[n_hidden2])),tf.Variable(initializer(shape=[out]))]                                   

def model(x):
    layer_1=tf.add(tf.matmul(x,weights[0]),biases[0])
    layer_1=tf.nn.relu(layer_1)
    
    layer_2=tf.add(tf.matmul(layer_1,weights[1]),biases[1])
    layer_2=tf.nn.relu(layer_2)
    
    out=tf.add(tf.matmul(layer_2,weights[2]),biases[2])
    return tf.nn.sigmoid(out)
    

def calculate_loss(y_pred,y_true):
    y_pred=tf.clip_by_value(y_pred,1e-9,1.)
    return tf.reduce_mean(tf.losses.binary_crossentropy(y_true,y_pred))


def acc(y_pred,y_true):
    correct=tf.equal(tf.cast(y_pred>0.5,tf.int64),tf.cast(y_true,tf.int64))
    return tf.reduce_mean(tf.cast(correct,dtype=tf.float32))


optimizer=tf.optimizers.Adam(learning_rate)

def train_step(x,y):
    with tf.GradientTape() as tape:
        pred=model(x)
        loss=calculate_loss(pred,y)
        
        variables=weights+biases
        grad=tape.gradient(loss,variables)
        
    optimizer.apply_gradients(zip(grad,variables))
    return pred,loss

for epoch in range(100):
    accuracy_history=[]
    loss_history=[]
    for step,(batch_x,batch_y) in enumerate(training_data.take(x_train.shape[0]//32),1):
        pred,loss=train_step(batch_x,batch_y)
        accuracy=acc(pred,batch_y)
        print(float(accuracy),float(loss))
        

        
#tf.saved_model.save(model.export_dir="Churn_Modelling.csv")     
        
        
        
        
        
        
        










