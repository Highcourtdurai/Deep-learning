import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#Fasion mnist=data of accesories like boats,dresses,bags etc


fashion_mnist=tf.keras.datasets.fashion_mnist
(train_images,train_labels),(test_images,test_labels)=fashion_mnist.load_data()

print(train_images.shape)

print(train_labels.shape)

print(test_images.shape)

print(test_labels.shape)

plt.imshow(train_images[4])
plt.show()

model=tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(units=120,activation="relu"))
model.add(tf.keras.layers.Dense(units=10,activation="softmax"))

# model.compile(optimizer=tf.keras.optimizers.Adam(0.01),loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),metrics=["accuracy"])
# model.fit(train_images,train_labels,epochs=20,batch_size=500)

def cross_entropy(y_pred,y_true):
    return tf.reduce_mean(tf.keras.losses.SparseCategoricalCrossentropy(y_true,y_pred))

def accuracy(y_pred,y_true):
    correct_prediction=tf.equal(tf.cast(y_pred,tf.int64),tf.cast(y_true,tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


optimizer=tf.optimizers.Adam()

def train_step(x,y):
    with tf.GradientTape() as tape:
        pred=tf.argmax(model.predict(x),axis=1)
        loss=cross_entropy(pred,y)
        
        trainable_variables=model.trainable_variables
        gradients=tape.gradient(loss,trainable_variables)
        
    optimizer.apply_gradients(zip(gradients,trainable_variables))
    return pred,loss

train_data=tf.data.Dataset.from_tensor_slices((train_images,train_labels))
train_data=train_data.repeat().shuffle(100).batch(32).prefetch(1)



for epoch in range(20):
    for step,(batch_x,batch_y) in enumerate(train_data.take(train_images.shape[0]//32),1):
        pred,loss=train_step(batch_x,batch_y)
        acc=accuracy(pred,batch_y)
        print(acc,loss)










