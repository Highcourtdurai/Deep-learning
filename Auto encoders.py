#same like principle component analysis,auto encoders also unsupervised learning

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

(x_train,y_train),(x_test,y_test)=tf.keras.datasets.fashion_mnist.load_data()

#2 dimensional data

print(x_train.shape)
print(x_test.shape)

x_train=x_train/255
x_test=x_test/255

class AutoEncoder(tf.keras.Model):
    def __init__(self):#Constructor
        super().__init__(name="autoencoder")
        self.encoder=tf.keras.Sequential([tf.keras.layers.Flatten(),tf.keras.layers.Dense(64,activation="relu")])#We indicate "relu" to show only +ve output but not negative output
        self.decoder=tf.keras.Sequential([tf.keras.layers.Dense(784,activation="relu"),tf.keras.layers.Reshape((28,28))])
        
    def call(self,x):
        encode=self.encoder(x)
        decode=self.decoder(encode)
        return decode
    
model=AutoEncoder()

model.compile(optimizer=tf.optimizers.Adam(),loss=tf.losses.MeanSquaredError())

model.fit(x_train,x_train,validation_data=(x_test,x_test),epochs=15)

encoded_images=model.encoder(x_test)
decoded_images=model.decoder(encoded_images).numpy()

n=10
plt.figure(figsize=(15,10))
for i in range(10):
    ax=plt.subplot(2,n,i+1)
    plt.imshow(x_test[i])
    plt.title("original")
    plt.gray()
    plt.xticks([])
    plt.yticks([])
    
    ax=plt.subplot(2,n,i+11)
    plt.imshow(decoded_images[i])
    plt.title("decoded")
    plt.gray()
    plt.xticks([])
    plt.yticks([])
    
    
#Noise adding and removing

(x_train,y_train),(x_test,y_test)=tf.keras.datasets.fashion_mnist.load_data()
x_train=x_train/255
x_test=x_test/255
x_train_noisy=x_train+0.2*tf.random.normal(shape=x_train.shape)
x_test_noisy=x_test+0.2*tf.random.normal(shape=x_test.shape)


x_train_noisy=tf.clip_by_value(x_train_noisy,clip_value_min=0.,clip_value_max=1.)
x_test_noisy=tf.clip_by_value(x_test_noisy,clip_value_min=0.,clip_value_max=1.)

denoise=AutoEncoder()

denoise.compile(optimizer=tf.optimizers.Adam(),loss=tf.losses.MeanSquaredError())

denoise.fit(x_train_noisy,x_train,validation_data=(x_test_noisy,x_test),epochs=15)

encoded_images=model.encoder(x_test_noisy)
decoded_images=model.decoder(encoded_images).numpy()

n=10
plt.figure(figsize=(15,10))
for i in range(10):
    ax=plt.subplot(2,n,i+1)
    plt.imshow(x_test_noisy[i])
    plt.title("original")
    plt.gray()
    plt.xticks([])
    plt.yticks([])
    
    ax=plt.subplot(2,n,i+11)
    plt.imshow(decoded_images[i])
    plt.title("decoded")
    plt.gray()
    plt.xticks([])
    plt.yticks([])



           
