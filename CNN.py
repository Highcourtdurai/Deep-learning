from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import random

images=[]
labels=[]

for label_path in os.listdir("test_set"):
    im=os.listdir(os.path.join("test_set",label_path))
    im=[random.choice(im ) for i in range(1000)]
    #print(im)
    for image_path in im:
        img=Image.open(os.path.join("test_set",label_path,image_path))
        if img.mode !="RGB":
            img=img.convert("RGB")
        arr=np.array(img)
        resized_img=np.array(Image.fromarray(arr).resize((64,64)))
        images.append(resized_img)
        labels.append(label_path)
        
x=np.array(images)
y=np.array(labels)

x=x/255#Normalizing

from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
y=lb.fit_transform(y)

from sklearn.model_selection import train_test_split
x_train,x_test_total,y_train,y_test_total=train_test_split(x,y,test_size=0.25,random_state=0)
x_test,x_val,y_test,y_val=train_test_split(x_test_total,y_test_total,test_size=0.25,random_state=0)


import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

model=Sequential()#To create model

model.add(Conv2D(filters=16,kernel_size=(3,3),strides=1,activation="relu",input_shape=(64,64,3)))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(filters=32,kernel_size=(3,3),activation="relu"))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())

model.add(Dense(units=500,activation="relu",kernel_initializer="uniform"))
model.add(Dense(units=128,activation="relu",kernel_initializer="uniform"))

model.add(Dense(units=1,activation="sigmoid"))

model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])

model.fit(x_train,y_train,validation_data=(x_val,y_val),epochs=30,batch_size=100)

model.save("cats&dogs.h5")


from keras.preprocessing import image
from keras.models import load_model

test_image=image.load_img("C:/Users/Galaxy/Documents/Anaconda(Spyder)/BIGDATA HANDSON/Deep Learning/test_set/dogs/dog.4050.jpg",target_size=(64,64,3))
plt.imshow(test_image)
plt.show()
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)
model=load_model("cats&dogs.h5")
result=model.predict(test_image)



































