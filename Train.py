# -*- coding: utf-8 -*-
"""
Created on Mon May 24 10:32:28 2021

@author: Admin
"""
import keras
import numpy as np
import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.datasets import mnist
from keras.layers import Dense,Dropout,Conv2D,MaxPooling2D,Flatten
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten,LeakyReLU
from keras.layers import Conv2D, MaxPooling2D,BatchNormalization
from keras import backend as K
from keras import optimizers
from keras.models import load_model
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
IMG_HEIGHT,IMG_WIDTH=int(300),int(350)
input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)
path=r"C:\Users\Admin\Desktop\Ai_test\T3"
path2=r"C:\Users\Admin\Desktop\Ai_test\TestImage"
path3=r"C:\Users\Admin\Desktop\Ai_test\test"
def show_image(image,prediction,num):
    
    prediction = model.predict_classes(image)
    for i in range(0,num):
        fig = plt.gcf()
        fig.set_size_inches(2,2)
        s = 'A'
        if prediction[i] == 0:
            s = 'AI_Predict:A'
        elif prediction[i] == 1:
            s = 'AI_Predict:B'
        else:
            s = 'AI_Predict:C'
        plt.title(s)
        plt.imshow(image[i], cmap='binary')
        plt.show()
bs = 32   
train_data = []
class_name = []
dict_label = {"A":0,"B":1}
train_label = []
for dir1 in os.listdir(path3):
    img_path = os.path.join(path3, dir1)
    for file in os.listdir(img_path):
        filepath = os.path.join(img_path,file)
        img = Image.open(filepath)
        img = np.array(img)
        train_data.append(img)
        class_name.append(dir1)
for i in class_name:
    train_label.append(int(dict_label[i]))
train_label = np.array(train_label)
train_data = np.array(train_data)
#train_date=np.zeros((0,300, 350), dtype=np.ubyte)
print(train_data.shape)
print(train_label.shape)
#print(type(train_data))
#print(train_data.shape)
#print(train_data.shape)
#print(train_label.shape)
train_data = train_data.reshape(train_data.shape[0],300,350,3).astype('float32')
train_data = train_data / 255
#print(train_data[0])

#print(train_label)
train_label = np_utils.to_categorical(train_label,2)
print(type(train_data))
print(type(train_label))
print(train_data.shape)
print(train_label.shape)
#print(train_label[0:5])
#show_image(train_data[100])

#np.save("train_data",train_data)
#np.save("train_label",train_label)


'''
model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ]
)
'''
model = Sequential()

model.add(Conv2D(filters=10,
                 kernel_size=(3,3),
                 padding='same',
                 input_shape=(300,350,3),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=20,
                 kernel_size=(3,3),
                 padding='same',
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())

model.add(Dense(units=256,
                activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=128,
                activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=10,
                activation='softmax'))

#model.summary()

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    path,
    target_size=(300, 350),
    batch_size=bs,
    class_mode='binary'
)
#color_mode='grayscale'
validation_generator = test_datagen.flow_from_directory(
    path2,
    target_size=(300, 350),
    batch_size=bs,
    class_mode='binary'
)
train_history = model.fit_generator(
    train_generator,
    steps_per_epoch=(train_data.shape[0]/bs)+1,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=1
)
'''
'''
prediction = model.predict_classes(train_data)
show_image(train_data, prediction, 10)
#model.load_weights('test.weight')

#model.load_weights('test.weight')

#train_history = model.fit(x=train_data,y=train_label,epochs=10)
