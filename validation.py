# -*- coding: utf-8 -*-
"""
Created on Mon May 31 08:56:02 2021

@author: Admin
"""

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
path2=r"C:\Users\Admin\Desktop\Ai_test\T3"
path3=r"C:\Users\Admin\Desktop\Ai_test\test"
def show_image(image,prediction,num):
    
    prediction = model.predict_classes(image)
    for i in range(0,num):
        fig = plt.gcf()
        fig.set_size_inches(4,4)
        s = 'A'
        if prediction[i] == 0:
            s = 'AI_Predict:Good'
        elif prediction[i] == 1:
            s = 'AI_Predict:Empty'
        elif prediction[i] == 2:
            s = 'AI_Predict:Empty2'
        else:
            s = 'AI_Predict:Bad'
        plt.title(s)
        plt.imshow(image[i], cmap='binary')
        plt.show()
bs = 32   
test_data = []
class_name = []
dict_label = {"A":0,"B":1}
test_label = []
for dir1 in os.listdir(path3):
    img_path = os.path.join(path3, dir1)
    for file in os.listdir(img_path):
        filepath = os.path.join(img_path,file)
        img = Image.open(filepath)
        img = np.array(img)
        test_data.append(img)
        class_name.append(dir1)
for i in class_name:
    test_label.append(int(dict_label[i]))
test_label = np.array(test_label)
test_label_num = test_label
test_data = np.array(test_data)
#train_date=np.zeros((0,300, 350), dtype=np.ubyte)
print(test_data.shape)
print(test_label.shape)
#print(type(test_data))
#print(test_data.shape)
#print(test_data.shape)
#print(test_label.shape)
test_data = test_data.reshape(test_data.shape[0],300,350,3).astype('float32')
test_data = test_data / 255
#print(test_data[0])

#print(test_label)
test_label = np_utils.to_categorical(test_label,2)
print(type(test_data))
print(type(test_label))
print(test_data.shape)
print(test_label.shape)
#print(test_label[0:5])
#show_image(test_data[100])

#np.save("test_data",test_data)
#np.save("test_label",test_label)

model = load_model("ABC.h5")
prediction = model.predict_classes(test_data)
show_image(test_data, prediction, test_data.shape[0])

