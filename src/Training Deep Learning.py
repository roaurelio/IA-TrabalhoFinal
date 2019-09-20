#!/usr/bin/env python
# coding: utf-8

# In[35]:


import pickle
import numpy as np
import os
import cv2
import random
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten


# In[21]:


data_dir = r"C:\Users\Rosana\Documents\DataSets\datas\train"
categories = ["fundusImage","other"]


# In[36]:


img_size = 256

def create_training_data():
    for category in categories:
        path = os.path.join(data_dir, category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (img_size, img_size))
                training_data.append([new_array, class_num])
            except Exception as e:
                print(e)
                pass

create_training_data() 


# In[38]:


random.shuffle(training_data)
x_training = []
y_training = []

for features, label in training_data:
    x_training.append(features)
    y_training.append(label)
    
x_training = np.array(x_training).reshape(-1, img_size, img_size, 1)


# In[44]:


x_training = x_training/255.0

model = Sequential()
model.add(Flatten(input_shape=x_training.shape[1:]))
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(128, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(2, activation="softmax"))

#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
historico = model.fit(x_training, y_training, batch_size=16, validation_split=0.1, epochs=10, shuffle=True)
model.save('trainingDL.model')

