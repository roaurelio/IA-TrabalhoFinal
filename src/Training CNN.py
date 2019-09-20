#!/usr/bin/env python
# coding: utf-8

# In[34]:


import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization


# In[35]:


data_dir = r"C:\Users\Rosana\Documents\DataSets\datas\train"
categories = ["fundusImage","other"]


# In[36]:


img_size = 256
training_data = []

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
                pass

create_training_data()                    


# In[37]:


import random
random.shuffle(training_data)

#print(training_data[0:])
#plt.imshow(training_data[1], cmap='gray')


# In[44]:


x_training = []
y_training = []

for features, label in training_data:
    x_training.append(features)
    y_training.append(label)
    
x_training = np.array(x_training).reshape(-1, img_size, img_size, 1)


# In[6]:


import pickle

pickle_out = open("x.pickle","wb")
pickle.dump(x_training, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y_training, pickle_out)
pickle_out.close()


# In[32]:


pickle_in = open("x.pickle","rb")
w = pickle.load(pickle_in)


# In[39]:


x_training = x_training/255.0

model = Sequential()

#bloco 1
model.add(Conv2D(8, (3,3), activation="relu", input_shape = x_training.shape[1:]))
model.add(Conv2D(16, (3,3), activation="relu"))
model.add(Conv2D(32, (3,3), activation="relu"))
model.add(Conv2D(64, (3,3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

#bloco 2
model.add(Conv2D(32, (3,3), activation="relu"))
model.add(Conv2D(64, (3,3), activation="relu"))
model.add(Conv2D(64, (3,3), activation="relu"))
model.add(Conv2D(128, (3,3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

#bloco 3
model.add(Conv2D(64, (3,3), activation="relu"))
model.add(Conv2D(128, (3,3), activation="relu"))
model.add(Conv2D(128, (3,3), activation="relu"))
model.add(Conv2D(64, (3,3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

#bloco 4
model.add(Conv2D(64, (3,3), activation="relu"))
model.add(Conv2D(32, (3,3), activation="relu"))
model.add(Conv2D(16, (3,3), activation="relu"))
model.add(Conv2D(8, (3,3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Flatten())

model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_training, y_training, batch_size=16, validation_split=0.1, epochs=1, shuffle=True)
model.save('trainingCNN.model')


# In[ ]:




