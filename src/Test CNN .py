#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import tensorflow as tf
import random


# In[ ]:


CATEGORIES = ["fundusImage", "other"]

def prepare(filepath):
    IMG_SIZE = 70  # 50 in txt-based
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


model = tf.keras.models.load_model("trainingCNN.model")


# In[ ]:


img_size = 256
test_data = []

def create_test_data():
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

create_test_data()   
random.shuffle(test_data)

x_test = []
y_test = []

for features, label in test_data:
    x_test.append(features)
    y_test.append(label)
    
x_test = np.array(x_test).reshape(-1, img_size, img_size, 1)


# In[ ]:


prediction = model.predict(x_test)
perda, acuracia = modelo.evaluate(x_test, y_test)

#print(prediction)  # will be a list in a list.
#print(CATEGORIES[int(prediction[0][0])])

