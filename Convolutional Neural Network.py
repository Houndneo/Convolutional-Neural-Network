#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as ts
import matplotlib.pyplot as plt
from tensorflow.keras import datasets,layers,models


# In[3]:


import ssl
ssl._create_default_https_context = ssl._create_unverified_context


# In[4]:


(train_images,train_labels),(test_images,test_labels)=datasets.cifar10.load_data()


# In[5]:


train_images.shape


# In[6]:


train_images,test_images=train_images/255.0,test_images/255.0
class_names =["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]


# In[7]:


image_index=1
plt.imshow(train_images[image_index],cmap=plt.cm.binary)
plt.xlabel(class_names[train_labels[image_index][0]])
plt.show()

