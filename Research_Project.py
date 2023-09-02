#!/usr/bin/env python
# coding: utf-8

# <h1><b> Welcome to my Reseach Project Code! </b></h1>
# 
# The topic of my research is to identify from Xray images if a person is normal, has pneumonia or covid. In this project, I have prepared the dataset, then used the Sequential Model of CNN to classify images.

# I have used the X-Ray images dataset available on Kaggle for my research. It contains the X-Ray images of normal people, of people having pneumonia and of people having Covid-19. 

# <h2><b> Importing Packages</h2>

# To download dataset directly from Kaggle if using Google Colab. This will load the dataset in the folder in Colab.
# 
#     -> import os
# 
#     -> os.environ['KAGGLE_USERNAME'] = "mrana14"
# 
#     ->  os.environ['KAGGLE_KEY'] = "0f0101b72d3483ba0b2684a0fc2b2a99"
# 
#     ->  !kaggle datasets download tawsifurrahman/covid19-radiography-database
# 
#     -> !unzip covid19-radiography-database.zip
# 
# 
# 

# In[1]:


import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from numpy.core.multiarray import asarray
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


# During my Research I performed two sets of Image-Classification. <br>
# 1. First iteration involved only classification of Normal and Covid-19 X-Ray Images.<br>
# 2. Second iteration involved classification of all three types - Normal Covid-19 and Viral Penumonia as well.
# <br>
# 

# I will show the implementation of both.

# <h2> Iteration 1 - Normal and Covid-19 Image Classification Using CNN 

# The following two command gives count of images in Normal and COVID folder of the dataset

# In[2]:


len(os.listdir('COVID-19_Radiography_Dataset/Normal/images'))


# In[3]:


len(os.listdir('COVID-19_Radiography_Dataset/COVID/images'))


# The next cell uses opencv command cv2.imread to read an image. This is to just visualize and understand the data we have.

# In[4]:


img = cv2.imread('COVID-19_Radiography_Dataset/Normal/images/Normal-10.png')


# The next cell uses matplotlib.-pyplot library to display the image

# In[5]:


plt.imshow(img)


# This command shows the shape of the image which is 299pixels *299 pixels with color depth of 3. It has also been verified for all images that they are of the same shape. If they would not have been of the same dimensions, resizing them to the same dimension would have been necessary.

# In[6]:


img.shape


# The below code is written to understand how I passed the urls to the loadImages function for all image categories.
# <br>'links' is used to store all the names of the files(images in our case) in that directory / folder.
# <br> 'path' will store the complete path for the images.

# In[7]:


links = os.listdir('COVID-19_Radiography_Dataset/COVID/images')


# In[8]:


path = "COVID-19_Radiography_Dataset/COVID/images/" + links[0]


# In[9]:


path


# <h2>Writing the the <b>loadImages </b> function </h2>
# </br>
# Now, to prepare the dataset for training, it shoould contain all normal and covid images. For this we need to load all the images. I have loaded each category of images separately as well as have stored the labels / targets values separately.
# <br> <br>
# images[ ] is a list to all store the images and labels[ ] stores the labels for those images - whether it is 0 or 1 where:
# <br>
# 0 -> Normal <br>
# 1 -> Covid <br>

# 1. We will pass the path, links/urls and target of each image to load them into the list. <br>
# 2. The for loop ensures that the images are read one-by-one. And the complete image path will be stored in the variable image_path. <br>
# 3. The image is now read using opencv by using the cv2.imread() function.<br> 
# 4. I have then performed the standardization of images by dividing it by 255.<br> 
# 5. The images are then stored in the images list and the targets are stored in labels. <br>
# 6. Here, we have also typecasted the lists to arrays

# In[10]:


def loadImages(path, links, target):
    images = []
    labels = []
    for i in range(len(links)):
        image_path = path + links[i]
        img = cv2.imread(image_path)
        img = img / 255.0
        img = cv2.resize(img, (100,100)) #if images were of different sizes, do this piece of code
        images.append(img)
        labels.append(target)
    images = np.asarray(images)
    return images, labels


# The normal_path contains the path for normal images and the normal_links contians the image urls. These two values are then passed to the load_images function which then returns the images and the targets. The step is repeated for covid images as well.

# In[11]:


normal_path = "COVID-19_Radiography_Dataset/Normal/images/"
normal_links = os.listdir(normal_path)
normal_images, normal_targets = loadImages(normal_path, normal_links, 0)


# In[12]:


covid_path = "COVID-19_Radiography_Dataset/COVID/images/"
covid_links = os.listdir('COVID-19_Radiography_Dataset/COVID/images')
covid_images, covid_targets = loadImages(covid_path, covid_links, 1)


# In[13]:


#covid_images = np.asarray(covid_images)


# In[14]:


#len(covid_images)


# In[15]:


#covid_images.shape


# In[16]:


#normal_images = np.asarray(normal_images)


# In[17]:


#normal_images.shape


# np.r_ is a numpy function used to stack data row-wise. Since we have all the data stored in variables, we can simply pass those variables and store in in the data variable for images and in the targets variable for all targets.

# In[18]:


data = np.r_[covid_images, normal_images]


# In[19]:


data.shape


# In[20]:


targets = np.r_[covid_targets, normal_targets]


# In[21]:


targets.shape


# I have now split the data into training and testing and used the 75-25 split. Since the data will be shuffled anyways, hence stacking them is not a problem.

# In[22]:


x_train, x_test, y_train, y_test = train_test_split(data, targets, test_size=0.25)


# Now, for model building, I have already imported tensorflow as well as the Sequenctial model from keras.<br>
# In the Sequential model, the layers are simply passed one-by-one and are stacked one-by-one.

# Now I created a Sequential object and stored it in a model variable. And I will now passing the layers.<br>
# The first layer I passed is the Conv2D. In Conv2D, the paramters passed are - the number of filters, I have chosen 32. One can choose 16, 32, 64, etc. The next paramter that I have passed is th kernel_size of 3.<br>
# I have then passed the imnput_shape of our data which is 100*100*3 in our case. <br>
# Additionally, I have used the activation type as "relu" which is usually the recommended one. I have used relu because it reduces the negative value. <br>

# Now, my next layer is the MaxPool2D. In this by default, a pool_size of (2,2) will be considered.
# I have then added the Conv2D and MaxPool2D 2 more times. We do not need to pass the input_shape again.

# Now, I will add the Flatten() layer which will add the input layer which will basically contain the neurons. <br>
# Post this, I have added the hidden layer and have passed 512 neurons with activation type as 'relu'.
# <br>For the output layer, as number of outputs will be 2, in that case the activation type will be "softmax".
# In my case, I wanted only one output and wanted to use the sigmoid activation type.
# 
# I have additionally added one more dense layer to add more neurons to the input.

# In[23]:


model = Sequential([
    Conv2D(32, 3, input_shape=(100,100,3), activation='relu'),
    MaxPooling2D(),
    Conv2D(16, 3, activation='relu'),
    MaxPooling2D(),
    Conv2D(16, 3, activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(256, activation='relu'),
    Dense(1, activation='sigmoid')
])


# In[24]:


model.summary()


# Now, it is important to compile the model and for this, I have added the optimizer as 'adam' - one of the commonly used optimizers in Deep Learning.
# 
# The loss function here I have used is Binary Crossentropy as I have 2 outputs. 
# 
# The metrics I wanted to focus on was accuracy, hence I have passed that.

# In[25]:


model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(),metrics=['accuracy'])


# Now, to train the model, I used the model.fit() method and passed the training data. I passed the default batch size of 32 and an epoch value of 5 to train 5 epochs, to begin with. I have passed the test data for now in validation_data.
# 
# 
# As we see, the accuracy has improved with epochs, as well as in comparison to the training accuracy, the test accuracy looks good.

# In[26]:


model.fit(x_train, y_train,batch_size=32,epochs=5,validation_data=(x_test, y_test))


# I have also tried to visualize the accuracy for training data.

# In[27]:


plt.plot(model.history.history['accuracy'], label = 'Training Accuracy')
plt.plot(model.history.history['val_accuracy'],label = 'Testing accuracy')
plt.legend()
plt.show()


# I have also plotted for the test data and hence we can see the loss.

# In[29]:


plt.plot(model.history.history['loss'], label = 'train loss')
plt.plot(model.history.history['val_loss'],label = 'test_loss')
plt.legend()
plt.show()


# <br>

# <h2> Iteration 2 - Normal, Covid-19 and Viral Pneumonia Image Classification Using CNN 

# The same steps have been repeated as done above, but this time, I have also considered the Pneumonia images.

# The following three command gives count of images in Normal, COVID and Pnemonia folder of the dataset

# In[30]:


len(os.listdir('COVID-19_Radiography_Dataset/Normal/images'))


# In[31]:


len(os.listdir('COVID-19_Radiography_Dataset/COVID/images'))


# In[32]:


len(os.listdir('COVID-19_Radiography_Dataset/Viral Pneumonia/images'))


# <h2>Writing the the <b>loadImages </b> function </h2>
# </br>
# Now, to prepare the dataset for training, it shoould contain all normal and covid images. For this we need to load all the images. I have loaded each category of images separately as well as have stored the labels / targets values separately.
# <br> <br>
# images[ ] is a list to all store the images and labels[ ] stores the labels for those images - whether it is 0 or 1 where:
# <br>
# 0 -> Normal <br>
# 1 -> Covid <br>

# In[33]:


def loadImages(path, links, target):
    images = []
    labels = []
    for i in range(len(links)):
        image_path = path + links[i]
        img = cv2.imread(image_path)
        img = img / 255.0
        img = cv2.resize(img, (100,100)) #if images were of different sizes, do this piece of code
        images.append(img)
        labels.append(target)
    images = np.asarray(images)
    return images, labels


# In[34]:


normal_path = "COVID-19_Radiography_Dataset/Normal/images/"
normal_links = os.listdir(normal_path)
normal_images, normal_targets = loadImages(normal_path, normal_links, 0)


# In[35]:


covid_path = "COVID-19_Radiography_Dataset/COVID/images/"
covid_links = os.listdir('COVID-19_Radiography_Dataset/COVID/images')
covid_images, covid_targets = loadImages(covid_path, covid_links, 1)


# In[36]:


pneumonia_path = "COVID-19_Radiography_Dataset/Viral Pneumonia/images/"
pneumonia_links = os.listdir('COVID-19_Radiography_Dataset/Viral Pneumonia/images')
pneumonia_images, pneumonia_targets = loadImages(pneumonia_path, pneumonia_links, 2)


# In[37]:


covid_images.shape


# In[38]:


normal_images.shape


# In[39]:


pneumonia_images.shape


# In[40]:


data = np.r_[covid_images, normal_images, pneumonia_images]


# In[41]:


data.shape


# In[42]:


targets = np.r_[covid_targets, normal_targets,pneumonia_targets]


# In[43]:


targets.shape


# In[44]:


x_train, x_test, y_train, y_test = train_test_split(data, targets, test_size=0.25)


# In[45]:


model = Sequential([
    Conv2D(32, 3, input_shape=(100,100,3), activation='relu'),
    MaxPooling2D(),
    Conv2D(16, 3, activation='relu'),
    MaxPooling2D(),
    Conv2D(16, 3, activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(256, activation='relu'),
    Dense(1, activation='sigmoid')
])


# In[46]:


model.summary()


# In[47]:


model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(),metrics=['accuracy'])


# In[57]:


model.fit(x_train, y_train,batch_size=32,epochs=7,validation_data=(x_test, y_test))


# In[58]:


plt.plot(model.history.history['accuracy'], label = 'train accuracy')
plt.plot(model.history.history['val_accuracy'],label = 'test accuracy')
plt.legend()
plt.show()


# In[59]:


plt.plot(model.history.history['loss'], label = 'train loss')
plt.plot(model.history.history['val_loss'],label = 'test_loss')
plt.legend()
plt.show()


# In[ ]:




