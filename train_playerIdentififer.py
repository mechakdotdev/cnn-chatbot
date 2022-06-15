# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 15:31:47 2021

@author: mecha
"""

# submission 2 libraries for image recognition with a CNN using keras + tensorflow on an NBA players dataset

# BUILDING THE IMAGE DATASET

import os

# loading training datasets
train_lebron_dir = os.path.join('train/lebron')
train_luka_dir = os.path.join('train/luka')

# loading validation datasets
valid_lebron_dir = os.path.join('valid/lebron')
valid_luka_dir = os.path.join('valid/luka')


train_lebron_names = os.listdir(train_lebron_dir)
#print(train_lebron_names[:10])
train_luka_names = os.listdir(train_luka_dir)
#print(train_luka_names[:10])

valid_lebron_names = os.listdir(valid_lebron_dir)
#print(train_lebron_names[:10])
valid_luka_names = os.listdir(valid_luka_dir)
#print(train_luka_names[:10])

#print('total training lebron images:', len(os.listdir(train_lebron_dir)))
#print('total training luka images:', len(os.listdir(train_luka_dir)))

#print('total validation lebron images:', len(os.listdir(valid_lebron_dir)))
#print('total validation luka images:', len(os.listdir(valid_luka_dir)))

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# parameters for graph in 4x4
nRows = 4
nColumns = 4

# index for iterating over imgs
pictureIndex = 0

fig = plt.gcf()
fig.set_size_inches(nColumns * 4, nRows * 4)

pictureIndex += 8
next_lebron_pic = [os.path.join(train_lebron_dir, fname) 
                for fname in train_lebron_names[pictureIndex-8:pictureIndex]]

next_luka_pic = [os.path.join(train_luka_dir, fname) 
                for fname in train_luka_names[pictureIndex-8:pictureIndex]]

for i, img_path in enumerate(next_lebron_pic + next_luka_pic):
  # Set up subplot; subplot indices start at 1
  sp = plt.subplot(nRows, nColumns, i + 1)
  sp.axis('Off') # Don't show axes (or gridlines)

  img = mpimg.imread(img_path)
  plt.imshow(img)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1/255)
validation_datagen = ImageDataGenerator(rescale=1/255)

# Flow training images in batches of 120 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        'train/',  # This is the source directory for training images
        classes = ['lebron', 'luka'],
        target_size=(200, 200),  # All images will be resized to 200x200
        batch_size=120,
        # Use binary labels
        class_mode='binary')

# Flow validation images in batches of 19 using valid_datagen generator
validation_generator = validation_datagen.flow_from_directory(
        'valid/',  # This is the source directory for training images
        classes = ['lebron', 'luka'],
        target_size=(200, 200),  # All images will be resized to 200x200
        batch_size=19,
        # Use binary labels
        class_mode='binary',
        shuffle=False)

# BUILDING THE MODEL

import tensorflow as tf
import numpy as np
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score


# flatten layer
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape = (200,200,3)), 
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)])

# model.summary()

model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss = 'binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_generator,
      steps_per_epoch=8,  
      epochs=15,
      verbose=1,
      validation_data = validation_generator,
      validation_steps=8)

from PIL import Image
path = input("Enter file path here (e.g. train/lebron/Lebron-Training-1)")
img = Image.open(path)

img = image.load_img(path, target_size=(200, 200))
x = image.img_to_array(img)
plt.imshow(x/255.)
x = np.expand_dims(x, axis=0)
images = np.vstack([x])
classes = model.predict(images, batch_size=10)
print(classes[0])
if classes[0]<0.5:
    print("The selected image is most liekly a picture of Lebron James")
else:
    print("The selected image is most likely a picture of Luka Doncic")

