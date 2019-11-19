# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 14:23:26 2019

@author: Sefa3
"""

#______________________________________________________________________________

#-----------------------| Part 1 : Building the CNN |--------------------------
"""
since our dataset no longer has the structure where the rows are the 
observations and the columns are the independent variables and the dependent 
variable next to each other, then we cannot add explicitly the dependent
variable in our dataset because it wouldn't make much sense to add this
dependent variable columns along the 3D arrays representing the images.
But we need to train a machinery model(we always need the dependent variable)
to have the real results that are required to understand the correlations
between the information contained in the independent variables, and
the real result contained in the dependent variable. Therefore
To extract the info of this dependent variable we have several solutions...
"""
#------------------------------------------------------------------------------
# importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#------------------------------------------------------------------------------
# to initialize our neural network
from keras.models import Sequential
#------------------------------------------------------------------------------
# for step one : convolution step, convolutional layers, to dealing images.
from keras.layers import Convolution2D
#------------------------------------------------------------------------------
# for step two : pooling step, this will add our pooling layers
from keras.layers import MaxPooling2D
#------------------------------------------------------------------------------
# for step three : flattening, converting all the pooled feature maps
# that we created through concolution and maxpooling into
# this large feature vector --> becoming the input our fully connected layers.
from keras.layers import Flatten
#------------------------------------------------------------------------------
# this is the package we use to add the fully connected layers
from keras.layers import Dense
#------------------------------------------------------------------------------
"""
so basically each package corresponds to one step of the construction
of the CNN.
"""
#------------------------------------------------------------------------------
# Initializing the CNN
classifier = Sequential()
#------------------------------------------------------------------------------
"""
we start with 32 feature detectors in the first convolutional layer,
and then we add other convolutional layers with more detectors like 64
and then 128 and then 256 maybe...
"""
#------------------------------------------------------------------------------
# Step 1 - Convolution

# classifier.add(Convolution2D(32, 3, 3, 
#                input_shape = (64, 64, 3))
#                ) means;
# we create 32 feature detectors of [3, 3] dimensions
# 32     : the number of filters you want to use
# 3      : the number of rows of each filter
# 3      : the number of columns of our feature detector/filter.
# input_shape = (64, 64, 3) means
# 64, 64 : the dimensions of our 2D arrays
# 3      : the number of channels
#-----------------------------------------------------------
classifier.add(Convolution2D(32, 
                             3, 
                             3, 
                             input_shape = (64, 64, 3),
                             activation = 'relu'
                            )
              )
#------------------------------------------------------------------------------
"""
to reduce the size of our feature maps
and therefore to reduce the number of nodes.

by taking of these 2x2 sub tables of the feature map,
we are in some way keeping the information because
we are keeping track of the part of the image, that
contained the high numbers corresponding to where
the feature detectors detected some specific features
in the input image.

so we don't lose the performance of the model
but at the same time, we managed to reduce
the time complexity and we make it less
compute-intensive.
"""
#------------------------------------------------------------------------------
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2),
                            )
               )
#------------------------------------------------------------------------------
"""
by creating our Feature Maps, we extracted the spatial
structure informations.

the spacial structure of our images, these high numbers
in the feature maps are associated to a specific feature
in the input image.

and since then, we apply the max pooling step, we keep
these high numbers because we take the max.
so, the flattening step just consists of putting all the
numbers in the cells of the feature maps into one, same,
single vector.

since these high numbers represent the spatial structure
of the input image and are associated to one specific
feature of this spatial structure, we keep this spatial
structure information.
-------------------------------------------------------------------------------
if we directly flatten the input image pixels into this
huge, single one dimentional vector then each node of this
huge vector will represent one pixel of the image,
independently of the pixels that are around it.

we only get informations of the pixel itself, and we don't
get informations of how this pixel is spatially connected
to other pixels around it.

so basically, we don't get any information of the spatial
structure around this pixel.
-------------------------------------------------------------------------------
therefore using convolution and maxpooling, we keep the
spatial structure information of the input image.
"""
#------------------------------------------------------------------------------
# Step 3 - Flattening
classifier.add(Flatten())
#------------------------------------------------------------------------------
# Step 4.a - Full connection
classifier.add(Dense(128, # output dimension
                     activation = 'relu'
                     )
               )
#------------------------------------------------------------------------------
"""
we are using sigmoid function, because
we have a binary outcome probability : cat or dog.
if we had an outcome with more than two categories,
we would need to use the softmax activation function.
but we have a binary outcome : therefore we are using
the sigmoid activation function.
"""
#------------------------------------------------------------------------------
# Step 4.b - Full connection

# output layer
classifier.add(Dense(1, # output : predicted probability.
                     activation = 'sigmoid'
                     )
               )
#------------------------------------------------------------------------------
# Compiling the CNN

# if we had more than two outcomes(cats, dogs, birds,...)
# we would need to choose categorical_crossentropy
classifier.compile(optimizer = 'adam',
                   loss      = 'binary_crossentropy',
                   metrics   = ['accuracy'])
#______________________________________________________________________________

#------------------| Part 2 : Fitting the CNN to the images |------------------

from keras.preprocessing.image import ImageDataGenerator
#------------------------------------------------------------------------------
"""
to increase the accuracy, we need to make a deeper deep learning model
we have 2 options;
1) add another convolutional layer
2) add another fully connected layer.
"""
#------------------------------------------------------------------------------
# Initializing the CNN
classifier = Sequential()
#------------------------------------------------------------------------------
# adding first convolutional layer
classifier.add(Convolution2D(32, 3, 3, 
                             input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
#------------------------------------------------------------------------------
# adding second convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
#------------------------------------------------------------------------------
# adding third convolutional layer
classifier.add(Convolution2D(64, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
#------------------------------------------------------------------------------
# adding fourth convolutional layer
classifier.add(Convolution2D(128, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
#------------------------------------------------------------------------------
classifier.add(Flatten())

classifier.add(Dense(128,activation = 'relu'))

classifier.add(Dense(1, activation = 'sigmoid'))

classifier.compile(optimizer = 'adam',
                   loss      = 'binary_crossentropy',
                   metrics   = ['accuracy'])
#------------------------------------------------------------------------------
train_datagen = ImageDataGenerator(rescale         = 1./255,
                                   shear_range     = 0.2,
                                   zoom_range      = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(directory   = 'dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size  = 32,
                                                 class_mode  = 'binary')

test_set = test_datagen.flow_from_directory(directory   = 'dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size  = 32,
                                            class_mode  = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch  = 8000,
                         epochs           = 6,
                         validation_data  = test_set,
                         validation_steps = 2000
                         )
#______________________________________________________________________________

#----------------------| Part 3 : Making new predictions |---------------------

import numpy as np

from keras.preprocessing import image

training_set.class_indices

test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', 
                            target_size = (64, 64))

test_image = image.img_to_array(test_image)

test_image = np.expand_dims(test_image, axis = 0)

result     = classifier.predict(test_image)

if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

#------------------------------------------------------------------------------

test_image = image.load_img('dataset/single_prediction/cat_or_dog_2.jpg', 
                            target_size = (64, 64))

test_image = image.img_to_array(test_image)

test_image = np.expand_dims(test_image, axis = 0)

result     = classifier.predict(test_image)

if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

#______________________________________________________________________________







































