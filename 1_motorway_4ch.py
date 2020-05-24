#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 14:48:19 2020

@author: swilson
"""

import os
import numpy as np
import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
import keras



#################################################
#   Setting GPU Usage
#################################################
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']= '0,1'
#os.environ['CUDA_VISIBLE_DEVICES']= ''


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


##############################################################
#      Data Preparation - Import datasets, set image sie
##############################################################
img_width, img_height = 256, 256

train_data_dir = '/workspaces/maprover--cnn-models/Data_for_CNN/highway_motorway/TRAIN/'
validation_data_dir = '/workspaces/maprover--cnn-models/Data_for_CNN/highway_motorway/TEST/'

train_pos_dir = 'train_motorway'
train_neg_dir = 'train_not_motorway'

test_pos_dir = 'test_motorway'
test_neg_dir = 'test_not_motorway'


num_train_pos = len([file for file in os.listdir(os.path.join(train_data_dir, train_pos_dir))])
num_train_others = len([file for file in os.listdir(os.path.join(train_data_dir, train_neg_dir))])

num_test_pos = len([file for file in os.listdir(os.path.join(validation_data_dir, test_pos_dir))])
num_test_neg = len([file for file in os.listdir(os.path.join(validation_data_dir, test_neg_dir))])

nb_train_samples = num_train_pos + num_train_others
nb_validation_samples = num_test_pos + num_test_neg

epochs = 50
batch_size = 16


    ###---  configure the shape of dataset
if K.image_data_format() == 'channels_first':
    input_shape = (4, img_width, img_height)
else:
    input_shape = (img_width, img_height, 4)
    
#######################################
#      CNN Model
#######################################
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))  #--32
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Flatten())
#model.add(Dense(64))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(2))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', 
              optimizer='rmsprop',
             # optimizer='adam',
              metrics=['accuracy'])
    

#######################################
#       Traing & Test Settings
#######################################

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                    color_mode='rgba',
                                                    target_size=(img_width, img_height), 
                                                    batch_size=batch_size, class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(validation_data_dir,
                                                        color_mode='rgba',
                                                        target_size=(img_width, img_height),
                                                        batch_size=batch_size,
                                                        class_mode='categorical')

# Save the model according to the conditions  
checkpoint = ModelCheckpoint("motorway_4ch.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')


    ###--- Model fitting and Validation
model.fit_generator(train_generator,
                    steps_per_epoch=nb_train_samples // batch_size,
                    epochs=epochs,
                    validation_data=validation_generator,
                    validation_steps=nb_validation_samples // batch_size,
                    callbacks = [checkpoint, early])




import tensorflowjs as tfjs

# specify where to save keras/tensorflow model
tfjs.converters.save_keras_model(model, './tfjs_model')



