#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 13:31:44 2019

@author: ashutosh.k

{ 25 Projects } Fashion MNIST

Fashion MNIST Classification using Keras

"""


from __future__ import print_function
import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K



## Data Import
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

## Check shapes and vizualize
print("X_train Shape",x_train.shape)
print("Y_train Shape",y_train.shape)
print("X_test Shape",x_test.shape)
print("Y_test Shape",y_test.shape)

## 60000 train images, 28x28 image size

set(y_train)
# 10 Classes



#Model Parameters
BATCH_SIZE = 128
NUM_CLASSES = 10
EPOCHS = 12

###   SIMPLE NEURAL NETWORK - No Convolution   ####

#Reshape input images
# Train: 60000x784 ; Test: 10000x784
x_train_ = x_train.reshape(x_train.shape[0],784)
x_test_ = x_test.reshape(x_test.shape[0],784)

## Convert to float and normalize
x_train_ = x_train_.astype('float32')
x_test_ = x_test_.astype('float32')

## Normalize
x_train_ /= 255
x_test_ /= 255

x_train_.shape

## Convert class vectors to binary one hot encoded
y_train_ = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test_ = keras.utils.to_categorical(y_test, NUM_CLASSES)

### Build Model
model = Sequential()
model.add(Dense(512, activation = 'relu', input_shape = (784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(NUM_CLASSES, activation = 'softmax'))

#print Model Summary
model.summary()

#Compile Model
model.compile(loss= 'categorical_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])

## Run Model and save metrices after each epoch in history
history = model.fit(x_train_, y_train_, batch_size = BATCH_SIZE, epochs = EPOCHS, 
                    verbose =1, validation_data = (x_test_, y_test_))

#Score Model against test data
score = model.evaluate(x_test_, y_test_, verbose=1)
print('Test loss:', score[0]) 
print('Test accuracy:', score[1])  #88.67


## Plot Accuracy and Loss stored in the 'history'
import matplotlib.pyplot as plt

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()




#############     Convolution Neural Network

#input image dimensions : 28x28 images (predefined)
img_rows, img_cols = 28,28

#Datasets: x_train, x_test, y_train, y_test

## Reshape Data - Channels Check
# X_train.shape = (60000,28,28) - to be converted to (60000,28,28,1)
# Check the format: channel first or last

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0],1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0],1, img_rows, img_cols)
    input_shape = (1,img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols,1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols,1)
    input_shape = (img_rows, img_cols,1)
    
    

## Normalize the inputs and convert to float
x_train = x_train.astype('float64')
x_test = x_test.astype('float64')
x_train/=255
x_test/=255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples') 

## Convert class vectors to binary class metrics
y_train = keras.utils.to_categorical(y_train,NUM_CLASSES)
y_test = keras.utils.to_categorical(y_test,NUM_CLASSES)


### CNN Model

model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape = input_shape))
model.add(Conv2D(64, kernel_size = (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(NUM_CLASSES, activation = 'softmax'))


model.summary()

#Compile model
model.compile(loss = keras.losses.categorical_crossentropy, optimizer = keras.optimizers.adam(), 
              metrics = ['accuracy'])

#Fit Model
history = model.fit(x_train,y_train, batch_size=BATCH_SIZE, epochs = 7,
          verbose=1, validation_data = (x_test, y_test))

#Score Model
score = model.evaluate(x_test, y_test, verbose=0)

print("Test Loss:", score[0])
print("Test Accuracy", score[1])


#Test Accuracy 0.9266 after 12 epochs

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
