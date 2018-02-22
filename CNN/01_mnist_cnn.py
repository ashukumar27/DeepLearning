#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 15:12:10 2018

@author: ashutosh

MNIST Classification using CNN - Basis NN and Convolution NN
"""

#Import Libraries
from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

# Parameters for the model
batch_size=128
#output classes - 10 in this case
num_classes=10
epochs =12

#input image dimensions : 28x28 images (predefined)
img_rows, img_cols = 28,28



##################################################################
#                   Simple Neural Network
###################################################################

# Data Import, and splitting between test and train sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Reshaping the data in 60000 rows and 784 columns (train) and 10000x784 (test)
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
#Converting to Float
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
#Normalizing values between 0 and 1
x_train /= 255
x_test /= 255

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices 
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

### Build Model
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

#Print Model Summary
model.summary()

#Compile Model
model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

#Run Model - save metrics after each epoch in history
history = model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1,
                    validation_data=(x_test, y_test))

#Score model against test data
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


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



##################################################################
#                  Convolution Neural Network
###################################################################


# Data Import, and splitting between test and train sets
(x_train, y_train),(x_test,y_test) = mnist.load_data()

#Reshape Data
# X_train.shape = (60000,28,28) - to be converted to (60000,28,28,1)
# Check the format: channel first or last
if K.image_data_format() =='channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)


## Normalize the inputs and convert to float
x_train = x_train.astype('float64')
x_test = x_test.astype('float64')
x_train/=255
x_test/=255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples') 


## Convert class vectors to binary class metrics
y_train = keras.utils.to_categorical(y_train,num_classes)
y_test = keras.utils.to_categorical(y_test,num_classes)



# Convolution Neural Network
model= Sequential()
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='softmax'))

#Compile model
model.compile(loss = keras.losses.categorical_crossentropy, optimizer = keras.optimizers.adam(), metrics = ['accuracy'])

#Fit Model
model.fit(x_train,y_train, batch_size=batch_size, epochs = epochs,
          verbose=1, validation_data = (x_test, y_test))

#Score Model
score = model.evaluate(x_test, y_test, verbose=0)

print("Test Loss:", score[0])
print("Test Accuracy", score[1])
