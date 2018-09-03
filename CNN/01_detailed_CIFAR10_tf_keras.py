# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 16:17:38 2018

@author: ashutosh

CIFAR10 in Tensorflow

"""
#Set Path
import os
path = "D:/DeepLearning/Tensorflow/Codes/"
os.chdir(path)

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras

from keras import backend as K

print("We are using Tensorflow version : ",tf.__version__)
print("We are using Keras version : ",keras.__version__)


##------------- Load Datasets -------------------------

from keras.datasets import cifar10
(x_train, y_train), (x_test,y_test) = cifar10.load_data()

print("Train samples:", x_train.shape, y_train.shape)
print("Test samples:", x_test.shape, y_test.shape)


## Vizualization
NUM_CLASSES = 10
cifar10_classes = ["airplane", "automobile", "bird", "cat", "deer", 
                   "dog", "frog", "horse", "ship", "truck"]

# show random images from train
cols = 8
rows = 2
fig = plt.figure(figsize=(2 * cols - 1, 2.5 * rows - 1))
for i in range(cols):
    for j in range(rows):
        random_index = np.random.randint(0, len(y_train))
        ax = fig.add_subplot(rows, cols, i * rows + j + 1)
        ax.grid('off')
        ax.axis('off')
        ax.imshow(x_train[random_index, :])
        ax.set_title(cifar10_classes[y_train[random_index, 0]])
plt.show()


## Prepare Data : X
x_train2 = x_train.astype('float32')/255 - 0.5
x_test2 = x_test.astype('float32')/255 - 0.5

## Prepare Data : Y
y_train2 = keras.utils.to_categorical(y_train)
y_test2 = keras.utils.to_categorical(y_test)


##------------- Load Datasets Complete -------------------------





##------------- Define CNN Architecture -------------------------

# import necessary building blocks
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from keras.layers.advanced_activations import LeakyReLU


def make_model():
    
    model = Sequential()
    
    model.add(Conv2D(16,(3,3), padding='same', input_shape=x_train2.shape[1:]))
    model.add(LeakyReLU(0.1))
    model.add(Conv2D(32,(3,3), padding='same'))
    model.add(LeakyReLU(0.1))
    
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(32,(3,3), padding='same'))
    model.add(LeakyReLU(0.1))
    model.add(Conv2D(64,(3,3), padding='same'))
    model.add(LeakyReLU(0.1))
    
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(256))
    model.add(LeakyReLU(0.1))
    
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    
    return model
    
model = make_model()


INIT_LR = 5e-3  # initial learning rate
BATCH_SIZE = 32
EPOCHS = 10

#s = reset_tf_session()  # clear default graph
# don't call K.set_learning_phase() !!! (otherwise will enable dropout in train/test simultaneously)
#model = make_model()  # define our model

# prepare model for fitting (loss, optimizer, etc)
model.compile(
    loss='categorical_crossentropy',  # we train 10-way classification
    optimizer=keras.optimizers.adamax(lr=INIT_LR),  # for SGD
    metrics=['accuracy']  # report accuracy during training
)

# scheduler of learning rate (decay with epochs)
def lr_scheduler(epoch):
    return INIT_LR * 0.9 ** epoch

# callback for printing of actual learning rate used by optimizer
class LrHistory(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs={}):
        print("Learning rate:", K.get_value(model.optimizer.lr))
        
model_filename = 'cifar.{0:03d}.hdf5'
last_finished_epoch = None


#### uncomment below to continue training from model checkpoint
#### fill `last_finished_epoch` with your latest finished epoch
# from keras.models import load_model
# s = reset_tf_session()
# last_finished_epoch = 7
# model = load_model(model_filename.format(last_finished_epoch))

# fit model
import keras.utils
model.fit(
    x_train2, y_train2,  # prepared data
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    #callbacks=[keras.callbacks.LearningRateScheduler(lr_scheduler), 
     #          LrHistory(), 
     #          keras.utils.TqdmProgressCallback(),
     #          keras.utils.ModelSaveCallback(model_filename)],
    validation_data=(x_test2, y_test2),
    shuffle=True,
    verbose=1,
    initial_epoch=last_finished_epoch or 0
)


#Model Evaluation
# make test predictions
y_pred_test = model.predict_proba(x_test2)
y_pred_test_classes = np.argmax(y_pred_test, axis=1)
y_pred_test_max_probas = np.max(y_pred_test, axis=1)

# confusion matrix and accuracy
from sklearn.metrics import confusion_matrix, accuracy_score
plt.figure(figsize=(7, 6))
plt.title('Confusion matrix', fontsize=16)
plt.imshow(confusion_matrix(y_test, y_pred_test_classes))
plt.xticks(np.arange(10), cifar10_classes, rotation=45, fontsize=12)
plt.yticks(np.arange(10), cifar10_classes, fontsize=12)
plt.colorbar()
plt.show()
print("Test accuracy:", accuracy_score(y_test, y_pred_test_classes))