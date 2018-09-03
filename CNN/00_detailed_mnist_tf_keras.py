# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 12:59:52 2018

@author: ashutosh

## MNIST in Tensorflow 
"""
#Set Path
import os
path = "D:/DeepLearning/Tensorflow/Codes/"
os.chdir(path)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

print("We are using Tensorflow version : ",tf.__version__)


## -------------------  Data Preparation  ------------------------
# Import MNIST dataset
from keras.datasets import mnist
(x_train, y_train), (x_test,y_test) = mnist.load_data()

print("Train Shape X: ", x_train.shape)
print("Test Shape X: ", x_test.shape)
print("Train Shape Y: ", y_train.shape)
print("Test Shape Y: ", y_test.shape)

#Vizualize 1 data and label

random_index = 3989
print("Y is : ",y_train[random_index])
plt.imshow(x_train[random_index], cmap="Greys")



# Reshaping Data (X) into Vecors of 784 dim
x_train = x_train.reshape([x_train.shape[0],x_train.shape[1]*x_train.shape[2]])
x_test = x_test.reshape([x_test.shape[0],x_test.shape[1]*x_test.shape[2]])

# Normalizing the data
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

#Reshape Data (Y) into one hot encoding
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

#------------ Data Preparation Complete -------------

## ---------- Tensorflow Implementation : Basic NN  --------------------


## TF Model parameters
learning_rate = 0.1
epochs = 500
batch_size = 512
display_step = 10


## Build Tensorflow Graph and define optimizing function
num_samples = x_train.shape[0]
num_features = x_train.shape[1]
num_classes = y_train.shape[1]

#Placeholders for X and Y
x = tf.placeholder(tf.float32, [None,num_features])
y = tf.placeholder(tf.float32, [None, num_classes])

# Model Parameters
W = tf.Variable(tf.zeros([num_features,num_classes]))
b = tf.Variable(tf.zeros([num_classes]))

#Define the model 
pred = tf.nn.softmax(tf.matmul(x,W)+b)

#Define cost
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices =1))

#OPtimize cost
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

## -----  Run Tensorflow Graph

init = tf.global_variables_initializer()


# Provinding Entire Dataset in 1 epoch
with tf.Session() as sess:
    #Run Initializer
    sess.run(init)
    
    for epoch in range(epochs):
        # Train without mini batches
        _,loss = sess.run([optimizer,cost], feed_dict= {x:x_train,y:y_train})
        print(loss)
    
    #Calculate accuracy on test data
    correct_prediction = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    
    print("Accuracy : ",accuracy.eval({x:x_test, y:y_test}))            
                   
    
    
#Training Cycle over mini batches
with tf.Session() as sess:
    #Run Initializer
    sess.run(init)
    
    num_batches = int(num_samples/batch_size)
    
    for epoch in range(epochs):
        #Loop Over MiniBatches
        batch_loss = []
        
        for batch_start in range(0,num_batches,batch_size):
            batch_x = x_train[batch_start: batch_start+batch_size,:]
            batch_y = y_train[batch_start: batch_start+batch_size,:]
            
            _,loss = sess.run([optimizer,cost], feed_dict = {x:batch_x, y:batch_y})
            batch_loss.append(loss)
        train_loss = np.mean(batch_loss)
        print(train_loss)
        
    #Calculate accuracy on test data
    correct_prediction = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    
    print("Accuracy : ",accuracy.eval({x:x_test, y:y_test}))            
        

## Add 1 more hidden Layer to the model

hidden1 = tf.layers.dense(x, 256, activation = tf.nn.relu)
hidden2 = tf.layers.dense(hidden1, 128, activation = tf.nn.relu)
pred = tf.layers.dense(hidden2, 10, activation = tf.nn.softmax)

#Define cost
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices =1))

#OPtimize cost
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

## -----  Run Tensorflow Graph
init = tf.global_variables_initializer()

#Training Cycle over mini batches
with tf.Session() as sess:
    #Run Initializer
    sess.run(init)
    
    num_batches = int(num_samples/batch_size)
    
    for epoch in range(epochs):
        #Loop Over MiniBatches
        batch_loss = []
        
        for batch_start in range(0,num_batches,batch_size):
            batch_x = x_train[batch_start: batch_start+batch_size,:]
            batch_y = y_train[batch_start: batch_start+batch_size,:]
            
            _,loss = sess.run([optimizer,cost], feed_dict = {x:batch_x, y:batch_y})
            batch_loss.append(loss)
        train_loss = np.mean(batch_loss)
        print(train_loss)
        
    #Calculate accuracy on test data
    correct_prediction = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    
    print("Accuracy : ",accuracy.eval({x:x_test, y:y_test}))            

## ---------- Tensorflow Implementation Complete  --------------------

## ---------- Tensorflow Implementation : Convolution Neural Network  --------------------

#Read Data Again since CNN takes an image as input
# Import MNIST dataset
from keras.datasets import mnist
(x_train, y_train), (x_test,y_test) = mnist.load_data()
    

def cnn_model_fn(x, labels, mode):
    """ Model function for CNN"""
    
    #input Layer
    input_layer = tf.reshape(x, [-1,28,28,1])
    
    #Convolution Layer 1
    conv1 = tf.layers.conv2d(inputs = input_layer,
                             filters = 32,
                             kernel_size = [5,5],
                             padding='same',
                             activation = tf.nn.relu)
    
    #Pooling Layer 1
    pool1 = tf.layers.max_pooling2d(inputs = conv1, pool_size=[2,2], strides=2)
    
    #Convolution Layer 2
    conv2 = tf.layers.conv2d(inputs = pool1,
                             filters = 64,
                             kernel_size = [5,5],
                             padding='same',
                             activation = tf.nn.relu)
    #Pooling Layer 2
    pool2 = tf.layers.max_pooling2d(inputs = conv2, pool_size=[2,2], strides=2)
    
    #Dense Layer
    pool2_flat = tf.reshape(pool2,[-1, 7*7*64])
    dense = tf.layers.dense(inputs = pool2_flat, units = 1024, activation = tf.nn.relu)
    
    #Dropout for regularization
    dropout = tf.layers.dropout(inputs = dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    
    #Logits layer
    logits = tf.layers.dense(inputs = dropout, units = num_classes)
    
    predictions = {
            #Generate predictions for predict & eval
            "classes": tf.argmax(logits, axis=1)
            # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
            # `logging_hook`.
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
            }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    