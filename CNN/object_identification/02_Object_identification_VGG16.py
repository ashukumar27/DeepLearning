#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 17:16:26 2018

@author: ashutosh

Image Identification using weights learnt from VGG16 network
"""

## Import VGG16 Model, save it locally (will take some time, approx 500MB file)
from keras.applications.vgg16 import VGG16
model = VGG16()

#Print model
print(model.summary())

#Plot model and save to file
plot_model(model, to_file='vgg.png')

# load an image from file and resize
from keras.preprocessing.image import load_img
image = load_img('mug.jpg', target_size=(224, 224))

# convert the image pixels to a numpy array
from keras.preprocessing.image import img_to_array
image = img_to_array(image)

# reshape data for the model : channel, m, num_rows, num_cols
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

# prepare the image for the VGG model
from keras.applications.vgg16 import preprocess_input
image = preprocess_input(image)

# predict the probability across all output classes
yhat = model.predict(image)

#Decode Prediction
# convert the probabilities to class labels
from keras.applications.vgg16 import decode_predictions
label = decode_predictions(yhat)

# retrieve the most likely result, e.g. highest probability
label = label[0][0]
# print the classification
print('%s (%.2f%%)' % (label[1], label[2]*100))

