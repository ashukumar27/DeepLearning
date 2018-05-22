#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 10:31:47 2018

@author: ashutosh

Image Classification with VGG19
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)

os.chdir("/Users/ashutosh/Documents/analytics/DeepLearning/VGG19")
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input, decode_predictions
from keras.preprocessing import image

from keras.models import Model
import cv2


# load pre-trained model
model = VGG19(weights='imagenet', include_top=True)
# display model layers
model.summary()

# display the image
img_disp = plt.imread('./peacock.jpg')
#img_disp = cv2.cvtColor(img_disp, cv2.COLOR_BGR2RGB)
plt.imshow(img_disp)
plt.axis("off")  
plt.show()

# pre-process the image
img = image.load_img('./peacock.jpg', target_size=(224, 224))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = preprocess_input(img)
# predict the output 
preds = model.predict(img)
# decode the prediction
pred_class = decode_predictions(preds, top=3)[0][0]
print ("Predicted Class: %s"%pred_class[1])
print ("Confidance: %s"%pred_class[2])