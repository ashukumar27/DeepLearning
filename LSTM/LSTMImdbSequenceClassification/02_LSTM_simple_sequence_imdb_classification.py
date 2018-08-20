# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 22:35:49 2018

@author: ashutosh

Simple LSTM for Sequence Classification of Movie Reviews
"""

import numpy as np
import pandas as pd
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
np.random.seed(7)

#Load the dataset, keep top 5000 words
top_words = 5000
(X_train, y_train),(X_test,y_test) = imdb.load_data(num_words=top_words)

#truncate and pad input sequence
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen = max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen = max_review_length)


## Build the Model
embedding_vector_length = 32
model = Sequential()
model.add(Embedding(top_words,embedding_vector_length, input_length = max_review_length ))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics = ['accuracy'])
print(model.summary())

model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs=3, batch_size=64, verbose=1)
#Final Evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=1)

print(scores[1])