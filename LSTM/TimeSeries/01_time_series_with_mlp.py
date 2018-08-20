# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 15:13:46 2018

@author: ashutosh
"""

datapath = "D:/DeepLearning/datasets/"
datafile = datapath + "international-airline-passengers.csv"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from keras.models import Sequential
from keras.layers import Dense



dataset = pd.read_csv(datafile)

dataset.head()
dataset.columns = ['Month','Count']

dataset.head()
dataset.shape

plt.plot(dataset['Month'],dataset['Count'])


np.random.seed(7)

#Load dataset in a different way
dataset = pd.read_csv(datafile, usecols=[1], engine='python', skipfooter=3)
dataset = dataset.values
dataset = dataset.astype('float32')

#Split into train and test with 2/3 in train and 1/3 in test

train_size = int(len(dataset)*0.67)
test_size = len(dataset)-train_size

train,test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

print(len(train), len(test))

"""
This default will create a dataset where X is
the number of passengers at a given time (t) and Y is the number of passengers at the next
time (t+1). It can be con
gured and we will look at constructing a di
erently shaped dataset
in the next section.
"""

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

## Create and fir a MLP model
model = Sequential()
model.add(Dense(8, input_dim = look_back, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=200, batch_size=2, verbose=2)


# Estimate model performance
trainScore = model.evaluate(trainX, trainY, verbose=0)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
testScore = model.evaluate(testX, testY, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))


# generate predictions for training
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
plt.plot(dataset)
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()


