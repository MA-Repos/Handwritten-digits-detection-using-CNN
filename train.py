#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import keras
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import numpy as np

(xTrain, yTrain), (xTest, yTest) = mnist.load_data()

# Info about dataset
print(xTrain.shape)
print( len(xTrain), " -Sample dataset for training. ")
print( len(yTrain), " -Labels in training dataset. " )
print( len(xTest) , " -Sample dataset for testing. " )
print( len(yTest) , " -Labels in test dataset. " )

####################
# Preprocess Data
####################
imgRows = xTrain[0].shape[0]
imgCols = xTrain[0].shape[1]

cnnInputShape = (imgRows, imgCols, 1)

# Reshape images for Keras - 60000,28,28 to 60000,28,28,1
xTrain = xTrain.reshape(xTrain.shape[0], imgRows, imgCols, 1)
xTest = xTest.reshape(xTest.shape[0], imgRows, imgCols, 1)

xTrain = xTrain.astype(float)
xTest = xTest.astype(float)

# (0 - 255) to (0 - 1) Normalization of dataset
xTrain /=255
xTest /=255

print("xTrain shape: ", xTrain.shape)

#labels encoding - yTrain & yTest
yTrain = np_utils.to_categorical(yTrain)
yTest = np_utils.to_categorical(yTest)
numOfClasses = yTest.shape[1]

print("No. of Classes: ", numOfClasses)


#####################
# Model Creation
#####################

model = Sequential()

model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=cnnInputShape))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(numOfClasses, activation='softmax'))

model.compile(loss= 'categorical_crossentropy', optimizer = SGD(0.1), metrics = ['accuracy'])

print(model.summary())


#####################
#Training the Model
#####################

batchSize = 32
epochs = 5

trainingHistory = model.fit(xTrain,
							yTrain,
							batch_size = batchSize,
							epochs = epochs,
							validation_data=(xTest,yTest),
							verbose= 1)

score = model.evaluate(xTest, yTest, verbose=0)
print("Test loss score: ", score[0])
print("Test accuracy score: ", score[1])

if(score[1] > 0.95):
	model.save("TrainedModels/mnist_cnn_with_5_Epochs.h5")
	print("Model Saved")
else:
	print("Model has low accuracy")

######################################
#Display Matplot of Accuracy
######################################

history_dict = trainingHistory.history

loss_values = history_dict['loss']
acc = history_dict['acc']
val_acc = history_dict['val_acc']
dis_epochs = range(1, len(loss_values) + 1)

line1 = plt.plot(dis_epochs, val_acc, label='Validation/Test Accuracy')
line2 = plt.plot(dis_epochs, acc, label='Training Accuracy')
plt.setp(line1, linewidth=2.5, marker = '+', markersize=10.0)
plt.setp(line2, linewidth=2.5, marker = '>', markersize=10.0)
plt.xlabel('Epochs') 
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()
plt.show()