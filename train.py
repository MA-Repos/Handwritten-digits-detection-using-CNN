from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

(xTrain, yTrain), (xTest, yTest) = mnist.load_data()

# Info about dataset
print(xTrain.shape)
print( len(xTrain), " -Sample dataset for training. ")
print( len(yTrain), " -Labels in training dataset. " )
print( len(xTest) , " -Sample dataset for testing. " )
print( len(yTest) , " -Labels in test dataset. " )

# Preprocess Data
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

