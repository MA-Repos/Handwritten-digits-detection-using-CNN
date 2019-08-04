#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 18:24:06 2019
"""

import cv2
from keras.datasets import mnist
import numpy as np
from keras.models import load_model

classifier = load_model('TrainedModels/mnist_cnn_with_5_Epochs.h5')

(xTrain, yTrain), (xTest, yTest)  = mnist.load_data()

for i in range(0,20):
    rand = np.random.randint(0,len(xTest))
    inputImg = xTest[rand]

    imageLg = cv2.resize(inputImg, None, fx=10, fy=10, interpolation = cv2.INTER_CUBIC)
    inputImg = inputImg.reshape(1,28,28,1) 
    
    res = str(classifier.predict_classes(inputImg, 1, verbose = 0)[0])

    expandedImage = cv2.copyMakeBorder(imageLg, 100, 100, 100, imageLg.shape[0] ,cv2.BORDER_CONSTANT,value=[0,0,0])
    expandedImage = cv2.cvtColor(expandedImage, cv2.COLOR_GRAY2BGR)
    cv2.putText(expandedImage, str(res), (175, 75) , cv2.FONT_HERSHEY_COMPLEX_SMALL,4, (0,255,0), 2)
    cv2.imshow("Model's Prediction", expandedImage)
    cv2.waitKey(0)

cv2.destroyAllWindows()
