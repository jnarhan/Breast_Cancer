#
# Author: Erik Nylander
#
#     Created: Mar 25, 2017
#
# Description: Code for running the VGG16 Model in Keras
#
# Note: Please add your name (comma seperated) when you have added a model to this file.
__author__ = 'Erik Nylander'

import os
import sys
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from skimage import color

# Loading the Model Helper function for my system
os.chdir('E:\erikn\Documents\DATA698\code')
import dwdii_bc_model_helper as bc

# global variables for loading the data
imagePath = "E:/erikn/Documents/DATA698/images/ddsm/png/"
trainDataPath = "E:/erikn/Documents/DATA698/images/ddsm/build/ddsm_train.csv"
testDataPath = "E:/erikn/Documents/DATA698/images/ddsm/build/ddsm_test.csv"
imgResize = (150, 150) # can go up to (224, 224)
modelPath = "E:/erikn/Documents/DATA698/model/"

# loading training and test data
lenTrain = 1440
X_data, Y_data = bc.load_data(trainDataPath, imagePath, maxData = lenTrain, verboseFreq = 50, imgResize=imgResize)
print(X_data.shape)
print(Y_data.shape)

lenTest = 195
X_test, Y_test = bc.load_data(testDataPath, imagePath, maxData = lenTrain, verboseFreq = 50, imgResize=imgResize)
print(X_test.shape)
print(Y_test.shape)

# modifying the data to match the requirments for the VGG16 network
def VGG_Prep(img_data):
    """
    :param img_data: training or test images of shape [#images, height, width]
    :return: the array transformed to the correct shape for the VGG network
                shape = [#images, height, width, 3] transforms to rgb and reshapes
    """
    images = np.zeros([len(img_data), img_data.shape[1], img_data.shape[2], 3])
    for i in range(0, len(img_data)):
        im = img_data[i]
        im = color.gray2rgb(im)
        im *= 255 # Orginal imagnet images were not rescaled
        images[i] = im
    return(images)

# Running the test and train data through VGG16 preperation function
X_data = VGG_Prep(X_data)
X_test = VGG_Prep(X_test)
print(X_data.shape)
print(X_test.shape)

# Creating the VGG16 model
model = applications.VGG16(include_top=False, weights='imagenet', input_shape=(X_data.shape[1], X_data.shape[2], 3))

# Generating the bottleneck features for the training data
bottleneck_features_train = model.predict(X_data)
# Saving the bottleneck features for the training data
np.save(open('bottleneck_features_train.npy', 'wb'), bottleneck_features_train)
np.save(open('labels_train.npy', 'wb'), Y_data)

# Generating the bottleneck features for the test data
bottleneck_features_test = model.predict(X_test)
# Saving the bottleneck features for the test data
np.save(open('bottleneck_features_test.npy', 'wb'), bottleneck_features_test)
np.save(open('labels_test.npy', 'wb'), Y_test)