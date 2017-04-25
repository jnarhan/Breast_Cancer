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
import cv2
from matplotlib import pyplot as plt

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

# Image transformations need to be done before conversion to rgb
# dimg = X_data[0]
# angle = 15
# rows,cols,depth = dimg.shape
# M = cv2.getRotationMatrix2D((cols/2,rows/2), angle, 1)
# drot = cv2.warpAffine(dimg,M,(cols,rows))
# plt.imshow(drot, cmap='gray') 

# t_image = X_data[0] * 255
# t_image = 255 - t_image
# t_image = color.gray2rgb(t_image)
# t_image.shape
# plt.imshow(t_image)

lenTest = 195
X_test, Y_test = bc.load_data(testDataPath, imagePath, maxData = lenTrain, verboseFreq = 50, imgResize=imgResize)
print(X_test.shape)
print(Y_test.shape)

# X_data.shape

# Function for rotating the image files.
def Image_Rotate(img, angle):
    """
    Rotates a given image the requested angle. Returns the rotated image.
    """
    rows,cols = img.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2), angle, 1)
    return(cv2.warpAffine(img,M,(cols,rows)))

# Function for augmenting the images
def Image_Augment(X, Y, vflip=False, hflip=False, major_rotate=False, minor_rotate=False):
    """
    :param  X np.array of images
            Y np.array of labels
            vflip, hflip, major_rotate, minor_rotate set to True to perform the augmentations
    :return The set of augmented iages and their corresponding labels
    
    """
    if len(X) != len(Y):
        print('Data and Label arrays not of the same length.')
    
    n = vflip + hflip + 2*major_rotate + 6*minor_rotate
    augmented = np.zeros([len(X) + n*len(X), X.shape[1], X.shape[2]])
    label = np.zeros([len(Y) + n*len(Y), 1])
    count = 0
    for i in range(0, len(X)):
        augmented[count] = X[i]
        label[count] = Y[i]
        count += 1
        if vflip:
            aug = cv2.flip(X[i], 0)
            augmented[count] = aug
            label[count] = Y[i]
            count += 1
        if hflip:
            aug = cv2.flip(X[i], 1)
            augmented[count] = aug
            label[count] = Y[i]
            count +=1 
        if major_rotate:
            angles = [90, 270]
            for angle in angles:
                aug = Image_Rotate(X[i], angle)
                augmented[count] = aug
                label[count] = Y[i]
                count += 1
        if minor_rotate:
            angles = [-45,-30,-15,15,30,45]
            for angle in angles:
                aug = Image_Rotate(X[i], angle)
                augmented[count] = aug
                label[count] = Y[i]
                count += 1
                
    return(augmented, label)

# Testing the image augmenter
img = X_data[0:1]
lab = Y_data[0:1]
X_aug, Y_aug = Image_Augment(img, lab, vflip=True, hflip=True, major_rotate=True, minor_rotate=True)


X_aug.shape
Y_aug.shape
plt.imshow(X_aug[0], cmap='gray')
plt.imshow(X_aug[1], cmap='gray')
plt.imshow(X_aug[2], cmap='gray')
plt.imshow(X_aug[3], cmap='gray')
plt.imshow(X_aug[4], cmap='gray')

# modifying the data to match the requirments for the VGG16 network
def VGG_Prep(img_data):
    """
    :param img_data: training or test images of shape [#images, height, width]
    :return: the array transformed to the correct shape for the VGG network
                shape = [#images, height, width, 3] transforms to rgb and reshapes
    """
    images = np.zeros([len(img_data), img_data.shape[1], img_data.shape[2], 3])
    for i in range(0, len(img_data)):
        im = 255 - (img_data[i] * 255) # Orginal imagnet images were not rescaled
        im = color.gray2rgb(im)
        images[i] = im
    return(images)


# Augmenting the test data
X_aug, Y_aug = Image_Augment(X_data, Y_data, vflip=True, hflip=True, major_rotate=True, minor_rotate=True)

print(X_aug.shape)
print(Y_aug.shape)
Y_aug[0:20]

# Running the test and train data through VGG16 preperation function
X_prep = VGG_Prep(X_aug)
X_test = VGG_Prep(X_test)
print(X_prep.shape)
print(X_test.shape)


# Creating the VGG16 model
model = applications.VGG16(include_top=False, weights='imagenet')

# Generating the bottleneck features for the training data
bottleneck_features_train = model.predict(X_prep)
# Saving the bottleneck features for the training data
np.save(open('bottleneck_features_train.npy', 'wb'), bottleneck_features_train)
np.save(open('labels_train.npy', 'wb'), Y_aug)

# Generating the bottleneck features for the test data
bottleneck_features_test = model.predict(X_test)
# Saving the bottleneck features for the test data
np.save(open('bottleneck_features_test.npy', 'wb'), bottleneck_features_test)
np.save(open('labels_test.npy', 'wb'), Y_test)
