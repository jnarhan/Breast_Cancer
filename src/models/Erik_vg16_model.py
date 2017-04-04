#
# Author: Erik Nylander
#
#     Created: Mar 25, 2017
#
# Description: Code for running the running the VGG16 model on data that as been processed through
#               the Erik_vgg16_botleneck.py 
#
# Note: Please add your name (comma seperated) when you have added a model to this file.
__author__ = 'Erik Nylander'

import os
import numpy as np
import keras.callbacks as cb
import keras.utils.np_utils as np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.layers.core import Activation
from keras import applications


# Locations for the bottleneck and labels files that we need
train_bottleneck = 'E:/erikn/Documents/DATA698/model/bottleneck_features_train.npy'
train_labels = 'E:/erikn/Documents/DATA698/model/labels_train.npy'
test_bottleneck = 'E:/erikn/Documents/DATA698/model/bottleneck_features_test.npy'
test_labels = 'E:/erikn/Documents/DATA698/model/labels_test.npy'
top_model_weights_path = 'E:/erikn/Documents/DATA698/model/top_weights01.h5'

# Setting Variables for the experiment
epoch = 50
batch = 48

class LossHistory(cb.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        batch_loss = logs.get('loss')
        self.losses.append(batch_loss)


def train_top_model():
    
    history = LossHistory()
    
    X_train = np.load(train_bottleneck)
    Y_train = np.load(train_labels)
    Y_train = np_utils.to_categorical(Y_train, nb_classes=3)
    
    X_test = np.load(test_bottleneck)
    Y_test = np.load(test_labels)
    Y_test = np_utils.to_categorical(Y_test, nb_classes=3)

    model = Sequential()
    model.add(Flatten(input_shape=X_train.shape[1:]))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3))
    model.add(Activation('softmax'))

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    model.fit(X_train, Y_train,
              nb_epoch=epoch,
              batch_size=batch,
              callbacks=[history],
              validation_data=(X_test, Y_test),
              verbose=2)
    
    score = model.evaluate(X_test, Y_test, batch_size=16, verbose=0)

    print "Network's test score [loss, accuracy]: {0}".format(score)
    
    model.save_weights(top_model_weights_path)


train_top_model()

