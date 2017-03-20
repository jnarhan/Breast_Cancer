#
# Author: Daniel Dittenhafer
#
#     Created: Mar 19, 2017
#
# Description: Model Creation Functions
#
# Note: Please add your name (comma seperated) when you have added a model to this file.
__author__ = 'Daniel Dittenhafer'

import time

import keras.callbacks as cb
import keras.utils.np_utils as np_utils
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import MaxPooling2D, ZeroPadding2D
from keras.preprocessing.image import ImageDataGenerator

# Based on emotion_model_jh_v5 from https://github.com/dwdii/emotional-faces/blob/master/src/emotion_model.py
def bc_model_v0(outputClasses, input_shape=(3, 150, 150), verbose=False):
    model = Sequential()
    model.add(Convolution2D(32, 8, 8, input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(32, 5, 5))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, 2, 2))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    # model.add(Dropout(0.4))
    model.add(Dense(outputClasses))
    model.add(Activation('softmax'))

    if verbose:
        print (model.summary())

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    return model

class LossHistory(cb.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        batch_loss = logs.get('loss')
        self.losses.append(batch_loss)

def run_network(data, model, epochs=20, batch=256, verbosity=2):
    """

    :param data: X_train, X_test, y_train, y_test
    :param model:
    :param epochs:
    :param batch:
    :return:
    """
    try:
        start_time = time.time()

        history = LossHistory()
        X_train, X_test, y_train, y_test = data

        y_trainC = np_utils.to_categorical(y_train )
        y_testC = np_utils.to_categorical(y_test)
        print y_trainC.shape
        print y_testC.shape

        print 'Training model...'
        model.fit(X_train, y_trainC, nb_epoch=epochs, batch_size=batch,
                  callbacks=[history],
                  validation_data=(X_test, y_testC), verbose=verbosity)

        print "Training duration : {0}".format(time.time() - start_time)
        score = model.evaluate(X_test, y_testC, batch_size=16, verbose=0)

        print "Network's test score [loss, accuracy]: {0}".format(score)
        return model, history.losses
    except KeyboardInterrupt:
        print ' KeyboardInterrupt'
        return model, history.losses