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
from matplotlib import pyplot as plt

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

# Based on emotion_model_jh_v5 from https://github.com/dwdii/emotional-faces/blob/master/src/emotion_model.py
def bc_model_v01(outputClasses, input_shape=(3, 150, 150), verbose=False):
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

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    return model

def bc_model_v02(outputClasses, input_shape=(3, 150, 150), verbose=False):
    model = Sequential()
    model.add(Convolution2D(32, 8, 8, input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add( Dropout(0.4)) # Added dropout

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
        
    if outputClasses > 2:
        loss = 'categorical_crossentropy'
    else:
        loss = 'binary_crossentropy'
        
    print loss   
    model.compile(loss=loss,
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    return model

def bc_model_v03(outputClasses, input_shape=(3, 150, 150), verbose=False):
    model = Sequential()
    model.add(Convolution2D(32, 8, 8, input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add( Dropout(0.4)) # Added dropout

    model.add(Convolution2D(32, 5, 5))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add( Dropout(0.4)) # Added dropout

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
        
    if outputClasses > 2:
        loss = 'categorical_crossentropy'
    else:
        loss = 'binary_crossentropy'
        
    print loss   
    model.compile(loss=loss,
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    return model


# From Jay Narhan
class LossHistory(cb.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        epoch_tr_loss  = logs.get('loss')
        epoch_val_loss = logs.get('val_loss')
        self.losses.append([epoch_tr_loss, epoch_val_loss])
        
        epoch_tr_acc  = logs.get('acc')
        epoch_val_acc = logs.get('val_acc')
        self.acc.append([epoch_tr_acc, epoch_val_acc])

def run_network(data, model, epochs=20, batch=256, verbosity=2, earlyStop=False):
    """

    :param data: X_train, X_test, y_train, y_test
    :param model:
    :param epochs:
    :param batch:
    :return:
    """
    try:
        start_time = time.time()
        cbs = []

        history = LossHistory()
        cbs.append(history)
        X_train, X_test, y_train, y_test = data

        y_trainC = np_utils.to_categorical(y_train )
        y_testC = np_utils.to_categorical(y_test)
        print y_trainC.shape
        print y_testC.shape
        
        if earlyStop:
            earlyStopping = cb.EarlyStopping(monitor='val_loss', min_delta=0, patience=1, verbose=2, mode='auto')
            cbs.append(earlyStopping)

        print 'Training model...'
        model.fit(X_train, y_trainC, nb_epoch=epochs, batch_size=batch,
                  callbacks=cbs,
                  validation_data=(X_test, y_testC), verbose=verbosity)

        print "Training duration : {0}".format(time.time() - start_time)
        score = model.evaluate(X_test, y_testC, batch_size=16, verbose=0)

        print "Network's test score [loss, accuracy]: {0}".format(score)
        print 'CNN Error: {:.2f}%'.format(100 - score[1] * 100)
        
        return model, history.losses, history.acc
    except KeyboardInterrupt:
        print ' KeyboardInterrupt'
        return model, history.losses, history.acc
    
# From Jay Narhan
def plot_losses(losses, acc):
    fig = plt.figure()
    ax = fig.add_subplot(221)
    ax.plot(losses)
    ax.set_title('Model Loss')
    ax.set_ylabel('loss')
    ax.set_xlabel('epoch')
    ax.legend(['train', 'test'], loc='upper left')
    
    ax = fig.add_subplot(222)
    ax.plot(acc)
    ax.set_title('Model Accuracy')
    ax.set_ylabel('accuracy')
    ax.set_xlabel('epoch')
    ax.legend(['train', 'test'], loc='upper left')
    plt.show()    
    
def imageDataGenTransform(img, y):
    # Using keras ImageDataGenerator to generate random images
    datagen = ImageDataGenerator(
        zoom_range = 0.1,
        horizontal_flip = True)
        
    #x = img_to_array(img)
    x = img.reshape(1, 1, img.shape[0], img.shape[1])
    j = 0
    for imgT, yT in datagen.flow(x, y, batch_size = 1, save_to_dir = None):
        img2 = imgT
        break

    return img2    