import datetime
#import Image
import gc
import numpy as np
import os
import random
from scipy import misc
import string
import time

# Set some Theano config before initializing
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32,allow_gc=False,openmp=True"
import theano

import keras.callbacks as cb
import keras.utils.np_utils as np_utils
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import MaxPooling2D, ZeroPadding2D
from keras.preprocessing.image import ImageDataGenerator

print "Hello from job!"
print str(datetime.datetime.now())

model = Sequential()
print (model.summary())
