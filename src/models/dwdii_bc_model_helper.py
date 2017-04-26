#
# Author: Daniel Dittenhafer
#
#     Created: Mar 14, 2017
#
# Description: Model Helper Functions
#
#
__author__ = 'Daniel Dittenhafer'
import collections
import csv
import os
import random
import sys
import gc
import itertools
from decimal import *

from scipy import misc

import numpy as np

from scipy import misc
from scipy import ndimage
import cv2

import matplotlib.cm as cm
import matplotlib.pyplot as plt

NDX_NAME = 0
NDX_TYPE = 1
NDX_ABTYPE = 2
NDX_SCANNER = 3
NDX_SUBFOLDER = 4
NDX_PATHOLOGY = 5

def load_training_metadata(metadataFile,
                           balanceViaRemoval = False,
                           verbose=False,
                           exclude = ['unproven', 'pathology', 'benign_without_callback'],
                           normalVsAbnormal=False):
    """ Loads the designated meta data optionally balancing the data by removing heavily weighted category entries.

    3 result sets are returned:
        1) Dictionary where key = filename and value = label (normal, benign, malignant)
        2) Dictionary where key = filename and value = list with values sub folder)= (0,1,2,3,4)
        3) Dictionary where key = label (normal, benign, etc) and value = count of images in category.

    :param metadataFile:
    :param balanceViaRemoval:
    :param verbose:
    :param exclude:
    :return:
    """

    # Load the existing CSV so we can skip what we've already worked on
    abnormalList = ["benign", "malignant"]
    bcDict = {}
    bcMetaDict = {}
    bcCounts = collections.defaultdict(int)
    with open(metadataFile, 'r') as csvfile:
        bcCsv = csv.reader(csvfile)
        headers = bcCsv.next()
        for row in bcCsv:
            subfld = row[NDX_SUBFOLDER]
            patho = row[NDX_PATHOLOGY].lower()
            if patho == "":
                patho = "normal"


            if patho in exclude:
                pass
            else:

                if normalVsAbnormal and (patho in abnormalList):
                    patho = "abnormal"

                # Load into our result sets
                bcDict[row[0]] = patho
                bcMetaDict[row[0]] = (subfld)
                bcCounts[patho] += 1

    if verbose:
        print "Raw Balance"
        print "----------------"
        for e in bcCounts:
            print e, bcCounts[e]
            
    if balanceViaRemoval:
        balanaceViaRemoval(bcCounts, bcDict, factor=1.0)
        if verbose:
            print
            print "After Balancing"
            print "----------------"
            for e in bcCounts:
                print e, bcCounts[e]

        
    return bcDict, bcMetaDict, bcCounts


def balanaceViaRemoval(emoCounts, emoDict, depth = 0, factor = 1.50):

    if(depth >= 2):
        return

    # First get mean items per category
    sum = len(emoDict)
    avgE = sum / len(emoCounts)
    theshold = avgE * factor
    
    if depth == 0:
        print "balanaceViaRemoval.avgE: " + str(avgE)
        print "balanaceViaRemoval.theshold: " + str(theshold)
        
    # Determine categories for balancing.
    toBeBalanced = []
    for e in emoCounts.keys():
        if emoCounts[e] > theshold:
            toBeBalanced.append(e)

    # iterate over categories to be balanced and do balancing.
    for b in toBeBalanced:
        candidatesForRemoval = []
        for f in emoDict.keys():
            if emoDict[f] == b:
                candidatesForRemoval.append(f)

        random.shuffle(candidatesForRemoval)
        candidatesForRemoval = candidatesForRemoval[avgE:]
        for c in candidatesForRemoval:
            del emoDict[c]

        emoCounts[b] = avgE

    balanaceViaRemoval(emoCounts, emoDict, depth + 1, factor)


def bcNumerics():
    emoNdx = {}
    emoNdx["normal"] = 0
    emoNdx["benign"] = 1
    emoNdx["malignant"] = 2
    return emoNdx

def numericBC():
    emoNdx = bcNumerics()
    ndxEmo = {}
    for k in emoNdx:
        ndxEmo[emoNdx[k]] = k

    return ndxEmo

def bcNormVsAbnormNumerics():
    emoNdx = {}
    emoNdx["normal"] = 0
    emoNdx["abnormal"] = 1
    return emoNdx

def reverseDict(d):
    ndxEmo = {}
    for k in d:
        ndxEmo[d[k]] = k

    return ndxEmo

def load_data(metadataFile,
              imagesPath,
              categories = bcNumerics(),
              verbose=True,
              verboseFreq = 200,
              maxData = None,
              imgSize = (350, 350),
              imgResize = None,
              theseEmotions = None,
              normalVsAbnormal = False):
    """Helper function to load the training/test data"""

    show = False

    # Load the CSV meta data
    emoMetaData, bcDetaDict, bcCounts = load_training_metadata(metadataFile, True, verbose=verbose, normalVsAbnormal=normalVsAbnormal)
    total = Decimal(len(emoMetaData))
    ndx = 0

    x, y = imgSize
    if imgResize is not None:
        x, y = imgResize
        
    if maxData is not None:
        total = Decimal(maxData)

    # Allocate containers for the data
    X_data = np.zeros([total, x, y])
    Y_data = np.zeros([total, 1], dtype=np.int8)

    # load the image bits based on what's in the meta data
    for k in emoMetaData.keys():
        
        if theseEmotions is None or emoMetaData[k] in theseEmotions:

            # Load the file
            filepath = os.path.join(imagesPath, bcDetaDict[k][0], k)
            #filepath = filepath + ".png"
            if(os.path.exists(filepath)):
                img = misc.imread(filepath, flatten = True) # flatten = True?
            else:
                img = None
                print "Not Found: " + filepath

            # Only accept images that were loaded
            if img is not None: 

                # Verbose status
                if verbose and ndx % verboseFreq == 0:
                    msg = "{0:.4f}: {1}\r\n".format(ndx / total, k)
                    sys.stdout.writelines(msg)
                
                # Resize if desired.
                if imgResize is not None:
                    img = misc.imresize(img, imgResize)
                    gc.collect()
                    if show:
                        plt.imshow(img, cmap=cm.gray)
                        plt.show()
                    
                X_data[ndx] = img

                rawEmotion = emoMetaData[k]
                emotionKey = rawEmotion.lower()
                emotionNdx = categories[emotionKey]
                
                Y_data[ndx] = emotionNdx

                ndx += 1
                
        if maxData is not None and maxData <= ndx:
            break
            
    Y_data = Y_data[:ndx]
    X_data = X_data[:ndx]
    
    X_data = X_data.astype('float32')
    X_data /= 255.0

    return X_data, Y_data

def to_categorical(y, nb_classes=None):
    '''Convert class vector (integers from 0 to nb_classes) to binary class matrix, for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
        nb_classes: total number of classes
    # Returns
        A binary matrix representation of the input.
    '''
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.
    return Y

def getNameParts(name):
    parts = name.split(".")
    sideParts = parts[1].split("_")

    case = parts[0]
    side = sideParts[0]
    return case, side

def splitTrainTestValSets(metadataFile, valCsv, testCsv, trainCsv, valSize = 100, trainPct = 0.80, seed = 20275):
    """Generates 3 CSV files containing the meta data split from the source meta data file. First a Numpy
    random shuffle is performed on the data loaded from the metadataFile.

        :param metadataFile: the path to the source CSV file
        :param valCsv: The path to the output CSV to be overwritten by the new validation meta data.
        :param testCsv: The path to the output CSV to be overwritten by the new test meta data.
        :param trainCsv: The path to the output CSV to be overwritten by the new train meta data.
        :param valSize: The number of data rows to pull out for validation purposes
        :param trainPct: Of the remaining data rows after the validation rows have been removed, the percent of
                         data to seperate for training purposes. After the training data is extracted, the final
                         remaining data is saved to the test data set.
    """

    caseSides = {}
    with open(metadataFile, 'r') as csvfile:
        bcCsv = csv.reader(csvfile)
        headers = bcCsv.next()
        headers = bcCsv.next()
        for row in bcCsv:
            case, side = getNameParts(row[NDX_NAME])

            key = "{0}-{1}".format(case, side)

            # build list of case-sides
            caseSides[key] = (case, side)

    # Split the keys up
    csKeys = caseSides.keys()

    # Shuffle
    np.random.seed(seed)
    np.random.shuffle(csKeys)

    valKeys = csKeys[0 : valSize]
    remainingKeys = csKeys[valSize + 1 : len(csKeys) - 1]

    trainNdx = int(round(len(remainingKeys) * trainPct))
    trainKeys = remainingKeys[0 : trainNdx]
    testKeys = remainingKeys[trainNdx + 1 : len(remainingKeys) - 1]

    # split the actual meta data
    with open(metadataFile, 'r') as csvfile:
        with open(valCsv, 'wb') as valfile:
            with open(testCsv, 'wb') as testfile:
                with open(trainCsv, 'wb') as trainfile:
                    bcCsv = csv.reader(csvfile)
                    valCsv = csv.writer(valfile)
                    testCsv = csv.writer(testfile)
                    trainCsv = csv.writer(trainfile)

                    headers = bcCsv.next()
                    headers = bcCsv.next()

                    valCsv.writerow(headers)
                    testCsv.writerow(headers)
                    trainCsv.writerow(headers)

                    for row in bcCsv:
                        case, side = getNameParts(row[NDX_NAME])
                        key = "{0}-{1}".format(case, side)

                        if(key in valKeys):
                            valCsv.writerow(row)
                        elif (key in testKeys):
                            testCsv.writerow(row)
                        elif (key in trainKeys):
                            trainCsv.writerow(row)


    return trainKeys, testKeys, valKeys

   # for k in csKeys:

def load_mias_labeldata(metadataFile, skip_lines=102):

    ld = {}
    with open(metadataFile, 'r') as csvfile:

        emoCsv = csv.reader(csvfile, delimiter=' ')
        # skip first 104 lines of description info
        for i in range(0, skip_lines):
            emoCsv.next()

        for row in emoCsv:
            if len(row) >= 2:
                ld[row[0]] = [row[2]]
                if row[2] != "NORM":
                    ld[row[0]].append(row[3])

    return ld


# From: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{0:.4f}'.format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def cleanDataSet(csvFile, imageRoot):

    data = []
    with open(csvFile, 'r') as csvfile:
        bcCsv = csv.reader(csvfile)
        headers = bcCsv.next()
        for row in bcCsv:

            name = row[NDX_NAME]
            subfld = row[NDX_SUBFOLDER]

            fullName = os.path.join(imageRoot, subfld, name)
            if os.path.exists(fullName):
                data.append(row)
            else:
                print "Not found: " + fullName

    with open(csvFile + "2.csv", 'wb') as file:
        dataCsv = csv.writer(file)
        dataCsv.writerow(headers)
        for row in data:
            dataCsv.writerow(row)



