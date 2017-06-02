# coding: utf-8

# A package of helper functions for Neural Net processing in breast cancer detection/diagnosis
# Author: Jay Narhan
# Date:   April-2017

import os
import gc
import csv
import sys
import time
import shutil
import itertools
import collections
import numpy as np

from scipy import misc

import keras.callbacks as cb
from keras.utils import np_utils

from matplotlib import pyplot as plt


def pprint(msg):
    print '-' * len(msg)
    print msg
    print '-' * len(msg)

# Copy of D. Dittenhafer's loading and balancing by removal.
# Balances the data set by removing images from over-represented classes
def load_meta(metaFile, patho_idx, file_idx, balanceByRemoval = False, verbose = False):
    bcMetaFile = {}
    bcCounts = collections.defaultdict(int)
    
    with open(metaFile, 'r') as csvfile:
        bcCSV = csv.reader(csvfile)
        headers = bcCSV.next()
        for row in bcCSV:
            patho = row[ patho_idx].lower()
            bcMetaFile[ row[file_idx]] = patho
            bcCounts[patho] += 1
    
    if verbose:
        pprint('Before Balancing')
        for k in bcCounts:
            print '{0:10}: {1}'.format(k, bcCounts[k])
            
    if balanceByRemoval:
        balanceViaRemoval(bcMetaFile, bcCounts, factor=1.0)
        
        if verbose:
            pprint('After Balancing')
            for k in bcCounts:
                print '{0:10}: {1}'.format(k, bcCounts[k])

    return bcMetaFile, bcCounts


def balanceViaRemoval(meta, counts, depth = 0, factor = 1.50):
    
    if(depth >= 2):
        return

    # First get mean items per category
    num_obs = len(meta)
    num_classes = len(counts)
    avgC = num_obs / num_classes
    theshold = avgC * factor
    
    if depth == 0:
        print "balanceViaRemoval.avgC: " + str(avgC)
        print "balanceViaRemoval.theshold: " + str(theshold)
        
    # Determine categories for balancing.
    toBeBalanced = []
    for c in counts.keys():
        if counts[c] > theshold:
            toBeBalanced.append(c)

    # iterate over categories to be balanced and do balancing.
    for b in toBeBalanced:
        candidatesForRemoval = []
        for f in meta.keys():
            if meta[f] == b:
                candidatesForRemoval.append(f)

        np.random.shuffle(candidatesForRemoval)
        candidatesForRemoval = candidatesForRemoval[avgC:]
        for c in candidatesForRemoval:
            del meta[c]

        counts[b] = avgC

    balanceViaRemoval(meta, counts, depth + 1, factor)

def get_clsCnts(y_data, cats):
    ys = np.ravel(y_data)
    labels = reverseDict(cats)
    bcCounts = collections.defaultdict(int)

    for lab in ys:
        bcCounts[lab] += 1
    try:
        for key, value in labels.items():
            bcCounts[value] = bcCounts.pop(key)
    except KeyError:
        pass
    return bcCounts

# Alternative to balancing by over-sampling of minority cases through synthetic augmentation
def balanceViaSmote(cls_cnts, meta_info, data_dir, aug_dir, catagories,
                    datagen, X_data, Y_data, seed=None, verbose=False):
    aug_imgs = []

    if seed:
        np.random.seed(seed)

    max_class_key = max(cls_cnts, key=cls_cnts.get)
    max_class_val = cls_cnts[ max_class_key ]

    for key, value in cls_cnts.items():
        if key == max_class_key:
            pass
        else:
            grow_by = max_class_val - value
            imgs = {k:v for k, v in meta_info.items() if v == key}
            # take a random selection of grow_by size, with replacement
            key_indxs = np.random.choice(imgs.keys(), size=grow_by, replace=True)
            for k1 in key_indxs:
                aug_imgs.append({k:v for k,v in imgs.items() if k == k1})

            save_dir = aug_dir + key + '/'

            # Overwrite folder and contents if folder exists:
            if os.path.exists(save_dir):
                shutil.rmtree(save_dir)
            os.makedirs(save_dir)

            # Load randomly selected images of given catagory into memory
            aug_X = list()
            aug_Y = list()
            for i in aug_imgs:
                img_x, img_y = load_data(i, data_dir, catagories)
                aug_X.append(img_x)
                aug_Y.append(img_y)

            # Generate augmented images
            aug_X = np.reshape(aug_X, (len(aug_X), aug_X[0].shape[0], aug_X[0].shape[1], aug_X[0].shape[2]))

            for x_batch, y_batch in datagen.flow(aug_X, aug_Y, batch_size=len(aug_X), seed=seed,
                                                 save_to_dir= save_dir,
                                                 save_prefix= key + '_aug',
                                                 save_format= 'png'):
                X_data = np.concatenate(
                    (X_data, np.reshape(x_batch, (len(x_batch), x_batch.shape[2], x_batch.shape[3]))))
                Y_data = np.concatenate((Y_data, np.reshape(y_batch, (len(y_batch), y_batch.shape[2]))))
                break

    if verbose:
        bcCounts = get_clsCnts(Y_data, catagories)
        pprint('After Balancing')
        for k in bcCounts:
            print '{0:10}: {1}'.format(k, bcCounts[k])

    return X_data, Y_data


def bcLabels(items):
    labels = {}
    for i, item in enumerate(items):
        labels[item] = i
    return labels


def reverseDict(d):
    ndxBC = {}
    for k in d:
        ndxBC[d[k]] = k

    return ndxBC

# Adapted from D. Dittenhafer
def load_data(metaData, imgPath, categories, imgSize = (255,255), imgResize = None,
              verbose = True,  verboseFreq = 200):
    
    total = len(metaData)
    
    x, y = imgSize
    if imgResize is not None:
        x, y = imgResize
    
    # Allocate containers for the data
    X_data = np.zeros( [total, x, y])
    Y_data = np.zeros( [total, 1], dtype=np.int8)
    
    # Load images based on meta_data:
    for i, fn in enumerate( metaData.keys()):
        filepath = os.path.join(imgPath, fn)
        if os.path.exists(filepath):
            img = misc.imread(filepath, flatten=True)
        else:
            img = None
            print "Not Found: " + filepath
            
        if imgResize is not None:
            img = misc.imresize(img, imgResize)
            gc.collect()
        
        X_data[i] = img
        Y_data[i] = categories[ metaData[fn].lower()]
        
    X_data = X_data.astype('float32')
    X_data /= float(255)

    return X_data, Y_data


def prep_data(data, labels):
    print 'Prep data for NNs ...'
    
    X_train, X_test, y_train, y_test = data

    # one-hot encoding of output i.e int to binary matrix rep:    
    y_train = np_utils.to_categorical(zip(*y_train)[0], len(labels))
    y_test  = np_utils.to_categorical(zip(*y_test)[0], len(labels))
    
    channel, width, height = (1, X_train[0].shape[0], X_train[0].shape[1])

    # CNN require [channel e.g grayscale = 1][width][height]
    X_train = np.reshape(X_train, (len(X_train), channel, width, height))      
    X_test  = np.reshape(X_test,  (len(X_test), channel, width, height))

    print 'Data Prepped for Neural Nets.'
    return [X_train, X_test, y_train, y_test]


class LossHistory(cb.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.acc = []
        
    def on_epoch_end(self, epoch, logs={}):
        epoch_tr_loss  = logs.get('loss')
        epoch_val_loss = logs.get('val_loss')
        self.losses.append([epoch_tr_loss, epoch_val_loss])
        
        epoch_tr_acc  = logs.get('acc')
        epoch_val_acc = logs.get('val_acc')
        self.acc.append([epoch_tr_acc, epoch_val_acc])


def run_network(data, model, aug=False, dataGen=None, earlyStop=False, epochs=20, batch=256, seed=7):
    try:
        if aug and (dataGen is None):
            raise ValueError('Attempting to augment data without providing inline data generator.')

        start_time = time.time()
        cbs = []

        X_train, X_test, y_train, y_test = data
        history = LossHistory()
        cbs.append(history)

        if earlyStop:
            earlyStopping = cb.EarlyStopping(monitor='val_loss', min_delta=0, patience=1, verbose=2, mode='auto')
            cbs.append(earlyStopping)

        print 'Training model...'
        if not aug:
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch,
                      callbacks=cbs, validation_data=(X_test, y_test), verbose=2)
        else:
            model.fit_generator(dataGen.flow(X_train, y_train, batch_size=batch, seed=seed),
                                steps_per_epoch=len(X_train) / batch,
                                epochs=epochs,
                                callbacks=cbs,
                                validation_data=(X_test, y_test), verbose=2)

        print "Training duration : {0}".format(time.time() - start_time)
        score = model.evaluate(X_test, y_test, batch_size=16, verbose=2)  # Evaluate the model

        print "Network's test score [loss, accuracy]: {0}".format(score)
        print 'CNN Error: {:.2f}%'.format(100 - score[1] * 100)
        
        return model, history.losses, history.acc, score

    except ValueError as err:
        print 'Error: {}'.format(err)
        sys.exit(1)

    except KeyboardInterrupt:
        print 'KeyboardInterrupt'
        return model, history.losses


def predict(model, images):
    return model.predict_classes(images, verbose=2)


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


def save_model(dir_path, model, name):
    curr_dir = os.getcwd()
    os.chdir(dir_path)

    with open(name + "_model.yaml", "w") as yaml_file:
        yaml_file.write(model.to_yaml())

    model.save_weights(name + "_weights.hd5", overwrite=True)
    print ("Model and Weights Saved to Disk")

    os.chdir(curr_dir)


# From: http://scikit-learn.org/stable/auto_examples/model_selection/
# plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
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


def cat_stats(matrix, TP_idx=3):
    matrix = matrix.flatten()
    TP = matrix[TP_idx]
    if TP_idx == 3:
        TN = matrix[0]
        FP = matrix[1]
        FN = matrix[2]
    elif TP_idx == 0:
        TN = matrix[3]
        FP = matrix[2]
        FN = matrix[1]
    else:
        print ('ERROR: Uncertain how to interpret confusion matrix')

    # Refers to the test's ability to correctly detect patients who do have the condition.  
    # When it’s actually yes, how often does it predict yes:
    Sensitivity = TP / float(TP + FN)

    # Refers to the test's ability to correctly detect patients without a condition.  
    # When it’s actually no, how often predict no.
    Specificity = TN / float(FP + TN)

    # The proportion of positive test results to the number of true positives.  
    # When it predicts yes, how often is it correct.
    PPV = TP / float(TP + FP)

    # The proportion of negative test results to the number of true negatives.  
    # When it predicts no, how often is is correct.
    NPV = TN / float(TN + FN)

    # Reaches its best at 1 and worst at 0.
    F1 = float((Sensitivity * PPV) / (Sensitivity + PPV)) * 2

    # Overall, how often is the classifier correct.
    Class_Acc = (TP + TN) / float(TP + FN + FP + TN)

    return {'Sensitivity': round(Sensitivity * 100, 2), 'Specificity': round(Specificity * 100, 2),
            'PPV': round(PPV * 100, 2), 'NPV': round(NPV * 100, 2), 'F1': round(F1, 2),
            'Accuracy': round(Class_Acc * 100, 2)}


if __name__ == '__main__':
    # Add test  cases here
    test_mat = np.array([[228, 40],
                         [37, 257]])
    print cat_stats(test_mat)