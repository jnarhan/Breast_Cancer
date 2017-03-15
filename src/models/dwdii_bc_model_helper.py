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

from scipy import misc

import numpy as np

from scipy import misc
from scipy import ndimage
import cv2

def load_training_metadata(metadataFile, balanceViaRemoval = False, verbose=False):
    # Load the existing CSV so we can skip what we've already worked on
    bcDict = {}
    bcCounts = collections.defaultdict(int)
    with open(metadataFile, 'r') as csvfile:
        bcCsv = csv.reader(csvfile)
        headers = bcCsv.next()
        for row in bcCsv:
            patho = row[5].lower()
            if patho == "":
                patho = "benign"
                
            bcDict[row[0]] = patho
            bcCounts[patho] += 1

    if verbose:
        print "Before Balancing"
        print "----------------"
        for e in bcCounts:
            print e, bcCounts[e]
            
    #if balanceViaRemoval:
        #balanaceViaRemoval(bcCounts, bcDict)

    if verbose:
        print
        print "After Balancing"
        print "----------------"
        for e in bcCounts:
            print e, bcCounts[e]
        
    return bcDict