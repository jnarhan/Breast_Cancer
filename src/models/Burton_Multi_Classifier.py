
# To run this program you need the SimpleCV package and the Orange machine learning library installed.
# https://orange.biolab.si and https://github.com/biolab/orange3
# http://simplecv.org and https://github.com/sightmachine/SimpleCV

# This code was modelled after the one found at the following page:
# http://jmgomez.me/a-fruit-image-classifier-with-python-and-simplecv/

from SimpleCV.base import *
from SimpleCV.ImageClass import Image
from SimpleCV.Features.FeatureExtractorBase import *
from SimpleCV import HueHistogramFeatureExtractor
from SimpleCV import EdgeHistogramFeatureExtractor
from SimpleCV import HaarLikeFeatureExtractor
# Import the classifiers.
from SimpleCV import SVMClassifier
from SimpleCV import TreeClassifier
from SimpleCV import NaiveBayesClassifier
from SimpleCV import KNNClassifier

import csv
import numpy as np
np.seterr(divide='ignore', invalid='ignore')

# Following are the three classes of images.
classes = ['malignant', 'benign', 'normal']

trainDataBasePath = "/Users/burton/DropBox_Local/test/"
testDataBasePath = "/Users/burton/DropBox_Local/test/"

# The file below contains the training image names and their labels.
trainDataPath = "/Users/burton/DropBox_Local/data/ddsm/build/ddsm_train.csv"

# The file below contains the test image names and their labels.
testDataPath = "/Users/burton/DropBox_Local/data/ddsm/build/ddsm_test.csv"

# Following is the base directory of the image files.
imagePath = "/Users/burton/DropBox_Local/data/ddsm/png/"

class MultiClassifier():
    def __init__(self, classes, trainPaths):
        self.classes = classes
        self.trainPaths = trainPaths

    def getFeatureExtractors(self):
        hueHistogramFeatureExtractor = HueHistogramFeatureExtractor(10)
        edgeHistogramFeatureExtractor = EdgeHistogramFeatureExtractor(10)
        haarLikeFeatureExtractor = HaarLikeFeatureExtractor(fname='/Users/burton/Downloads/SimpleCV/SimpleCV/Features/haar.txt')
        return [hueHistogramFeatureExtractor, edgeHistogramFeatureExtractor, haarLikeFeatureExtractor]

    def getModels(self, extractors):
        tree = TreeClassifier(extractors)
        bayes = NaiveBayesClassifier(extractors)
        knn = KNNClassifier(extractors)
        return [tree, bayes, knn]
        # return [knn]

    def train(self):
        self.classifiers = self.getModels(self.getFeatureExtractors())
        for classifier in self.classifiers:
            classifier.train(self.trainPaths, self.classes, verbose=False)

    def test(self, testPaths):
        for classifier in self.classifiers:
            print classifier.test(testPaths, self.classes, verbose=False)

    def classifyImages(self, classifier, imgs):
        for img in imgs:
            className = classifier.classify(img)
            print(className)


# The following function can be run once (or as required) to copy images to folders with the named with the appropriate labels. This is required prior to running the
# Multi-classifier model.
def copyImages(metadataFile, mode,  exclude = ['unproven', 'pathology', 'benign_without_callback']):
    i=0;
    with open(metadataFile, 'r') as csvfile:
        bcCsv = csv.reader(csvfile)
        headers = bcCsv.next()
        for row in bcCsv:
            patho = row[NDX_PATHOLOGY].lower()
            imageFile = row[NDX_NAME]
            if patho == "":
                patho = "normal"
            if patho in exclude:
                pass
            else:
                i=i+1
                # print(patho)
                print(imageFile)
                print(mode)
                print(finalImagePath + patho + "/" + mode + "/")
                copyfile(allImagesPath + imageFile, finalImagePath + patho + "/" + mode + "/" + imageFile)

# The main fucntion to run the classifiers. The images need to be copied to the folders with appropriate labels before running this function.
# The copy images function above can be used to copy the images.
def main():
    # Uncomment the lines below whenever a new set of images ares to be loaded into the source folders.
    # copyImages(trainDataPath, "train")
    # copyImages(testDataPath, "test")
    trainPaths = [trainDataBasePath + c + '/train/' for c in classes]
    testPaths = [testDataBasePath + c + '/test/' for c in classes]
    multiClassifier = MultiClassifier(classes, trainPaths)
    multiClassifier.train()
    print "Result test"
    multiClassifier.test(testPaths)
