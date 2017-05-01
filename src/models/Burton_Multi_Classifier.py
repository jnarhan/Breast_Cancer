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

classes = ['malignant', 'benign', 'normal', ]
trainDataBasePath = "/Users/burton/DropBox_Local/test/"
testDataBasePath = "/Users/burton/DropBox_Local/test/"

trainDataPath = "/Users/burton/DropBox_Local/data/ddsm/build/ddsm_train.csv"
# trainDataPathSim = "/Users/burton/DropBox_Local/data/ddsm/build/simulated_train.csv"
# trainDataPathMixed = "/Users/burton/DropBox_Local/data/ddsm/build/mixed_train.csv"
# trainDataPathMixed_8000 = "/Users/burton/DropBox_Local/data/ddsm/build/mixed_train_8000.csv"
# trainDataPath = trainDataPathSim
# trainDataPath = trainDataPathMixed_8000

testDataPath = "/Users/burton/DropBox_Local/data/ddsm/build/ddsm_test.csv"
# testDataPathSim = "/Users/burton/DropBox_Local/data/ddsm/build/sim_test.csv"
# testDataPath = testDataPathSim

imagePath = "/Users/burton/DropBox_Local/data/ddsm/png/"
modelPath = "/Users/burton/DropBox_Local/model/"


class MultiClassifier():
    def __init__(self, classes, trainPaths):
        self.classes = classes
        self.trainPaths = trainPaths

    def getExtractors(self):
        hhfe = HueHistogramFeatureExtractor(10)
        ehfe = EdgeHistogramFeatureExtractor(10)
        haarfe = HaarLikeFeatureExtractor(fname='/Users/burton/Downloads/SimpleCV/SimpleCV/Features/haar.txt')
        return [hhfe, ehfe, haarfe]

    def getClassifiers(self, extractors):
        tree = TreeClassifier(extractors)
        bayes = NaiveBayesClassifier(extractors)
        knn = KNNClassifier(extractors)
        return [tree, bayes, knn]

    def train(self):
        self.classifiers = self.getClassifiers(self.getExtractors())
        for classifier in self.classifiers:
            classifier.train(self.trainPaths, self.classes, verbose=False)

    def test(self, testPaths):
        for classifier in self.classifiers:
            print classifier.test(testPaths, self.classes, verbose=False)

    def classifyImages(self, classifier, imgs):
        for img in imgs:
            className = classifier.classify(img)
            print(className)


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
                print(imageFile)
                print(mode)
                print(finalImagePath + patho + "/" + mode + "/")
                copyfile(allImagesPath + imageFile, finalImagePath + patho + "/" + mode + "/" + imageFile)
def main():
    #Uncomment the lines below whenever a new set of images ins to be loaded into the source folders.
    # copyImages(trainDataPath, "train")
    # copyImages(testDataPath, "test")
    trainPaths = [trainDataBasePath + c + '/train/' for c in classes]
    testPaths = [testDataBasePath + c + '/test/' for c in classes]
    multiClassifier = MultiClassifier(classes, trainPaths)
    multiClassifier.train()
    print "Result test"
    multiClassifier.test(testPaths)
