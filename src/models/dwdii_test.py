#
# Author: Daniel Dittenhafer
#
#     Created: Mar 27, 2016
#
# Description: Test script
#
__author__ = 'Daniel Dittenhafer'

import dwdii_bc_model_helper as bc
import os
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

def barChart(data, filename="barchart.png", title='Bar Chart', xLabelRotation=10, colors=('b','r','g')):

    theKeys = []
    theValues = []
    for key, value in sorted(data.items()):
        theKeys.append(key)
        theValues.append(value)

    width = 0.35  # the width of the bars
    ind = np.arange(len(data))
    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, theValues, width, color=colors)

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(theKeys, rotation=xLabelRotation)

    # ax.legend((rects1[0]), ('MIAS'))
    plt.show()
    fig.savefig(filename, dpi=fig.dpi)

def main():
    """Our main function."""

    action = "split"

    #compareFolders()
    #compareLegendAndFiles()

    #dataFile = "C:\Code\Data\DATA698-ResearchProj\Ddsm.csv"
    #imgPath = "C:\Code\Data\DATA698-ResearchProj\ddsm-sm"
    dataFile = "C:\Users\Dan\Dropbox (DATA698-S17)\DATA698-S17\data\ddsm\png\Ddsm_png.csv"
    imgPath = "C:\Users\Dan\Dropbox (DATA698-S17)\DATA698-S17\data\ddsm\png"

    if(action == "load"):
        bc.load_training_metadata(dataFile, True)

        X_data, Y_data = bc.load_data(dataFile,
                                      imgPath,
                                      maxData = 1000,
                                      imgResize = (150, 150),
                                      verboseFreq = 25)

        print X_data.shape

        bc.to_categorical(Y_data)
    elif action == "split":

        outDir = "C:\Code\Other\Breast_Cancer\data"
        valCsv = os.path.join(outDir, "ddsm_val.csv")
        testCsv = os.path.join(outDir, "ddsm_test.csv")
        trainCsv = os.path.join(outDir, "ddsm_train.csv")

        bc.splitTrainTestValSets(dataFile, valCsv, testCsv, trainCsv)

        a, b, valData = bc.load_training_metadata(valCsv)
        barChart(valData, filename="../../figures/ddsm_val_dist.png", title="DDSN Validation Set Distribution")

        a, b, testData = bc.load_training_metadata(testCsv)
        barChart(testData, filename="../../figures/ddsm_test_dist.png", title="DDSN Test Set Distribution")

        a, b, trainData = bc.load_training_metadata(trainCsv)
        barChart(trainData, filename="../../figures/ddsm_train_dist.png", title="DDSN Training Set Distribution")

    print "Done"


# This is the main of the program.
if __name__ == "__main__":
    main()