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
import csv
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

def barChart(data, filename="barchart.png", title='Bar Chart', xLabelRotation=10, colors=cm.get_cmap("Paired"), show=True, yAxisLabel="Count"):

    theKeys = []
    theValues = []
    valTotal = 0
    for key, value in sorted(data.items()):
        theKeys.append(key)
        theValues.append(value)
        valTotal += value

    width = 0.35  # the width of the bars
    ind = np.arange(len(data))
    fig, ax = plt.subplots()
    cl = []
    for i in range(len(theValues)):
        cl.append(colors(theValues[i]))
        
    rects1 = ax.bar(ind, theValues, width, color=cl)

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(theKeys, rotation=xLabelRotation)

    # ax.legend((rects1[0]), ('MIAS'))
    if show:
        plt.show()
        
    fig.savefig(filename, dpi=fig.dpi)
    
    return plt

def mias_convert_metadata(mias, destPath):

    masses = ["CIRC", "SPIC", "MISC"]
    with open(destPath, 'wb') as destfile:
        destCsv = csv.writer(destfile)

        head = ["Name","Type","AbType","Scanner","SubFolder","Pathology","LesionType"]
        destCsv.writerow(head)

        for k in mias.keys():
            row = []
            row.append(k + ".png")
            row.append("")
            row.append("")
            row.append("")
            row.append("")

            # Pathology
            if len(mias[k]) > 1:
                if mias[k][1] == "M":
                    row.append("MALIGNANT")
                else:
                    row.append("BENIGN")
            else:
                row.append("")

            # Lesion Type
            if mias[k][0] == "CALC":
                row.append("CALCIFICATION")
            elif mias[k][0] in masses:
                row.append("MASS")
            else:
                row.append("")

            # Write the row to the CSV file
            destCsv.writerow(row)


def main():
    """Our main function."""

    action = "load"

    #compareFolders()
    #compareLegendAndFiles()

    #dataFile = "C:\Code\Data\DATA698-ResearchProj\Ddsm.csv"
    #imgPath = "C:\Code\Data\DATA698-ResearchProj\ddsm-sm"
    dataFile = "C:\Users\Dan\Dropbox (DATA698-S17)\DATA698-S17\data\ddsm\png\Ddsm_png.csv"
    imgPath = "C:\Users\Dan\Dropbox (DATA698-S17)\DATA698-S17\data\ddsm\png"
    outDir = "C:\Code\Other\Breast_Cancer\data"
    valCsv = os.path.join(outDir, "ddsm_val.csv")
    testCsv = os.path.join(outDir, "ddsm_test.csv")
    trainCsv = os.path.join(outDir, "ddsm_train.csv")

    if(action == "load"):
        bc.load_training_metadata(dataFile, True, verbose=True, normalVsAbnormal=True)

        X_data, Y_data = bc.load_data(dataFile,
                                      imgPath,
                                      maxData = 1000,
                                      imgResize = (150, 150),
                                      verboseFreq = 25)

        print X_data.shape

        bc.to_categorical(Y_data)
    elif action == "split":

        bc.splitTrainTestValSets(dataFile, valCsv, testCsv, trainCsv)

    elif action == "cleanDataSets":

        bc.cleanDataSet(valCsv, imgPath)
        bc.cleanDataSet(testCsv, imgPath)
        bc.cleanDataSet(trainCsv, imgPath)

    elif action == "splitDist":
        a, b, valData = bc.load_training_metadata(valCsv)
        barChart(valData, filename="../../figures/ddsm_val_dist.png", title="DDSN Validation Set Distribution")

        a, b, testData = bc.load_training_metadata(testCsv)
        barChart(testData, filename="../../figures/ddsm_test_dist.png", title="DDSN Test Set Distribution")

        a, b, trainData = bc.load_training_metadata(trainCsv)
        barChart(trainData, filename="../../figures/ddsm_train_dist.png", title="DDSN Training Set Distribution")

    elif action == "mias-labels":
        miasCsv = "C:\Users\Dan\Dropbox (DATA698-S17)\DATA698-S17\data\mias\README"
        destCsv = "C:\Code\Other\Breast_Cancer\data\mias_all.csv"
        mias = bc.load_mias_labeldata(miasCsv)
        mias_convert_metadata(mias, destCsv)

    print "Done"


# This is the main of the program.
if __name__ == "__main__":
    main()