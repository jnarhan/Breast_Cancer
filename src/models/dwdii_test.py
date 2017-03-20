#
# Author: Daniel Dittenhafer
#
#     Created: Mar 27, 2016
#
# Description: Test script
#
__author__ = 'Daniel Dittenhafer'

import dwdii_bc_model_helper as bc


def main():
    """Our main function."""

    destPath = "C:\Code\Python\emotional-faces\data\\transformed"
    srcPath = "C:\Code\Python\emotional-faces\data\Resize"

    #labelFaces()

   #print "OpenCV version: " + cv2.__version__

    #transformFaces(srcPath, destPath)

    #compareFolders()
    #compareLegendAndFiles()

    #dataFile = "C:\Code\Data\DATA698-ResearchProj\Ddsm.csv"
    #imgPath = "C:\Code\Data\DATA698-ResearchProj\ddsm-sm"
    dataFile = "C:\Users\Dan\Dropbox (DATA698-S17)\DATA698-S17\data\ddsm\png\Ddsm_png.csv"
    imgPath = "C:\Users\Dan\Dropbox (DATA698-S17)\DATA698-S17\data\ddsm\png"

    bc.load_training_metadata(dataFile, True)

    X_data, Y_data = bc.load_data(dataFile,
                                  imgPath,
                                  maxData = 1000,
                                  imgResize = (150, 150),
                                  verboseFreq = 25)

    print X_data.shape

    bc.to_categorical(Y_data)

    print "Done"


# This is the main of the program.
if __name__ == "__main__":
    main()