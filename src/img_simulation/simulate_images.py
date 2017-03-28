#!/usr/bin/env python

from subprocess import call
import os
import random
from random import randint
import time
import math

# Set the global properties below:

NUM_OF_IMAGES = 25 # The number of images to generate
PERCENTAGE_OF_IMAGES_WITH_TUMOR = 40
IMAGE_BASE_NAME = "breast_" # The base name for the images.
IMAGE_EXTENSION = ".png" # The extension of the generated images.
TARGET_DIR = '/Users/burton/simulated_images' # The target folder for where images will be generated.
IMAGE_HEIGHT = 50 # Height of the generated images.
IMAGE_WIDTH = 50 # Width of the generated images.
LEFT_MARGIN_PCT = 5 # Left margin from where the breast will be painted in the image.
RIGHT_MARGIN_PCT = 5 # Right margin from where the breast will be painted in the image.
TOP_MARGIN_PCT = 5 # Top margin from where the breast will be painted in the image.
BOTTOM_MARGIN_PCT = 5 # Bottom margin from where the breast will be painted in the image.
START_DEGREE_OFFSET = 5
END_DEGREE_OFFSET = 10
NO_OF_DOTS_BASE_IMAGE = 50
NO_OF_DOTS_MALIGNANT = 200
NO_OF_DOTS_NON_MALIGNANT = 100

# The function below creates a single breast image.
# height: height of the resultant image
# width: width of the resultant image
# bgColor: background color of the image.
# fillColor: The fillColor for the arc.
# strokeColor: The outline color for the arc
# sx: Starting x ordinate of bounding rectangle
# sy: starting y ordinate of bounding rectangle
# ex: ending x ordinate of bounding rectangle
# ey: ending y ordinate of bounding rectangle
# sd: starting degrees of rotation
# ed: ending degrees of rotation
# name: name of the image file.
def createImage(height, width, bgColor, fillColor, strokeColor,  x, y, ex, ey, sd, ed, name ):
     call(["convert", "-size", str(height) + "x" + str(width), "xc:" + bgColor, "-fill", fillColor, "-stroke", strokeColor, "-draw", "arc " + str(x) + "," + str(y)+ "," + str(ex)+ "," + str(ey)+ "," + str(sd)+ "," + str(ed), "-define", "convolve:scale = 60, 40 %", "-morphology", "Convolve",  "Gaussian:0x1", name])
     # -define convolve:scale = 60, 40 % -morphologyConvolve 'Gaussian:0x3'

# The function below generates a shade of gray color to fill in the breast.
def generateRandomBreastColor():
    base = random.randint(140, 255) # Generate a random number in the gray range.
    return "rgba(" + str(base) + "," + str(base) + "," + str(base) + ")"

# The following function uses the ellipse function to decide if a point is within a breast.
def isPointInsideBreast(breast_x, breast_y, imageHeight, imageWidth):
    ellipseVal = (math.pow(breast_x, 2)/math.pow(imageWidth, 2) +  math.pow(breast_y, 2)/math.pow(imageHeight, 2))
    if(  ellipseVal  < 1):
        return True
    else:
        return False

def createNoise(image_name, left_x, left_y, right_x, right_y):
    #NO_OF_DOTS_BASE_IMAGE
    x = [random.uniform(left_x, right_x) for p in range(0, NO_OF_DOTS_BASE_IMAGE)]
    y = [random.uniform(left_y, right_y) for p in range(0, NO_OF_DOTS_BASE_IMAGE)]
    for i in range(1, NO_OF_DOTS_BASE_IMAGE):
        if isPointInsideBreast(x[i], y[i], right_y - left_y, right_x - left_x):
            call(["convert", image_name, "-strokewidth", str(1), "-fill", "rgba( 0, 0, 0 , 0.5 )", "-draw", "point     " + str(x[i]) + "," + str(y[i]), image_name])



# The function below enhances the breast image with randomly generated points.
def createTumor(image_name, left_x, left_y, right_x, right_y, malignant):
    pointLoc_x = (left_x + right_x)/2 # Mid-point x coordinate
    pointLoc_y = (left_y + right_y)/2 # Mid-point y coordinate
    if malignant:
        totalPoints = NO_OF_DOTS_MALIGNANT
    else:
        totalPoints = NO_OF_DOTS_NON_MALIGNANT

    for num in range(1, totalPoints):
        pointRand_x = randint(0, 1)
        pointRand_y = randint(0, 1)
        if pointRand_x == 1:
            pointLoc_x_new = pointLoc_x + random.uniform(0, 10)*random.uniform(0, 1)
        else:
            pointLoc_x_new = pointLoc_x - random.uniform(0, 10)*random.uniform(0, 1)
        if pointRand_y == 1:
            pointLoc_y_new = pointLoc_y + random.uniform(0, 10)*random.uniform(0, 1)
        else:
            pointLoc_y_new = pointLoc_y - random.uniform(0, 10)*random.uniform(0, 1)
        # print("Filling Image " + image_name + " : " + str(pointLoc_x) + " : " + str(pointLoc_y))
        call(["convert", image_name, "-strokewidth", str(1), "-fill", "rgba( 0, 0, 0 , 0.5 )", "-draw", "point     " + str(pointLoc_x_new) + "," + str(pointLoc_y_new), image_name])

# The following function is the main function which will call other required function to create the breast images.
# numOfImages : The number of simulated images to generate.
# pctImagesWithTumor: The percentage of images with tumor in addition to the number of images above.
# imageHeight : Height of the image to be created
# imageWidth : # Width of the image to be created. This will normally be equal to the height.
# leftMarginPct : # The percentage of the image that constitutes the left margin
# rightMarginPct : # The percentage of the image that constitutes the right margin
# topMarginPct : # The percentage of the image that constitutes the top margin
# bottomMarginPct : # The percentage of the image that constitutes the bottom margin
def createImages(numOfImages, pctImagesWithTumor, baseImageName, imageExt, targetDir, imageHeight, imageWidth, leftMarginPct, rightMarginPct, topMarginPct, bottomMarginPct, startDegOffset, endDegOffset):
    numOfImagesWithTumor = (int)((numOfImages/100.0)*pctImagesWithTumor)
    print("numOfImagesWithTumor : " + str(numOfImagesWithTumor))
    os.chdir(targetDir)
    f = open('out.csv', 'w')
    f.write("image_name" + "," + "label" + "\n")
    image_leftMargin = imageWidth*leftMarginPct/100
    image_rightMargin = imageWidth*rightMarginPct/100
    image_topMargin = imageWidth*topMarginPct/100
    image_bottomMargin = imageWidth*bottomMarginPct/100
    t1 = time.ctime()
    print(t1)
    totalNumberOfImages = numOfImages + numOfImagesWithTumor*2
    tumorImagesNotDone = True
    nonTumorImagesNotDone = True
    tumorImagesCount = 0
    for num in range(1, totalNumberOfImages + 1):
        # Randomly create select a box to draw the breast
        image_name = baseImageName + str(num) + imageExt
        left_x = image_leftMargin - image_leftMargin * random.uniform(0, 1)
        left_y = image_leftMargin - image_leftMargin * random.uniform(0, 1)
        right_x = imageWidth - image_rightMargin * random.uniform(0, 1)
        right_y = imageHeight - image_bottomMargin * random.uniform(0, 1)
        angleRand = randint(0, 1) # Angle of the image
        if angleRand == 1:
            start_degree = 90 - START_DEGREE_OFFSET * random.uniform(0, 1)
            end_degree = 270 - END_DEGREE_OFFSET * random.uniform(0, 1)
        else:
            start_degree = 90 + START_DEGREE_OFFSET * random.uniform(0, 1)
            end_degree = 270 + END_DEGREE_OFFSET * random.uniform(0, 1)
        print("Creating Image" + image_name + " : " + str(left_x) + " : " + str(left_y) + " : " + str(right_x) + " : " + str(right_y))
        createImage(imageHeight, imageWidth, "black", generateRandomBreastColor(), "gray", left_x, left_y, 2*right_x, right_y, start_degree, end_degree, image_name)

        createNoise(image_name, left_x, left_y, right_x, right_y)

        if tumorImagesNotDone:
            f.write(image_name + "," + "tumor" + "\n")
            createTumor(image_name, left_x, left_y, right_x, right_y, True)
            tumorImagesCount = tumorImagesCount + 1
            if(tumorImagesCount >= numOfImagesWithTumor):
                tumorImagesNotDone = False
                tumorImagesCount = 0
        elif nonTumorImagesNotDone:
            f.write(image_name + "," + "benign" + "\n")
            createTumor(image_name, left_x, left_y, right_x, right_y, False)
            tumorImagesCount = tumorImagesCount + 1
            if(tumorImagesCount >= numOfImagesWithTumor):
                nonTumorImagesNotDone = False
        else:
            f.write(image_name + "," + "no_tumor" + "\n")
    f.close()
    # print(time.ctime())
    print(t1)
    t2 = time.ctime()
    print(t2)

# ----------- START OF THE MAIN PROGRAM -----------.

createImages(NUM_OF_IMAGES, PERCENTAGE_OF_IMAGES_WITH_TUMOR,  IMAGE_BASE_NAME, IMAGE_EXTENSION, TARGET_DIR, IMAGE_HEIGHT, IMAGE_WIDTH, LEFT_MARGIN_PCT, RIGHT_MARGIN_PCT, TOP_MARGIN_PCT, BOTTOM_MARGIN_PCT, START_DEGREE_OFFSET, END_DEGREE_OFFSET)


