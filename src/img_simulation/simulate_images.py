#!/usr/bin/env python

from subprocess import call
import os
import random
from random import randint
import time

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
     call(["convert", "-size", str(height) + "x" + str(width), "xc:" + bgColor, "-fill", fillColor, "-stroke", strokeColor, "-draw", "arc " + str(x) + "," + str(y)+ "," + str(ex)+ "," + str(ey)+ "," + str(sd)+ "," + str(ed), name])

# The function below enhances the breast image with randomly generated graphics o
def fillImage(image_name, left_x, left_y, right_x, right_y):
    pointLoc_x = (left_x + right_x)/2 # Mid-point x coordinate
    pointLoc_y = (left_y + right_y)/2 # Mid-point y coordinate

    for num in range(1,100):
        pointRand_x = randint(0, 1)
        pointRand_y = randint(0, 1)
        if pointRand_x == 1:
            pointLoc_x = pointLoc_x + randint(0, 2)
        else:
            pointLoc_x = pointLoc_x - randint(0, 2)
        if pointRand_y == 1:
            pointLoc_y = pointLoc_y + randint(0, 2)
        else:
            pointLoc_y = pointLoc_y - randint(0, 2)
        call(["convert", image_name, "-strokewidth", str(1), "-fill", "rgba( 150, 150, 150 , 0.5 )", "-draw", "point     " + str(pointLoc_x) + "," + str(pointLoc_y), image_name])


# BEGIN SIMULATED IMAGE GENERATION:

#Replace the directory in the call below with your working directory. This is the folder to which the images will be written.
os.chdir('/Users/burton/simulated_images')
imageHeight = 50 # Height of the image to be created
imageWidth = 50 # Width of the image to be created. This will normally be equal to the height.
image_left_Margin_percentage = 10 # The percentage of the image that constitutes the left margin
image_right_Margin_percentage = 10 # The percentage of the image that constitutes the right margin
image_top_Margin_percentage = 10 # The percentage of the image that constitutes the top margin
image_bottom_Margin_percentage = 10 # The percentage of the image that constitutes the bottom margin

image_variation_percentage = 20

image_leftMargin = imageWidth*image_left_Margin_percentage/100
image_rightMargin = imageWidth*image_right_Margin_percentage/100
image_topMargin = imageWidth*image_top_Margin_percentage/100
image_bottomMargin = imageWidth*image_bottom_Margin_percentage/100

# Number of images to generate.
noImages = 1000;
imageNameBase = "breast_"
imageExtension = ".png"

t1 = time.ctime()
print(t1)
for num in range(1,noImages+1):
     # Randomly create select a box to draw the breast
    image_name = imageNameBase + str(num) + imageExtension
    left_x = image_leftMargin - image_leftMargin * random.uniform(0, 1)
    left_y = image_leftMargin - image_leftMargin * random.uniform(0, 1)
    right_x = 2*(imageWidth - image_leftMargin - image_leftMargin * random.uniform(0, 1))
    right_y = imageHeight - image_leftMargin - image_leftMargin * random.uniform(0, 1)
    angleRand = randint(0, 1)
    if angleRand == 1:
        start_degree = 90 - 10*random.uniform(0, 1)
        end_degree = 270 - 20 * random.uniform(0, 1)
    else:
        start_degree = 90 + 10 * random.uniform(0, 1)
        end_degree = 270 + 20 * random.uniform(0, 1)

    createImage(imageHeight, imageWidth, "black", "rgb(179, 179, 179)", "gray", left_x, left_y, right_x, right_y, start_degree , end_degree, image_name)
    fillImage(image_name, left_x, left_y, right_x, right_y)

# print(time.ctime())
print(t1)
t2 = time.ctime()
print(t2)


