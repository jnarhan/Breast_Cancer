# -*- coding: utf-8 -*-

# Authors : Jay Narhan and Youqing Xiang

# Using the original images

import os
import cv2
import copy
import numpy as np
from matplotlib import pyplot as plt


IMG_DIR = raw_input('Give the file direction of images to be transformed: ') 

filenames = [ filename for filename in os.listdir(IMG_DIR) if filename.endswith('.png')]
os.chdir(IMG_DIR)

def get_hists(image, b):
    hist, bins = np.histogram(img.flatten(), bins=b, range=[0,255])
    cdf = hist.cumsum()
    cdf_normalized = cdf *hist.max()/ cdf.max()
    
    return [hist, cdf_normalized]

def plot(img, img_hists):
    plt.figure(1)
    plt.subplot(121)
    plt.imshow(img, cmap='gray')
    
    plt.subplot(122)
    plt.plot(img_hists[1], color = 'b')
    plt.plot(img_hists[0], color = 'r')
    plt.xlim([0,256])
    plt.legend(('cdf','histogram'), loc = 'upper left')
    
    plt.subplots_adjust(top=0.92, bottom=0.08,
                        left=0.10, right=0.95,
                        hspace=0.25, wspace=0.35)
    plt.show()
    _ = raw_input("Press [enter] to continue.")
    plt.close()
                        
def mask(image, labels, region):
    labels = copy.deepcopy(labels)  
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            if labels[row, col] != region:
                labels[row, col] = 0  # mask the artifact
            else:
                labels[row, col] = 1  # retain the breast
    return labels

dirname = raw_input('Create the folder name for transformed images: ')

if not os.path.exists(dirname):
    os.makedirs(dirname)
    
for filename in filenames:
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    print 'The image to be transformed is {}'.format(filename)
    
    img_hists = get_hists( img, b=256)
    plot(img, img_hists)
    images_thresh =  cv2.threshold(img, np.median(img), 
                                        255, cv2.THRESH_BINARY)[1]
                                        
    plt.imshow(images_thresh, cmap='gray')
    plt.show()
    _ = raw_input("Press [enter] to continue.")
    plt.close()
    
    _, l, s, _ = cv2.connectedComponentsWithStats(images_thresh)
    
    regions = {'labels':l, 'count':s[:, -1]}
    my_mask = mask( images_thresh, regions['labels'], region=1)
    image = img * my_mask  # notice I'm using the CLAHE image, you could use the original
    plt.imshow(image, cmap='gray')
    plt.show()
    _ = raw_input("Press [enter] to continue.")
    plt.close()
    
    ANSWER = raw_input('Are you happy with the default transformed image, yes or no?' )
    if ANSWER[:1].upper() == 'Y':
        cv2.imwrite(os.path.join(dirname, filename), image)
        print 'The transformed image is saved.'
    else:
        while True:
            factor = float(raw_input('Give a factor for threhold: '))
            images_thresh =  cv2.threshold(img, (np.median(img))*factor, 
                                        255, cv2.THRESH_BINARY)[1]
                                        
            plt.imshow(images_thresh, cmap='gray')
            plt.show()
            _ = raw_input("Press [enter] to continue.")
            plt.close()
            _, l, s, _ = cv2.connectedComponentsWithStats(images_thresh)
            regions_new = {'labels':l, 'count':s[:, -1]}
            reg = int(raw_input('Choose the region, 1 or 0?'))
            my_mask = mask( images_thresh, regions_new['labels'], region=reg)
            image = img * my_mask  # notice I'm using the CLAHE image, you could use the original
            plt.imshow(image, cmap='gray')
            plt.show()
            _ = raw_input("Press [enter] to continue.")
            plt.close()
            
            ANSWER = raw_input('Are you happy with the transformed image now, yes or no?' )
            if ANSWER[:1].upper() == 'N':
                print 'Try again!'
            else:
                cv2.imwrite(os.path.join(dirname, filename), image)
                print 'The transformed image is saved.'
                break
          