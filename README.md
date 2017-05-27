# Differentiating Normal, Benign and Malignant Masses in Mammograms with Deep Learning

Repository for CUNY M.Sc. Data Analytics Capstone Project on Breast Cancer detection and diagnosis.

## Overview

This repository contains source code, results and literature related to our research project investigating breast cancer detection and diagnosis using whole image mammograms.

![Confusion Matrix Results for Pretrained CNN using Differenced Data](https://raw.githubusercontent.com/jnarhan/Breast_Cancer/master/figures/jn_Transfer_Detection_CM_20170526.png)

## Abstract

Mammograms are arguably the gold standard in visually screening for breast cancer. 
Motivated by the ability of convolutional neural networks to classify images in a wide spectrum of fields, 
this study presents a pre-processing treatment and neural network architecture for automated detection and 
classification of lesions using full mammogram images. The process leverages thresholding and image differencing 
to improve on detection and classification performance. For detection we expose the differenced images to a 
pre-trained neural network for feature extraction and then train a smaller network on these features for 
classification. For pathology classification of abnormalities, we train a fully connected CNN on the raw images. 
Given the limited mammography data we use dropout and kernel normalization to control overfitting.  Experimental 
results on two public dataset, DDSM and MIAS achieved state-of-the-art results in detection of lesions.


## Data Sets

We used the following publicly available data sets with our research:

[Digital Database of Screening Mammography (DDSM)](http://marathon.csee.usf.edu/Mammography/Database.html)

[Mammographic Image Analysis Society (MIAS)](http://peipa.essex.ac.uk/info/mias.html)

## References

[DDSM Mammography Software](http://marathon.csee.usf.edu/Mammography/software/heathusf_v1.1.0.html)

[Image Registration for Breast Imaging: A Review](https://www.ncbi.nlm.nih.gov/pubmed/17280947)

[Discrimination of Breast Cancer with Microcalcifications on Mammography by Deep Learning](http://www.nature.com/articles/srep27327)

[Representation learning for mammography mass lesion classification with convolutional neural networks](http://www.sciencedirect.com/science/article/pii/S0169260715300110)

...More to come...