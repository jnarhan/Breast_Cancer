# Detection and Diagnosis of Breast Cancer Using Deep Learning

Repository for CUNY M.Sc. Data Analytics Capstone Project on Breast Cancer detection and diagnosis.

## Overview

This repository contains source code and results related to our research project investigating breast cancer detection and diagnosis using whole image mammograms.

Classification Metric      | Detection Score | Diagnosis Score
-------------------------- | --------------- | ---------------
Accuracy                   | 88.99%          | 78.15%
Sensitivity                | 90.76%          | 75.46%
Specificity                | 87.23%          | 80.86%
Positive Predictive Value  | 87.66%          | 79.87%
Negative Predictive Value  | 90.42%          | 76.61%
F1-Score                   | 0.89            | 0.78


## Abstract

Mammograms are arguably the gold standard in visually screening for breast cancer. However  variability in human tissue and the subtlety of abnormalities can challenge lesion identification. Motivated by the ability of convolutional neural networks to classify images in a wide spectrum of fields, this study presents a pre-processing treatment and neural network architecture for the automated detection of abnormalities (such as masses or micro-calcifications) and the pathology classification of identified lesions (i.e.using diagnosis as either benign or malignant). These task are performed using full mammogram images. The process leverages thresholding, image registration and differencing to improve on detection and diagnosis performance. For both classification objectives, we expose the differenced images to a pre-trained neural network for feature extraction and then train a smaller network on these features for classification as either normal or abnormal and where abnormalities exist, as either benign or malignant. Given the limited mammography data we use a variety of regularization techniques including dropout and kernel normalization to control overfitting.  Experimental results on two public dataset, DDSM and MIAS achieved state-of-the-art results in detection of lesions.


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