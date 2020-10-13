# OSIC_Pulmonay_Fibrosis_Progression_Kaggle2020
We present here our Kaggle project for the [OSIC Pulmonary Fibrosis Progression competition](https://www.kaggle.com/c/osic-pulmonary-fibrosis-progression).
With this code we ranked 351th in the final ranking (top 17%) with team name Zariski Topology. 
## Introduction
The task of the challenge was to predict the decline during a period of time in lung function measured as FVC, based on CT scans and some other metadata on the patients. In this repository we will provide most of the code we us and analyze our results. All the original data and additional information on the competition can be found [here](https://www.kaggle.com/c/osic-pulmonary-fibrosis-progression).

The main part of our solution can be summarized in the following points:
* Robust pipeline for preprocessing and feature extraction of CT scans.
* Fully connected network for prediction of the FVC values.
* Linear prediction for the confidence value

As a first step of our solution we built a robust pipeline for preprocessing and feature extraction for the CT scans. For the feature extraction part we used the VGG16 network of pytorch

## Preprocessing
