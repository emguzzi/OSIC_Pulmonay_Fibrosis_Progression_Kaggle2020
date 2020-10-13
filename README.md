# OSIC_Pulmonay_Fibrosis_Progression_Kaggle2020
We present here our Kaggle project for the [OSIC Pulmonary Fibrosis Progression competition](https://www.kaggle.com/c/osic-pulmonary-fibrosis-progression).
With this code we ranked 351th in the final ranking (top 17%) with team name Zariski Topology. 

## Introduction
The task of the challenge was to predict the decline during a period of time in lung function measured as FVC.  Based on CT scans, metadata on the patients and FVC measurements for past weeks, we predicted the value of the FVC at each future week together with a confidence value. The goodness of the prediction was then assesed using the [Laplacian log likelihood](https://www.kaggle.com/c/osic-pulmonary-fibrosis-progression/overview/evaluation). In this repository we will provide most of the code we us and analyze our results. All the original data and additional information on the competition can be found [here](https://www.kaggle.com/c/osic-pulmonary-fibrosis-progression).

The main part of our solution can be summarized in the following points: 
* Robust pipeline for preprocessing and feature extraction of CT scans.
* Fully connected network for prediction of the FVC values & linear model for confidence value.
* Fine-tuning of the model.

## Preprocessing & Feature Extraction
As a first step of our solution we built a robust pipeline for preprocessing and feature extraction for the CT scans. For the feature extraction part we used the VGG16 network of pytorch. The main difficulties for this part was the robustness of the pipeline; in particular dealing with corrupted dicom files and different format/size (see Figure §§§§§) of scans was fundamental. Since we mainly used the HPC cluster Euler from ETH and the notebooks from Kaggle to train our model some robustness of the code was also need to run on all the platforms.

For the preprocessing of the images we extracted the Hounsfield units (HU) from the dicom files. Setting a threshold on these units allowed us to consider the parts of the scan containing only air, this helped us to isolate the lungs from the rest of the body (see Figures §§§§§). After resizing and reformatting the thresholded images we fed them into the VGG16 network for the feature extranction.

As a result of this step we obtained 25088 dimensional feature vector. Since the images were very similar it was reasonable to expect very similar feature vector, indeed most of the feature had 0 variance (see Figure §§§§§). Therefore we set a threshold of 1 and kept only those feature with variance greater or equal than the threshold. A problem with this approach is the fact that the main differences between the images arise from the position of the scan, *i.e.* CT scans around the throat looks very different from CT scans at the center of the lungs (see Figure §§§§§ and Figure §§§§§). Therefore, we refrained from increasing the threshold even further, since we did not wanted the position of the CT scan to be the dominant feature obtained via this feature extraction procedure.

## Model

## Fine tuning 



