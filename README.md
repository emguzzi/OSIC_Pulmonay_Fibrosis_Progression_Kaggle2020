# OSIC_Pulmonay_Fibrosis_Progression_Kaggle2020
We present here our Kaggle project for the [OSIC Pulmonary Fibrosis Progression competition](https://www.kaggle.com/c/osic-pulmonary-fibrosis-progression).
With this code we ranked 351th in the final ranking (top 17%) with team name Zariski Topology. 

## Introduction & Task Description
The task of the challenge was to predict the decline during a period of time in lung function measured as FVC.  Based on CT scans, metadata on the patients and FVC measurements for past weeks, we predicted the value of the FVC at each future week together with a confidence value. The goodness of the prediction was then assesed using the [Laplacian log likelihood](https://www.kaggle.com/c/osic-pulmonary-fibrosis-progression/overview/evaluation).
§§§ opposition with quantile regression§§§
Our model used a fully connected network to predict the slope of a linear decay for the FVC value of each patient, and we predicted a value of the confidence linearly increasing with the number of the weeks.
In this repository we will provide most of the code we us and analyze our results. All the original data and additional information on the competition can be found [here](https://www.kaggle.com/c/osic-pulmonary-fibrosis-progression).

The main part of our solution can be summarized in the following points: 
* Robust pipeline for preprocessing and feature extraction of CT scans.
* Fully connected network for prediction of the FVC values & linear model for confidence value.
* Fine-tuning of the model.

## Preprocessing & Feature Extraction
As a first step of our solution we built a robust pipeline for preprocessing and feature extraction for the CT scans. For the feature extraction part we used the VGG16 network of pytorch. The main difficulties for this part was the robustness of the pipeline; in particular dealing with corrupted dicom files and different format/size (see Figure §§§§§) of scans was fundamental. Since we mainly used the HPC cluster Euler from ETH and the notebooks from Kaggle to train our model some robustness of the code was also need to run on all the platforms.

For the preprocessing of the images we extracted the Hounsfield units (HU) from the dicom files. Setting a threshold on these units allowed us to consider the parts of the scan containing only air, this helped us to isolate the lungs from the rest of the body (see Figures §§§§§). After resizing and reformatting the thresholded images we fed them into the VGG16 network for the feature extranction.

As a result of this step we obtained 25088 dimensional feature vector. Since the images were very similar it was reasonable to expect very similar feature vector, indeed most of the feature had 0 variance (see Figure §§§§§). Therefore we set a threshold of 1 and kept only those feature with variance greater or equal than the threshold. A problem with this approach is the fact that the main differences between the images arise from the position of the scan, *i.e.* CT scans around the throat looks very different from CT scans at the center of the lungs (see Figure §§§§§ and Figure §§§§§). Therefore, we refrained from increasing the threshold even further, since we did not want the position of the CT scan to be the dominant feature obtained via this feature extraction procedure.
TODO (?) few words on the preprocessing of tabular data
## Model
After some data exploration, we decided to model the FVC decay with a linear function associating to `weeks` the `FVC` values and passing though `(base_week,base_FVC)`. This decision was taken for a series of observation, between which a significan better performance that other models -- such as polynomial regression (in various degrees) and log-decay. We fed a regression neural network (see below) with the data obtained by the above preprocessing with one final node. The output of the NN was expected to be the slope of the linear function for the given patient. So, concisely, the NN was used to obtain a personalised decay slope for each patient based on the tablular data and CT features. We optimised the network using a MSE loss.

As the competition was asking also for a confidence value, we adopeted a similar linear assumption for this parameter. In particular we assumed a minimal confidence value for the measurement closest to the CT scan (i.e. at `base_week`) and a linear increase with `abs(week - base_week)`. The parameters of this model (i.e. slope and minimal value at `base_week`) were optimised with a grid search. We also tried to make these parameters part of the NN output and optimise with respect to the true metric of the competition. However, the optimiser was not able to bring the values out of small neighbourhood around local optimal values which were not global optimal values.

## Fine tuning 
Due to time constraints our fine tuning was a bit limited. However, in this section, we will illustrate the main parameters that we considered for fine tuning. 

To start with, the parameters with the strongest effect on the score were the bias (value at the base week) and slope of the confidence line. This was expected since, due to the nature of the used metric (Laplacian log likelihood), the confidence value has a strong effect on the resulting score and therefore an accurate tuning was fundamental.

The remaining part of our fine tuning process was devoted to finding the best architecture, in particular number of layers and neurons, and attributes, in particular number of epochs and activation functions, for the network. The results of this process can be found in §§§§§. Intrestingly we discovered that shallower networks, *i.e.* from 1 to 3 fully connected layers, failed to extract information from the data and, as a result, they always predicted the value of the base week as a constant value.

## Conclusion
