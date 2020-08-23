---
layout: post
title:  "Review: Classification of Early- and Late- Mild Cognitive Impairment using 2D-CNN"
date:   2019-09-02 15:00:00 +0800
categories: CNN 2D-CNN sMRI MCI
---


In this blog, the paper [A Deep Learning approach for Diagnosis of Mild Cognitive Impairment Based on MRI Images](https://www.mdpi.com/2076-3425/9/9/217) is reviewed. This paper was published on [*Brain Sciences*](https://www.mdpi.com/journal/brainsci) in 2019.

The authors trained several binary classifiers using 5-layers-2D-CNN (3 convolutional layers and 2 fully connected layer) to classify early mild cognitive impairment (EMCI), late MCI (LMCI), and normal control (NC, denoted "CN" in the paper). That is, 3 classifiers for distinguishing EMCI-LMCI, EMCI-NC, and LMCI-NC.

<br/>
<br/>




## 1. Background

> There is no clear differentiation between the brain structure of healthy people and MCI patients, especially in the EMCI stage. 

> An early diagnosis at the stage of MCI can be very helpful in identifying people who are at risk of AD.

> These two groups are discriminated from each other based on the degree of memory impairment. In the EMCI patients, the decline in memory is approximately between 1.0–1.5 standard deviations (SD) below the normative mean, while in LMCI, the decline in memory is at least approximately 1.5 SD below the normative mean.

> Due to the similarities between the normal aging and MCI patients’ brain structures, a diagnosis of the MCI stage based on MRI and the discrimination between these two groups, mainly between EMCI and normal aging, is one of the most challenging parts of aging research.

<br/>
<br/>




## 2. Methods and Materials

# 2.1. Structural MRI Data

* **Data source**: [*Alzheimer’s disease Neuroimaging Initiative (ADNI) database*](http://www.loni.ucla.edu/ADNI).

* **Data type**: structural MRI

* **Number of subjects**: 200 EMCI, 200 LMCI patients, and 200 CN individuals

![fig1]({{site.baseurl}}/assets/190901_MCI_CNN/img/fig1.png)
Figure 1. The subjects’ clinical and demographic characteristics. For each group, N represents the total number of subjects, M and F show number of males and females, along with the average age, standard deviation (SD) and average mini-mental state examination (MMSE) score.
{: style="text-align: center; color: gray"}

<br/>


# 2.2. Preprocessing of Structural MRI

* **Toolbox**: [*SPM12*](https://www.fil.ion.ucl.ac.uk/spm/software/spm12/)

* **Segmentation**: segment the brain tissue into GM, WM, and CSF
> This paper set the bias regularization on the light regularization (0.001), the full width at half maximum (FWHM) of Gaussian smoothness of bias on the 60 mm cutoff, and affine regularization on the ICBM space template. Moreover, for the spatial normalization of the data to the Montreal Neurological Institute (MNI) spaces, the deformation field was set in the forwarding mode.

* **Normalization of GM images**: only use GM images for further analysis.
> For normalizing all GM images to MNI space, this paper set the written normalized images voxel size on (2 2 2) mm and, for sampling the images to MNI space, the 4th-degree B-Spline for interpolation was considered.

* **Smoothing of GM images**: GM images were smoothed by a Gaussian kernel (FWHM = [2 2 2] mm). 

* **Resize**: from 176 × 240 × 256 to 79 × 95 × 79

* **Decompose 3D to 2D**: 
> 3D images were decomposed into 2D slices along the third direction, which named axial, coronal, and sagittal views. 

* **Convert to PNG and resize**:
> All the 2D.nii GM files were converted to a portable network graphics (PNG) format and resized to 64 × 64

* **Discard useless slices for each view**: only retain 20 slices for each view (60 2D images for each subject)

<br/>


# 2.3. CNN Architecture

![fig2]({{site.baseurl}}/assets/190901_MCI_CNN/img/fig2.png)
Figure 2. CNN Architecture
{: style="text-align: center; color: gray"}

* 3 convolutional layers followed by ReLu and max-pooling layer 
	* 1st: 32 channels
	* 2nd: 128 channels
	* 3rd: 512 channels

* 1 fully connected layer: 128 kernels (neurons) with ReLu activation

* 1 neuron with sigmoid activation

<br/>


# 2.4. Training Details

**Train classifier for each pair of group (3 pairs) and anatomical view (3 views), that is, 9 classifiers**

* **Initializer**: glorot uniform initializer 

* **Optimization**: Adam

* **Loss function**: binary-cross-entropy

* **Regularization**: regularization for weights and bias, and [**dropout**](https://arxiv.org/abs/1307.1493) were employed

* **Image Augmentation**: sheering, random rotation, zooming (not explained in details)

* **Training and Testing Split**: 70% (280 subjects) for training, 30% (120 subjects) for testing

<br/>


# 2.5. Performance Measurements

* **Accuracy**: 

$$ Accuracy = (TP+TN) / (TP+TN+FP+FN)$$

* **Sensitivity (Recall)**: 

$$ R = Recall = Sensitivity = TP / (TP+FN)$$

* **Specificity**: 

$$ Specificity = TN / (TN+FP)$$
	
* **F-score**: 

$$P = Precision = TP / (TP+FP)$$

$$F-score = 2* (P+R) / (P*R)$$

* **AUC-ROC**

<br/>
<br/>


## 3. Results

![fig3]({{site.baseurl}}/assets/190901_MCI_CNN/img/fig3.png)
Figure 3. The classification results of the control normal (CN) versus early mild cognitive impairment (EMCI), CN versus late mild cognitive impairment (LMCI) and EMCI versus LMCI.
{: style="text-align: center; color: gray"}

![fig4]({{site.baseurl}}/assets/190901_MCI_CNN/img/fig4.png)
Figure 4. Receiver operating characteristic-area under the curve (ROC-AUC) results of the sagittal, coronal, and axial views.
{: style="text-align: center; color: gray"}



## 4. Conclusions
> The proposed method for feature extraction and classification delivered a high accuracy for the EMCI, LMCI, and CN groups.

> The best results were achieved for the classification between CN and LMCI groups in the sagittal view and also, the pairs of EMCI/LMCI have achieved slightly better accuracy than CN/EMCI concerning all views of the MRI.





















