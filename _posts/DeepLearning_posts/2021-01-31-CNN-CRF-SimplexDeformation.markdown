---
layout: post
title: "Review: Segmentation of 12 Knee Joint Anatomies Using CNN, Conditional Random Field, and Simplex Deformation"
date:  2021-01-31 22:00:00 +0800
categories: deeplearning
---

In this blog, the paper [Deep convolutional neural network for segmentation of knee joint anatomy ](https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.27229) is reviewed. It was published on **Magnetic Resonance in Medicine** (2018).

In this paper, the author proposed a segmentation pipeline which consists of a **semantic segmentation CNN**, **3D fully connected conditional random field (CRF)**, and **3D simplex deformable modeling**.
* A convolutional encoder‐decoder network was designed as the core of the segmentation method to perform high resolution pixel‐wise multi‐class tissue classification for 12 different joint structures. 
* The 3D fully connected CRF was applied to regularize contextual relationship among voxels within the same tissue class and between different classes.
* The 3D simplex deformable modeling refined the output from 3D CRF to preserve the overall shape and maintain a desirable smooth surface for joint structures.

## 1. Background

> Quantitative measures of joint degeneration have also been used in OA research studies and have the advantages of being objective and highly reproducible with a greater dynamic range for assessing tissue degeneration than semi-quantitative grading scales. 
>
> Segmentation of musculoskeletal tissues is the crucial first step in the processing pipeline to acquire quantitative measures of joint degeneration from MR images.
>
> CNN approaches have shown promising results for segmenting cartilage and bone. However, the adaptation of CNN methods for rapid and accurate segmentation of all joint structures that may be sources of pain in patients with OA has yet to be investigated.

## 2. Methods

### 2.1 Pipeline

![Fib1]({{site.baseurl}}/assets/210131_CNN_CRF_SimplexDeformation/img/fig2.png)
**Figure 1:** Flowchart of the fully automated musculoskeletal tissue segmentation method. The well-trained CED network is used to segment the testing images and to generate tissue class probabilities for each pixel. The CRF model is further applied to promote the label assignment for voxels. The processed labels representing **cartilage and bone** are discretized by using the marching cube algorithm and are sent to the 3D simplex deformable process. In the deformable process, each individual target is refined to preserve smooth tissue boundary and to maintain overall anatomical geometry
{: style="text-align: center; color: gray"}

* **the convolutional encoder-decoder (CED) network:** 2D segmentation network for generating tissue class probabilities for each voxel
* **the fully connected 3D CRF model:** refine the overall segmentation results by improving the label assignment for voxels with similar contrast and taking into account the 3D contextual relationships
* **the 3D simplex deformable modeling:** refine the bone and cartilage segmentations to preserve smooth tissue boundary and to maintain overall anatomical geometry

### 2.2 CED Network Architecture

![Fib2]({{site.baseurl}}/assets/210131_CNN_CRF_SimplexDeformation/img/fig1.png) 
**Figure 2:** Illustration of the CNN architecture. All convolutional filters use 3 3 3 filter size and the max-pooling layers use a 2 3 2 window and stride 2. All the input 2D images from the training data set were first cropped to enclose as much of the knee joint as possible while removing excessive image background. The images were then resampled to 320 3 224 matrix size using bilinear interpolation to match the fixed input size of the CED network
{: style="text-align: center; color: gray"}

* adapt VGG16 as encoder
* mirror the encoder structure to the decoder, yet replace the max-pooling by up-sampling
* add softmax layer to the end of decoder
* add shortcut connections based on pre-activation scheme in the residual network (Ref: [Identity Mappings in Deep Residual Networks (ECCV)](https://link.springer.com/chapter/10.1007/978-3-319-46493-0_38).)

### 2.3 Fully Connected 3D CRF

**CRF reference papers:**
* [Multiscale conditional random fields for image labeling](https://ieeexplore.ieee.org/abstract/document/1315232)
* [Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials](https://arxiv.org/abs/1210.5644)

**Why 3D CRF:** Although a 2D CNN is highly efficient in processing high in-plane resolution images, the inter-slice contextual information might not be fully handled. Therefore, irregular labels such as holes and small isolated objects are likely to be generated in regions with ambiguous image contrast.

### 2.4 3D Simplex Deformable Model for Bone and Cartilage

**Simplex Deformable Model reference papers:**
* [General object reconstruction based on simplex meshes](https://link.springer.com/article/10.1023/A:1008157432188)

**Why Simplex Deformable Model:** A smooth and well‐defined boundary is desirable for cartilage and bone. Therefore, 3D deformable modeling is implemented for cartilage and bone refinement. 

### 2.5 Implementation Details

*  Image Preprocessing
   * central cropping: all the input 2D images were first cropped to enclose as much of the knee joint as possible while removing excessive image background
   * resampling: the images were then resampled to 320 × 224 matrix size using bilinear interpolation
   * image augmentation: translation, shearing, rotation
*  Transfer Learning
   * pretrain the CDE network using 60 sagittal 3D T1-weighted spoiled gradient recalled-echo (3D-SPGR) knee image data sets from SKI10
*  Optimizer: Adam (fixed LR=0.001)
*  batch size: 10
*  20-fold leave-one-out-cross-validation

## 3. Dataset

* 20 subjects with knee OA (12 males & 8 females with an average age of 58 years)
* MR sequence: 3D fast spin-echo

## 4. Evaluation of Segmentation Accuracy

> Note, the VOE and VD values were calculated within a ROI that was drawn in each of 3 consecutive central slices on the medial and lateral tibial plateau, medial and lateral femoral condyles, and patella on the 3D-FSE image data sets.
>
> To compare the performance with networks using 3D convolutional filters, 2 state-of-the-art 3D CNNs, **deep-Medic** and **V-Net,** were adapted in the current study for simultaneously segmenting all knee joint tissue structures.

* Dice Coefficient (DC): 
  $$\mathrm{DC}=\frac{2|S \cap R|}{|S|+|R|}$$
* Volumetric Overlap Error (VOE): 
  $$\mathrm{VOE}=100\left(1-\frac{|S \cap R|}{|S \cup R|}\right)$$
* Volumetric Difference (VD):  
  $$\mathrm{VD}=100 \frac{|S|-|R|}{|R|}$$
* Average Symmetric Surface Distance (ASSD): 
  $$\operatorname{ASSD}=\frac{\sum_{s \in \partial(S)} \min _{r \in \partial(R)}\|s-r\|+\sum_{r \in \partial(R)} \min _{s \in \partial(S)}\|r-s\|}{|\partial(S)|+|\partial(R)|}$$

## 5. Results

![Fib3]({{site.baseurl}}/assets/210131_CNN_CRF_SimplexDeformation/img/fig3.png)
**Figure 3:** Bar plot (mean and SD) of the Dice coefficient and average symmetric surface distance (ASSD) values for each individual segmented joint structure in the 20 subjects for CED, combination of CED and CRF, the combination of CED, CRF and deformable process, deepMedic and V-Net.
{: style="text-align: center; color: gray"}

* DC > 0.9: 
  * femur (mean ± SD: 0.970 ± 0.010)
  * tibia (0.962 ± 0.015)
  * muscle (0.932 ± 0.024)
  * other non‐specified tissues (0.913 ± 0.017)
* 0.9 > DC > 0.8:
  * femoral cartilage (0.806 ± 0.062)
  * tibial cartilage (0.801 ± 0.052)
  * patella (0.898 ± 0.033)
  * patellar cartilage (0.807 ± 0.101)
  * meniscus (0.831 ± 0.031)
  * quadriceps and patellar tendon (0.815 ± 0.029)
  * infrapatellar fat pad (0.882 ± 0.040)
* 0.8 > DC > 0.7:
  * joint effusion and Baker's cyst (0.736 ± 0.069)

![Fib4]({{site.baseurl}}/assets/210131_CNN_CRF_SimplexDeformation/img/fig4.png)
**Figure 4:** Examples of tissue segmentation performed on the 3D-FSE images in 2 subjects with knee OA using the CED network only, the CED network combined with 3D fully connected CRF, and the CED network combined with both CRF and 3D deformable modeling. 
{: style="text-align: center; color: gray"}

