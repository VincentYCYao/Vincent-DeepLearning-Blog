---
layout: post
title: "Review: Knee Cartilage Segmentation Using a Triplanar Convolutional Neural Network"
date:  2021-01-25 19:30:00 +0800
categories: deeplearning
---

In this blog, the paper [Deep Feature Learning for Knee Cartilage Segmentation Using a Triplanar Convolutional Neural Network](https://link.springer.com/chapter/10.1007/978-3-642-40763-5_31) is reviewed. This paper was published on **MICCAI 2013** (Medical Image Computing and Computer-Assisted Intervention).

In this study, the authors proposed a **triplanar CNN** for knee cartilage segmentation from 3D MRI, yet just using 2D convolution layer. Although the proposed method uses only 2D features at a single scale, it performs better than a state-of-the-art method at that time using 3D multi-scale features.

This is one of the early explorations of the applications of CNN on knee cartilage segmentation. We highlight the proposed network architecture in this blog.

## 1. Background

> Convolutional neural networks (CNNs) are deep learning architectures and have recently been employed successfully for 2D image segmentation tasks. Here we use them for voxel classification in 3D images. There is a generic extension of CNNs from 2D to 3D images. However, these truly 3D CNNs have large memory and training time requirements, retarding their application for time-constrained large-scale medical imaging tasks.
>
> In this paper we propose a classifier based on a system of triplanar 2D CNNs which classifies voxels from 3D images with high accuracy and is easy to train.

## 2. Triplanar CNN

**Input of the CNN:** three 2D patches centered around a voxel, extracted from each of the three planes

**Output of the CNN:** probability map

![fig1]({{site.baseurl}}/assets/210125_TriplanarCNN//img/fig1.png)
**Figure 1.** The proposed CNN architecture.
{: style="text-align: center; color: gray"}

![fig2]({{site.baseurl}}/assets/210125_TriplanarCNN//img/fig2.png)
**Figure 2.** The three images planes giving rise to our triplanar convolutional neural network (CNN) architecture. One patch centered in the voxel is extracted from each of the planes. The three CNNs are fused in the final layer.
{: style="text-align: center; color: gray"}

**Note:** For each voxel, three 2D patches centered around itself are extracted. Each 2D patch image are the input to a CNN as shown in Figure 1. The last layer of the proposed CNN for each patch in each of the three planes are concatenated together, followed by a softmax layer.

## 3. Result

| Method               | Over 114 Scans | DSC        | Accuracy | Accuracy | Specificity |
| -------------------- | -------------- | ---------- | -------- | -------- | ----------- |
| Triplanar CNN        | Mean           | **0.8249** | 99.93%   | 81.92%   | 99.97%      |
| State-of-the-art [5] | Mean           | 0.8135     | 99.92%   | 80.52%   | 99.96%      |

[5] [Segmenting Articular Cartilage Automatically Using a Voxel Classification Approach](https://ieeexplore.ieee.org/document/4039531)







