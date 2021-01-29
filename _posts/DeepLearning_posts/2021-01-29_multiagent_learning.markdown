---
layout: post
title: "Review: Collaborative Multi-agent Learning for Knee Cartilage Segmentation"
date:  2021-01-29 22:00:00 +0800
categories: deeplearning
---

In this blog, the paper [Collaborative Multi-agent Learning for MR Knee Articular Cartilage Segmentation](https://link.springer.com/chapter/10.1007/978-3-030-32245-8_32) is reviewed. It was published on **MICCAI 2019** (Medical Image Computing and Computer Assisted Intervention).

In this paper, the author proposed a novel framework for knee cartilage segmentation. The key contribution is the **adversarial learning based collaborative multi-agent segmentation network** with features as follow:

* three parallel **segmentation agents** to segment cartilages in their respective region of interest (ROI) — one network for each of the Femoral Cartilage (FC), Tibial Cartilage (TC), and Patellar Cartilage (PC)

* a novel **ROI-fusion layer** to fuse the segmentations from the above three networks

* **collaborative learning** driven by an adversarial sub-network.

  > The ROI-fusion layer not only fuses the individual cartilages from multiple agents, but also backpropagates the training loss from the adversarial sub-network to each agent to enable joint learning of shape and spatial constraints

## 1. Background

> Effective and efficient segmentation of all articular cartilages in  high-resolution and large-sized data is challenging.

## 2. Framework

![Fib1]({{site.baseurl}}/assets/210129_multiagent_learning/img/fig1.png)
**Figure 1:** Overview of the multiple cartilage ROIs extraction (only show the sagittal view). The number of feature maps in the network is displayed under each block.
{: style="text-align: center; color: gray"}

The proposed framework consists of two parts as follow:
* Coarse cartilage segmentor for ROI extraction
* Collaborative multi-agent learning
  * segmentation agents for each structure (FC, TC, PC)
  * ROI-fusion layer
  * Joint-label discriminator

### 2.1 ROI Extraction

![Fib2]({{site.baseurl}}/assets/210129_multiagent_learning/img/fig2.png)
**Figure 2:** Overview of the multiple cartilage ROIs extraction (only show the sagittal view). The number of feature maps in the network is displayed under each block.
{: style="text-align: center; color: gray"}

* The segmentor's structure is like [VNet](https://ieeexplore.ieee.org/abstract/document/7785132)

  > All the convolutional layers in the residual blocks have filter size 3, stride 1 and zero-padding 1. PReLU activation and batch normalization follow the convolutional and deconvolutional layers. The coarse cartilage segmentor is trained based on multi-class cross entropy loss ℓ𝑚𝑐𝑒ℓmce to obtain cartilage masks from the down-sampled MR data (e.g., 192×192×160192×192×160)

### 2.2 Collaborative multi-agent learning

![Fib3]({{site.baseurl}}/assets/210129_multiagent_learning/img/fig3.png)
**Figure 3:** Demonstration of the collaborative multi-agent learning framework for fine- grained cartilage segmentation. The agents yield binary labels and the spatial fusion operation outputs a 4-channel result (FC, TC, PC and background). (Color figure online)
{: style="text-align: center; color: gray"}

#### 2.2.1 Segmentation Agents

* adapt VNet-like structure

* add **attention mechanism** to skip connections

  the connecting operation becomes $o\left(\alpha \odot I_{l}, I_{h}^{u p}\right)$, where $$o$$ denotes concatenation along the channel dimension, and ⊙ is element- wise multiplication

#### 2.2.2 ROI-fusion Layer



#### 2.2.3 Joint-label discriminator











