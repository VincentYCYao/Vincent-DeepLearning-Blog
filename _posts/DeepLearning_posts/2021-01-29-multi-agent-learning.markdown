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
  > All the convolutional layers in the residual blocks have filter size 3, stride 1 and zero-padding 1. PReLU activation and batch normalization follow the convolutional and deconvolutional layers. The coarse cartilage segmentor is trained based on multi-class cross entropy loss $$l_{mce}$$ to obtain cartilage masks from the down-sampled MR data (e.g., 192×192×160192×192×160)

### 2.2 Collaborative multi-agent learning

![Fib3]({{site.baseurl}}/assets/210129_multiagent_learning/img/fig3.png)
**Figure 3:** Demonstration of the collaborative multi-agent learning framework for fine- grained cartilage segmentation. The agents yield binary labels and the spatial fusion operation outputs a 4-channel result (FC, TC, PC and background). (Color figure online)
{: style="text-align: center; color: gray"}

#### 2.2.1 Segmentation Agents

> The segmentation agent $$A_{c=\{f, t, p\}}$$ ($$f$$, $$t$$ and $$p$$ stand for FC, TC and PC, respectively) aims to generate fine cartilage binary mask $$A_{c}(x_{i, c})$$  in the respective ROI $$x_{i, c}$$ (its ground truth (GT) ROI is $$y_{i,c}$$ and $$i$$ is the data index).

* adapt VNet-like structure
* add **attention mechanism** to skip connections
  > the connecting operation becomes $$o\left(\alpha \odot I_{l}, I_{h}^{up}\right)$$, where $$I_l$$ denotes low level features, $$I_{h}^{up}$$ denotes high level features, $$o$$ denotes concatenation along the channel dimension, and $$\odot$$ is element-wise multiplication. The attention mask $$\alpha=m\left(\sigma_{r}\left(c_{l}\left(I_{l}\right)+c_{h}\left(I_{h}^{u p}\right)\right)\right)$$ serves as a weight map that guides the learning to focus on desired region. Here, $$c_{h}$$ and $$c_{l}$$ are two convolutions of filter size 1 and stride 1; $$\sigma_{r}$$ is an activation function (e.g., ReLU); $$m$$ is another convolution of filter size 1 and stride 1 with sigmoid to contract the features to a single-channel mask.

#### 2.2.2 ROI-fusion Layer

> ROI-fusion layer $$\mathcal{F}$$ restores the single-cartilage output from each agent back to the original knee joint space where the mutual constraints and priors can be encoded. $$\mathcal{F}\left(A_{f}, A_{t}, A_{p}\right)$$ is implemented by using the location information of the three input ROIs to fuse the fine cartilage masks back to the original space. 

#### 2.2.3 Joint-label Discriminator (D)

> We utilize a discriminator sub-network *D* to classify the fused multi-cartilage mask as “fake” and the whole GT label $$y_{i}$$ as “real”. 
>
> In adversarial learning, the agents and the discriminator are trained alternatively. The parameters of agents are fixed when training the discriminator, and vice verse. In this way, discriminator sub-network can learn joint priors of multiple cartilages and guide the agents to produce better segmentation.
>
> The input to the discriminator is a pair of MR knee image $$x_{i}$$ and multi-label cartilage mask (either the GT label $$y_i$$ or $$\mathcal{F}\left(A_{f}, A_{t}, A_{p}\right)$$). A global average layer is utilized at the end to generate a probability value for fake/real mask discrimination.

#### 2.2.4 Loss Functions

**Loss function for discriminator:**

$$\sum_{i}\left\{\ell_{b}\left[D\left(\mathbf{x}_{i}, \mathbf{y}_{i}\right), 1\right]+\ell_{b}\left[D\left(\mathbf{x}_{i}, \mathcal{F}\left(A_{f}, A_{t}, A_{p}\right)\right), 0\right]\right\}\\$$

**Loss function for agents:**

$$\sum_{i}\left\{\sum_{c=\{f, t, p\}} L_{s}\left(\mathbf{x}_{i, c}, \mathbf{y}_{i, c}\right)+L_{m}+\ell_{b}\left[D\left(\mathbf{x}_{i}, \mathcal{F}\left(A_{f}, A_{t}, A_{p}\right)\right), 1\right]\right\}$$

$$L_{s}=\ell_{b}\left[A_{c}\left(\mathbf{x}_{i, c}\right), \mathbf{y}_{i, c}\right]$$

$$L_{m}=\ell_{\text {mce }}\left[\mathcal{F}\left(A_{f}, A_{t}, A_{p}\right), \mathbf{y}_{i}\right]$$

## 3. Dataset & Training Details

* OAI iMorphics dataset
  * 176 3D MR (sagittal DESS sequences) knee images
  * Training/validation/testing: 120/26/30
* Training
  * Batch size = 1 
  * LR * 0.95 every 10 epochs
  * Adam (with initial LR 0.001) for agent 
  * SGD (with initial LR 0.0002) for discriminator
  * The agents and the discriminator are trained alternatively (The parameters of agents are fixed when training the discriminator, and vice verse)

## 4. Results

|        | FC        |           |           | TC        |           |           | PC        |           |           | All       |           |           |
| :----- | :-------- | :-------- | :-------- | :-------- | :-------- | :-------- | :-------- | :-------- | :-------- | :-------- | :-------- | --------- |
|        | *DSC*     | *VOE*     | *ASD*     | *DSC*     | *VOE*     | *ASD*     | *DSC*     | *VOE*     | *ASD*     | *DSC*     | *VOE*     | *ASD*     |
| *D*1   | 0.862     | 24.15     | 0.103     | 0.869     | 22.93     | 0.104     | 0.844     | 26.65     | 0.107     | 0.866     | 23.59     | 0.095     |
| *D*2   | 0.832     | 28.64     | 0.131     | 0.879     | 21.38     | 0.088     | 0.861     | 23.69     | 0.091     | 0.851     | 25.94     | 0.111     |
| *C*0   | 0.814     | 31.30     | 0.205     | 0.806     | 32.42     | 0.199     | 0.771     | 35.74     | 0.350     | 0.809     | 31.99     | 0.213     |
| **P1** | 0.868     | 23.19     | 0.108     | 0.854     | 25.17     | 0.126     | 0.824     | 28.78     | 0.201     | 0.862     | 24.24     | 0.110     |
| **P2** | **0.900** | **18.82** | **0.074** | **0.889** | **19.81** | **0.082** | **0.880** | **21.19** | **0.075** | **0.893** | **19.19** | **0.073** |


* C0: the coarse cartilage extraction by the segmentor 
* P1: the fused results generated by the proposed segmentation agents, without the joint learning by the adversarial sub-network
* P2: results from the proposed method by employing the collaborative multi-agent learning framework
* D1: the residual blocks and skip connections are replaced by **DenseASPP** blocks in the two down-sampled levels of the agent network
* D2: only the deepest level is replaced with **DenseASPP** block

