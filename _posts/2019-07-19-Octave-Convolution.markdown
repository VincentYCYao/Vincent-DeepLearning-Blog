---
layout: post
title:  "Review: Reducing Spatial Redundancy with Octave Convolution"
date:   2019-07-19 06:00:00 +0800
categories: CNN
---


In this blog, the paper [**Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave Convolution**](http://arxiv.org/abs/1904.05049) is reviewed. This paper was first published on arXiv in 2019.

The authors proposed **Octave Convolution (OctConv)**, which can replace vanilla convolution without changing the network structure. 

The implementation of OctConv can be found [**here**](https://github.com/facebookresearch/OctConv).

<br/>
<br/>




## 1. Background and Inspiration
# 1.1. Background

> Convolutional Neural Networks (CNNs) have achieved remarkable success in many computer vision tasks and their efficiency keeps increasing with recent efforts to reduce the inherent redundancy in dense model parameters and in the channel dimension of feature maps.

> Substantial redundancy also exists in the spatial dimension of the feature maps produced by CNNs

<br/>


# 1.2. Inspiration

> A natural image can be decomposed into a low spatial frequency component that describes the smoothly changing structure and a high spatial frequency component that describes the rapidly changing fine details 

> The output feature maps of a convolution layer can also be seen as a mixture of information at different frequencies.

**Octave Convolution** is proposed to factorize the mixed feature maps into low- and high- frequency features maps. The ratio **low-frequency features to high-frequency features** is defined by a hyper-parameter **`alpha`**. The proposed Octave Convolution can process and store the low- and high- feature maps in such a way that it can replace the conventional convolution operation without network structure adjustment.

![fig1]({{site.baseurl}}/assets/190719_OctConv/img/fig1.png)

Figure 1. (a) Motivation. The spatial frequency model for vision [1, 10] shows that natural image can be decomposed into a low and a high spatial frequency part. (b) The output maps of a convolution layer can also be factorized and grouped by their spatial frequency. (c) The proposed multifrequency feature representation stores the smoothly changing, low-frequency maps in a low-resolution tensor to reduce spatial redundancy. (d) The proposed Octave Convolution operates directly on this representation. It updates the information for each group and further enables information exchange between groups.
{: style="text-align: center; color: gray"}

<br/>
<br/>




## 2. Methods

# 2.1. Octave

> defines an octave as a division of the spatial dimensions by a power of 2 (we only explore 21 in this work) 

<br/>


# 2.2. Octave Convolution

> The goal of our design is to effectively process the low and high frequency in their corresponding frequency tensor but also enable efficient communication between the high and low frequency component of our Octave feature representation

![fig2]({{site.baseurl}}/assets/190719_OctConv/img/fig2.png)

Figure 2. Detailed design of the Octave Convolution. Green arrows correspond to information updates while red arrows facilitate information exchange between the two frequencies. 
{: style="text-align: center; color: gray"}

* Here α ∈ [0, 1] denotes the ratio of channels allocated to the low-frequency part and the low-frequency feature maps are defined an octave lower than the high frequency ones, i.e. at half of the spatial resolution as shown in Figure 2

**Octave Convolution can be rewritten as**:

$$ 
Y_H = f(X^H; W^{H \rightarrow H}) + upsample(f(X^L; W^{L \rightarrow H}), 2)
$$

$$
Y_L = f(X^L; W^{L \rightarrow L}) + f(pool(X^H, 2); W^{L \rightarrow H})
$$

* `f(X ; W )` denotes a convolution with parameters W
* `pool(X, k)` is an **average pooling** operation with kernel size k × k and stride k
* `upsample(X, k)` is an up-sampling operation by a factor of k via **nearest interpolation**

<br/>


# 2.3. Integrating OctConv into Backbone Networks

 * To convert a vanilla feature representation to a multi-frequency feature representation, i.e. **at the first OctConv layer**, we set $$ α_{in} = 0 $$ and $$ α_{out} = α $$. In this case, OctConv paths related to the low-frequency input is disabled, resulting in a simplified version which only has two paths. 

 * To convert the multi-frequency feature representation back to vanilla feature representation, i.e. **at the last OctConv layer**, we set $$ α_{out} = 0 $$. In this case, OctConv paths related to the low-frequency output is disabled, resulting in a single full resolution output.

<br/>


# 2.4. Comparison to Multi-grid Convolution (MG-Conv)
> The multi-grid convolution (MG-Conv) is a bi-directional and cross-scale convolution operator that can be integrated throughout a CNN, conceptually similar to our OctConv. 

> The core difference is the design of the operator, stemming from different motivations for each design, which leads to significant performance difference. MG-Conv aims to exploit multi-scale information, while OctConv is optimized for reducing spatial redundancy.

[**MG-Conv**](http://openaccess.thecvf.com/content_cvpr_2017/html/Ke_Multigrid_Neural_Architectures_CVPR_2017_paper.html) and **OctConv** rely on different down- and up- sampling strategies for the information exchange between features at different scales

* **Downsampling**
	* **MG-Conv** relies on **max-pooling** to extract low-frequency features from the high-frequency ones, which requires extra memory to store the index of the maximum value during training
	* **OctConv** adopts **average-pooling** for distilling low-frequency features from the high-frequency ones which might be better to downsample the feature maps and does not require extra memory

* **Upsampling**
	*  **MG-Conv** first upsamples and then convolves with the feature map
	*  **OctConv** performs upsampling after convolution, which is more efficient than MG-Conv


<br/>
<br/>





## 3. Experiments and Results
# 3.1. Theoretical Gains On Computational Cost and Memory Consumption

![fig6]({{site.baseurl}}/assets/190719_OctConv/img/fig6.png)

**Table 1**: Relative theoretical gains for the proposed multi- frequency feature representation over vanilla feature maps for varying choices of the ratio α of channels used by the low-frequency feature. When `α = 0`, no low-frequency feature is used which is the case of vanilla convolution. Note the number of parameters in OctConv operator is constant regardless of the choice of ratio. 
{: style="color: gray"}

<br/>



# 3.2. Ablation Study on ImageNet

> We conduct a series of ablation studies aiming to answer the following questions: 
1) Does OctConv have better FLOPs-Accuracy trade-off than vanilla convolution? 
2) In which situation does the OctConv work the best?


**Resuls:**
* On **ResNet-50**
	* The flops-accuracy trade-off curve is a concave curve
	* At `α = 0.5`, where the network gets similar or better results even when the FLOPs are reduced by about half
	* At `α = 0.125`, where the network reaches its best accuracy, 1.2% higher than baseline (black circle)
	* 75% of the feature maps can be compressed to half the resolution with only 0,4% accuracy drop

* On [**ResNet-(26;50;101;200)**](https://arxiv.org/abs/1603.05027), [**ResNeXt-(50,32×4d;101,32×4d)**](https://arxiv.org/abs/1611.05431), [**DenseNet-121**](https://arxiv.org/abs/1608.06993) and [**SE-ResNet-50**](https://arxiv.org/abs/1709.01507)
	* OctConv equipped networks for different architecture behave similarly to the Oct-ResNet-50

![fig3]({{site.baseurl}}/assets/190719_OctConv/img/fig3.png)

Figure 4: Ablation study results on ImageNet. OctConv- equipped models are more efficient and accurate than base- line models. Markers in black in each line denote the corresponding baseline models without OctConv. The colored numbers are the ratio α. Numbers in X axis denote FLOPs in logarithmic scale.
{: style="text-align: center; color: gray"}



**Other Findings**
* Both the information exchanging paths are important, since removing any of them can lead to accuracy drop as shown in Table 3.

![fig5]({{site.baseurl}}/assets/190719_OctConv/img/fig5.png)

**Table 3**: Ablation on down-sampling and inter-octave connectivity on ImageNet.
{: style="color: gray"}


* At test time, the gain of OctConv over baseline models increases as the test image resolution grows because OctConv can detect large objects better due to its larger receptive field, as shown in Table 4. 

![fig4]({{site.baseurl}}/assets/190719_OctConv/img/fig4.png)

**Table 4**: ImageNet classification accuracy. The short length of input images are resized to the target crop size while keeping the aspect ratio unchanged. A center crop is adopted if the input image size is not square. ResNet-50 backbone trained with crops size of 256 × 256 pixels.
{: style="color: gray"}

<br/>



# 3.3. Comparing with SOTAs on ImageNet

**Small Models**
* [**“0.75 MobileNet (v1)”**](https://arxiv.org/abs/1704.04861)
	* OctConv can reduce the FLOPs of **MobileNetV1** by 34%, and provide better accuracy and faster speed in practice; 

* [**“1.0 MobileNet (v2)”**](https://arxiv.org/abs/1801.04381) as baseline and replace the regular convolution with our proposed OctConv.
	* OctConv is able to reduce the FLOPs of **MobileNetV2** by 15%, achieving the same accuracy with faster speed. 


![fig7]({{site.baseurl}}/assets/190719_OctConv/img/fig7.png)

**Table 5**: ImageNet classification results for Small models. ∗ indicates it is better than original reproduced by MXNet GluonCV v0.4. The inference speed is tested using TVM on Intel Skylake processor (2.0GHz, single thread).
{: style="color: gray"}




**Medium Models**
>  we compare OctConv with MG-Conv, Elastic and bL-Net which share a similar idea as our method.

![fig8]({{site.baseurl}}/assets/190719_OctConv/img/fig8.png)

Table 6: ImageNet Classification results for Middle sized models. ‡ refers to method that replaces “Max Pooling” by extra convolution layer(s) [[4]](https://arxiv.org/abs/1807.03848). § refers to method that uses balanced residual block distribution [[4]](https://arxiv.org/abs/1807.03848).
{: style="color: gray"}

* [**MG-Conv**](http://openaccess.thecvf.com/content_cvpr_2017/html/Ke_Multigrid_Neural_Architectures_CVPR_2017_paper.html)
	* **Oct-ResNet-26** shows 0.4% better accuracy than **R-MG-34** while costing only one third of FLOPs and half of Params. 
	* **Oct-ResNet-50**, which costs less than half of FLOPS, achieves 1.8% higher accuracy than **R-MG-34**

* [**Elastic**](https://arxiv.org/abs/1812.05262)
	* **Oct-ResNeXt-50** achieves better accuracy than the **Elastic based method** (78.7% v.s. 78.4%) while reducing the computational cost by 31%
	* **Oct-ResNeXt-101** also achieves higher accuracy than the **Elastic based method** (79.5% v.s. 79.2%) while costing 38% less computation

* [**bL-Net**](https://arxiv.org/abs/1807.03848)
	* OctConv equipped methods achieve better FLOPs-Accuracy trade-off without bells and tricks
	* **Oct-ResNet-50** achieves 0.8% higher accuracy than **bL-ResNet-50** under the same computational budget (group 4)
	* **Oct-ResNeXt-50** (group 5) and **Oct-ResNeXt-101** (group 6) get better accuracy under comparable or even lower computational budget



**Large Models**
> Table 7 shows the results of OctConv in large models. 

> Here, we choose the ResNet-152 as the back- bone CNN, replacing the first 7 × 7 convolution by three 3 × 3 convolution layers and removing the max pooling by a lightweight residual block [[4]](https://arxiv.org/abs/1807.03848). We report results for Oct- ResNet-152 with and without the SE-block [[19]](https://arxiv.org/abs/1709.01507).


![fig9]({{site.baseurl}}/assets/190719_OctConv/img/fig9.png)

**Table 7**: ImageNet Classification results for Large models. The names of OctConv-equipped models are in bold font and performance numbers for related works are copied from the corresponding papers. Networks are evaluated using CuDNN v10.03in flop16 on a single Nvidia Titan V100 (32GB) for their training memory cost and speed. Works that employ neural architecture search are denoted by ($$ \diamond $$). We set batch size to 128 in most cases, but had to adjust it to 64 (noted by †), 32 (noted by ‡) or 8 (noted by §) for networks that are too large to fit into GPU memory.
{: style="color: gray"}

<br/>
<br/>


## 4. Conclusions
* The authors propose a novel Octave Convolution operation to store and process low- and high-frequency features separately to improve the model efficiency
* Octave Convolution is sufficiently generic to replace the regular convolution operation in-place, and can be used in most 2D and 3D CNNs without model architecture adjustment
* Beyond saving a substantial amount of computation and memory, Octave Convolution can also improve the recognition performance 





















