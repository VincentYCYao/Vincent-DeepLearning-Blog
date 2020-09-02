---
layout: post
title:  "Learn PyTorch: my steps into the PyTorch world"
date:   2020-08-27 06:00:00 +0800
categories: python
---



**Background Knowledge on deep learning:**

I have taken the famous deep learning course [cs231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu) provided by [Feifei Li](https://profiles.stanford.edu/fei-fei-li) from Stanford University. It’s a good starting point to learn the covolutional neural network and gain your first concept of deep learning. In the latter part of this course, other neural network like GAN and RNN are introduced. More importantly, from the assignment, you can learn **TensorFlow** and **PyTorch**!

CS231n is highly recommended, and you must finish the assignments to consolidate what you have learned. 

**PyTorch Version: 1.6.0**

# 1. Setup Virtual environment

Reference: [Get start with Anaconda, PyTorch & CUDA](https://pytorch.org/get-started/locally/)

* install anaconda and create a virtual environment (env)
* (within env) install pytorch, CUDA, and dependencies

# 2. Have a glance at PyTorch

Reference: [DEEP LEARNING WITH PYTORCH: A 60 MINUTE BLITZ](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) which includes 4 parts:

* [WHAT IS PYTORCH?](https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html#sphx-glr-beginner-blitz-tensor-tutorial-py) (view on GitHub: [tensor_tutorial.py](https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/tensor_tutorial.py))
  * [pytorch tensor operation](https://pytorch.org/docs/torch)
* [AUTOGRAD: AUTOMATIC DIFFERENTIATION](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#sphx-glr-beginner-blitz-autograd-tutorial-py) (view on GitHub: [autograd_tutorial.py](https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/autograd_tutorial.py))
  * [autograd.Function](https://pytorch.org/docs/stable/autograd.html#function)
* [NEURAL NETWORKS](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#sphx-glr-beginner-blitz-neural-networks-tutorial-py) (view on GitHub: [neural_networks_tutorial.py](https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/neural_networks_tutorial.py))
  * [loss function](https://pytorch.org/docs/nn.html#loss-functions)
  * [torch.nn](https://pytorch.org/docs/nn)
  * [Backprop](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#backprop)
* [TRAINING A CLASSIFIER](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py) (view on GitHub: [cifar10_tutorial.py](https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/cifar10_tutorial.py) )
  *  [Optional: Data Parallelism](https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html)

**Note:** I first go through all the basic introductions ignoring the reference reading. Some simple concepts are introduced. Let’s do it within 60 min, yet try your best to understand. **Don't** read the [doc](https://pytorch.org/docs/stable/index.html) at this stage since it contains too much details which is not friendly to beginner.

# 3. Learn simple PyTorch concepts

Reference: [LEARNING PYTORCH WITH EXAMPLES](https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#examples-download)

* The **NumPy** can be used to implement simple forward and backward passes in neural network training (by manual differentiation). 
* **PyTorch Tensors** can utilize GPUs to accelerate their numeric computations
* The `autograd` package in PyTorch can carry out automatic differentiation to automate the computation of backward passes in neural networks
* Each primitive **autograd operator** is really two functions that operate on Tensors:
  *  the **forward** function that computes output Tensors from input Tensors
  *  the **backward** function that computes the gradient of the input Tensors from the gradient of the output Tensors
* In PyTorch we can easily define new **autograd operator** by defining a subclass of `torch.autograd.Function` and implementing the `forward` and `backward` functions
* In PyTorch, the `nn` package serves this same purpose. 
  * The `nn` package defines a set of **Modules**, which are roughly equivalent to neural network layers. A Module receives input Tensors and computes output Tensors, but may also hold internal state such as Tensors containing learnable parameters. 
  * The `nn` package also defines a set of useful **loss functions** that are commonly used when training neural networks
* The `optim` package in PyTorch abstracts the idea of an **optimization algorithm** and provides implementations of commonly used optimization algorithms
* You can **define your own Modules** by subclassing `nn.Module` and defining a `forward` which receives input Tensors and produces output Tensors using other modules or other autograd operations on Tensors

# 4. *General data pipeline and training loop

Reference : [WHAT IS TORCH.NN *REALLY*?](https://pytorch.org/tutorials/beginner/nn_tutorial.html)    (View on GitHub: [nn_tutorial.py](https://github.com/pytorch/tutorials/blob/master/beginner_source/nn_tutorial.py))

— **Acknowledgement**: Jeremy Howard ([fast.ai](https://www.fast.ai/)), Rachel Thomas, Francisco Ingham ---

From this tutorial, you will learn how to write a **general data pipeline** and **training loop** which you can use for training many types of models using PyTorch. 

PyTorch provides the elegantly designed modules and classes [torch.nn](https://pytorch.org/docs/stable/nn.html) , [torch.optim](https://pytorch.org/docs/stable/optim.html) , [Dataset](https://pytorch.org/docs/stable/data.html?highlight=dataset#torch.utils.data.Dataset) , and [DataLoader](https://pytorch.org/docs/stable/data.html?highlight=dataloader#torch.utils.data.DataLoader) to help you create and train neural networks.

## 4.1 Define network structure: torch.nn.Module

Reference: [nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)

`nn.Module` (uppercase M) is a PyTorch specific concept, and is a **class** we’ll be using a lot.

* **Base class for all neural network modules**

* Your models should also subclass this class

* Modules can also contain other Modules, allowing to nest them in a tree structure. You can assign the submodules as regular attributes

A example of subclassing the nn.Module:

```python
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))
```

## 4.2 Define network optimizer: torch.optim

Reference: [torch.optim](https://pytorch.org/docs/stable/optim.html#module-torch.optim)

`torch.optim` is a package implementing various optimization algorithms. Most commonly used methods are already supported, and the interface is general enough, so that more sophisticated ones can be also easily integrated in the future.

A manually coded optimization step would be like :

```python
with torch.no_grad():
    for p in model.parameters(): p -= p.grad * lr
    model.zero_grad()
```

With optimizer `opt`, it would be:

```python
# update
opt.step()
# clean grad
opt.zero_grad()
```

> **Note:** **construct an optimizer before moving to GPU**
>
> If you need to move a model to GPU via `.cuda()`, please do so **before** constructing optimizers for it. Parameters of a model after `.cuda()` will be different objects with those before the call.
>
> In general, you should make sure that **optimized parameters live in consistent locations when optimizers are constructed and used.**

##  4.3 Import Data: TensorDataset, DataLoader & Wrapping DataLoader

Reference: [TORCH.UTILS.DATA](https://pytorch.org/docs/stable/data.html#map-style-datasets)

PyTorch has an abstract Dataset class. A Dataset can be anything that has a `__len__` function (called by Python’s standard `len` function) and a `__getitem__` function as a way of indexing into it.

### 4.3.1 torch.utils.data.TensorDataset

* Dataset wrapping tensors

* Each sample will be retrieved by indexing tensors along the first dimension

```python
  from torch.utils.data import TensorDataset
  # combine dependent variables, for example images and its labels
  train_ds = TensorDataset(x_train, y_train)
  # slice dataset
  xb,yb = train_ds[i*bs : i*bs+bs]
```

### 4.3.2 torch.utils.data.DataLoader

It represents a Python iterable over a dataset, with support for

* map-style and iterable-style datasets
* customizing data loading order
* automatic batching
* single- and multi-process data loading
* automatic memory pinning

These options are configured by the constructor arguments of a `DataLoader`, which has signature:

```python
DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None)
```

The most important argument of `DataLoader`constructor is `dataset`, which indicates a dataset object to load data from. PyTorch supports two different types of datasets:

* **map-style datasets**
  * implements the `__getitem__()` and `__len__()` protocols, and represents a map from (possibly non-integral) indices/keys to data samples
  * e.g.: image and its corresponding label

* **iterable-style datasets**
  * an instance of a subclass of `IterableDataset` that implements the `__iter__()`protocol, and represents an iterable over data samples
  * suitable for cases where random reads are expensive or even improbable, and where the batch size depends on the fetched data
  * e.g.: such a dataset, when called `iter(dataset)`, could return a stream of data reading from a database, a remote server, or even logs generated in real time

Pytorch’s `DataLoader` is responsible for managing batches. You can create a `DataLoader` from any `Dataset`. `DataLoader` makes it easier to iterate over batches. Rather than having to use `train_ds[i*bs :i*bs+bs]`, the DataLoader gives us each minibatch automatically: 

```python
from torch.utils.data import DataLoader

train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs)
```

### 4.3.3 Wrapping DataLoader

**create a get_data() to load train_ds and val_ds**

```python
def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )
```

**wrap data preprocessing into dataloader**

```python
def preprocess(x, y):
    return x.view(-1, 1, 28, 28), y

class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))
# ds:dataset dl: dataloader
train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
train_dl = WrappedDataLoader(train_dl, preprocess)
valid_dl = WrappedDataLoader(valid_dl, preprocess)
```

## 4.4 Using GPU

**create a device object**

```python
dev = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
```

**data preprocessing: move batches to the GPU**

```python
def preprocess(x, y):
    return x.view(-1, 1, 28, 28).to(dev), y.to(dev)

train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
train_dl = WrappedDataLoader(train_dl, preprocess)
valid_dl = WrappedDataLoader(valid_dl, preprocess)
```

## 4.5 Closing thoughts

We now have a general data pipeline and training loop which you can use for training many types of models using Pytorch. To see how simple training a model can now be, take a look at the mnist_sample sample notebook.

Of course, there are many things you’ll want to add, such as data augmentation, hyperparameter tuning, monitoring training, transfer learning, and so forth. These features are available in the fastai library, which has been developed using the same design approach shown in this tutorial, providing a natural next step for practitioners looking to take their models further.

We promised at the start of this tutorial we’d explain through example each of `torch.nn`, `torch.optim`, `Dataset`, and `DataLoader`. So let’s summarize what we’ve seen:

* torch.nn
  * `Module`: creates a callable which behaves like a function, but can also contain state(such as neural net layer weights). It knows what `Parameter` (s) it contains and can zero all their gradients, loop through them for weight updates, etc.
  * `Parameter`: a wrapper for a tensor that tells a `Module` that it has weights that need updating during backprop. Only tensors with the requires_grad attribute set are updated
  * `functional`: a module(usually imported into the `F`namespace by convention) which contains activation functions, loss functions, etc, as well as non-stateful versions of layers such as convolutional and linear layers.
* `torch.optim`: Contains optimizers such as `SGD`, which update the weights of `Parameter` during the backward step
* `Dataset`: An abstract interface of objects with a `__len__` and a `__getitem__`, including classes provided with Pytorch such as `TensorDataset`
* `DataLoader`: Takes any `Dataset` and creates an iterator which returns batches of data.



# 5. Visualization PyTorch model with TensorBoard

Reference: 

* [VISUALIZING MODELS, DATA, AND TRAINING WITH TENSORBOARD](https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html)
* [TORCH.UTILS.TENSORBOARD](https://pytorch.org/docs/stable/tensorboard.html)

Tensroboard introduction: [TensorFlow's visualization toolkit](https://www.tensorflow.org/tensorboard/)

Once you’ve installed TensorBoard, these utilities let you log PyTorch models and metrics into a directory for visualization within the TensorBoard UI. Scalars, images, histograms, graphs, and embedding visualizations are all supported for PyTorch models and tensors as well as Caffe2 nets and blobs.

#### Entry to TensorBoard logging: SummaryWriter

The `SummaryWriter` class is your main entry to log data for consumption and visualization by TensorBoard. For example:

```python
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

# Writer will output to ./runs/ directory by default
writer = SummaryWriter()

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = datasets.MNIST('mnist_train', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
model = torchvision.models.resnet50(False)
# Have ResNet model take in grayscale rather than RGB
model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
images, labels = next(iter(trainloader))

grid = torchvision.utils.make_grid(images)
writer.add_image('images', grid, 0)
writer.add_graph(model, images)
writer.close()
```

**Tips:** Lots of information can be logged for one experiment. To avoid cluttering the UI and have better result clustering, we can group plots by naming them hierarchically. For example, “Loss/train” and “Loss/test” will be grouped together, while “Accuracy/train” and “Accuracy/test” will be grouped separately in the TensorBoard interface.