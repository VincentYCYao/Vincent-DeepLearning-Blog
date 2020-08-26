---
layout: post
title:  "Learn PyTorch with me: a beginner guidance"
date:   2020-08-27 05:00:00 +0800
categories: python
---


# Learn PyTorch with me: a beginner guidance

**Background Knowledge on deep learning:**

I have taken the famous deep learning course [cs231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu) provided by [Feifei Li](https://profiles.stanford.edu/fei-fei-li) from Stanford University. It’s a good starting point to learn the covolutional neural network and gain your first concept of deep learning. In the latter part of this course, other neural network like GAN and RNN are introduced. More importantly, from the assignment, you can learn **TensorFlow** and **PyTorch**!

CS231n is highly recommended, and you must finish the assignments to consolidate what you have learned. 

**PyTorch Version: 1.6.0**





## My steps into the PyTorch world

### 1. Setup Virtual environment

Reference: [Get start with Anaconda, PyTorch & CUDA](https://pytorch.org/get-started/locally/)

* install anaconda and create a virtual environment (env)
* (within env) install pytorch, CUDA, and dependencies



### 2. Have a glance at PyTorch

Reference: [DEEP LEARNING WITH PYTORCH: A 60 MINUTE BLITZ](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) which includes 4 parts:

* [WHAT IS PYTORCH?](https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html#sphx-glr-beginner-blitz-tensor-tutorial-py) (view on GitHub: [**tensor_tutorial.py**](https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/tensor_tutorial.py))
  * [pytorch tensor operation](https://pytorch.org/docs/torch)
* [AUTOGRAD: AUTOMATIC DIFFERENTIATION](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#sphx-glr-beginner-blitz-autograd-tutorial-py) (view on GitHub: [**autograd_tutorial.py**](https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/autograd_tutorial.py))
  * [autograd.Function](https://pytorch.org/docs/stable/autograd.html#function)
* [NEURAL NETWORKS](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#sphx-glr-beginner-blitz-neural-networks-tutorial-py) (view on GitHub: [**neural_networks_tutorial.py**](https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/neural_networks_tutorial.py))
  * [loss function](https://pytorch.org/docs/nn.html#loss-functions)
  * [torch.nn](https://pytorch.org/docs/nn)
  * [Backprop](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#backprop)
* [TRAINING A CLASSIFIER](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py) (view on GitHub: [**cifar10_tutorial.py**](https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/cifar10_tutorial.py) )
  *  [Optional: Data Parallelism](https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html)

**Note:** I first go through all the basic introductions ignoring the reference reading. Some simple concepts are introduced. Let’s do it within 60 min, yet try your best to understand. **Don't** read the [doc](https://pytorch.org/docs/stable/index.html) at this stage since it contains too much details which is not friendly to beginner.



### 3. Learn simple PyTorch concepts

Reference: [LEARNING PYTORCH WITH EXAMPLES](https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#examples-download)

* The **NumPy** can be used to implement simple forward and backward passes in neural network training (by manual differentiation). 
* **PyTorch Tensors** can utilize GPUs to accelerate their numeric computations
* The `autograd` package in PyTorch can carry out automatic differentiation to automate the computation of backward passes in neural networks
* Each primitive **autograd operator** is really two functions that operate on Tensors:
  *  the **forward** function that computes output Tensors from input Tensors
  *  the **backward** function that computes the gradient of the input Tensors from the gradient of the output Tensors
* In PyTorch we can easily **define new autograd operator** by defining a subclass of `torch.autograd.Function` and implementing the `forward` and `backward` functions
* In PyTorch, the `nn` package serves this same purpose. The `nn` package defines a set of **Modules**, which are roughly equivalent to **neural network layers**. A Module receives input Tensors and computes output Tensors, but may also hold internal state such as Tensors containing learnable parameters. The `nn` package also defines a set of useful **loss functions** that are commonly used when training neural networks
* The `optim` package in PyTorch abstracts the idea of an **optimization algorithm** and provides implementations of commonly used optimization algorithms
* You can **define your own Modules** by subclassing `nn.Module` and defining a `forward` which receives input Tensors and produces output Tensors using other modules or other autograd operations on Tensors



### 4. *General data pipeline and training loop

Reference : [WHAT IS TORCH.NN *REALLY*?](https://pytorch.org/tutorials/beginner/nn_tutorial.html)                 (View on GitHub: [nn_tutorial.py](https://github.com/pytorch/tutorials/blob/master/beginner_source/nn_tutorial.py))

**Acknowledgement:** This tutorial is provided by Jeremy Howard, [fast.ai](https://www.fast.ai/). Thanks to Rachel Thomas and Francisco Ingham.

From this tutorial, you will learn how to write a **general data pipeline** and **training loop** which you can use for training many types of models using PyTorch. 

PyTorch provides the elegantly designed modules and classes [torch.nn](https://pytorch.org/docs/stable/nn.html) , [torch.optim](https://pytorch.org/docs/stable/optim.html) , [Dataset](https://pytorch.org/docs/stable/data.html?highlight=dataset#torch.utils.data.Dataset) , and [DataLoader](https://pytorch.org/docs/stable/data.html?highlight=dataloader#torch.utils.data.DataLoader) to help you create and train neural networks.

#### 4.1 Define network structure: torch.nn.Module

Reference: [nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)

`nn.Module` (uppercase M) is a PyTorch specific concept, and is a **class** we’ll be using a lot.

* **Base class for all neural network modules.**

* Your models should also subclass this class.

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

#### 4.2 Define network optimizer: torch.optim

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

> **Caveate:** **construct an optimizer before moving to GPU**
>
> If you need to move a model to GPU via `.cuda()`, please do so **before** constructing optimizers for it. Parameters of a model after `.cuda()` will be different objects with those before the call.
>
> In general, you should make sure that **optimized parameters live in consistent locations when optimizers are constructed and used.**

####  4.3 Import Data: TensorDataset, DataLoader & Wrapping DataLoader

Reference: [TORCH.UTILS.DATA](https://pytorch.org/docs/stable/data.html#map-style-datasets)

PyTorch has an abstract Dataset class. A Dataset can be anything that has a `__len__` function (called by Python’s standard `len` function) and a `__getitem__` function as a way of indexing into it.

##### 4.3.1 torch.utils.data.TensorDataset

* Dataset wrapping tensors

* Each sample will be retrieved by indexing tensors along the first dimension

```python
  from torch.utils.data import TensorDataset
  # combine dependent variables, for example images and its labels
  train_ds = TensorDataset(x_train, y_train)
  # slice dataset
  xb,yb = train_ds[i*bs : i*bs+bs]
```

##### 4.3.2 torch.utils.data.DataLoader

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
*  an instance of a subclass of `IterableDataset` that implements the `__iter__()`protocol, and represents an iterable over data samples
  * suitable for cases where random reads are expensive or even improbable, and where the batch size depends on the fetched data
  * e.g.: such a dataset, when called `iter(dataset)`, could return a stream of data reading from a database, a remote server, or even logs generated in real time

Pytorch’s `DataLoader` is responsible for managing batches. You can create a `DataLoader` from any `Dataset`. `DataLoader` makes it easier to iterate over batches. Rather than having to use `train_ds[i*bs :i*bs+bs]`, the DataLoader gives us each minibatch automatically: 

```python
from torch.utils.data import DataLoader

train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs)
```

##### 4.3.3 Wrapping DataLoader

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

#### 4.4 Using GPU

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

#### 4.5 Closing thoughts

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



### 5. Visualization in PyTorch

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



### 6. Explore the [pytorch.org](https://pytorch.org)

The [official site of PyTorch](https://pytorch.org) is elegantly designed to provide useful information categorized into the following groups: get started, blog, tutorials, and Docs. Below are the resources of my interest.

[**Official Blogs:**](https://pytorch.org/blog/)

* [ ] [PyTorch 1.6 now includes Stochastic Weight Averaging](https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/)
* [ ] [Introducing native PyTorch automatic mixed precision for faster training on NVIDIA GPUs](https://pytorch.org/blog/accelerating-training-on-nvidia-gpus-with-pytorch-automatic-mixed-precision/)
* [ ] [Introduction to Quantization on PyTorch](https://pytorch.org/blog/introduction-to-quantization-on-pytorch/)

[**PyTorch Recipes:**](https://pytorch.org/tutorials/recipes/recipes_index.html) recipes with tag `Basic`would be a good start. Below are recommended recipes:

* (Recommended) **Saving and loading**
  * [x] [what is a state_dic in pytorch](https://pytorch.org/tutorials/recipes/recipes/what_is_state_dict.html)
  * [x] [saving and loading models for inference in pytorch](https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_models_for_inference.html)
  * [x] [saving and loading a general checkpoint in pytorch](https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html)
  * [x] [saving and loading multiple models in one file using pytorch](https://pytorch.org/tutorials/recipes/recipes/saving_multiple_models_in_one_file.html)
  * [ ] [saving and loading models across devices in pytorch](https://pytorch.org/tutorials/recipes/recipes/save_load_across_devices.html)
* (Recommended)  **Data customization**
  
  * [ ] [developing custom pytorch dataloader](https://pytorch.org/tutorials/recipes/recipes/custom_dataset_transforms_loader.html)
* **Monitoring the time and memory consumption** 

  * [ ] [pytorch profiler](https://pytorch.org/tutorials/recipes/recipes/profiler.html#pytorch-profiler) 
* **Tranfer learning: warmstart a model**

  * [ ] [warmstarting model using parameters from a different model in pytorch](https://pytorch.org/tutorials/recipes/recipes/warmstarting_model_using_parameters_from_a_different_model.html) 
* **Model optimization & quantization**
  * [ ] [dynamic quantization](https://pytorch.org/tutorials/recipes/recipes/dynamic_quantization.html#dynamic-quantization)

* **Model deployment**

  * [ ] [deploying with Flask](https://pytorch.org/tutorials/recipes/deployment_with_flask.html)

    * [ ] related tutorial: [Deploying pytorch in python via a rest API with Flask](https://pytorch.org/tutorials/intermediate/flask_rest_api_tutorial.html)

  * [ ] [TorchScript for deployment](https://pytorch.org/tutorials/recipes/torchscript_inference.html)

    * [ ] related tutorial: [Introduction to TorchScript](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html#introduction-to-torchscript)

* **Interpretability**
  * [ ]  [model interpretability using Captum](https://pytorch.org/tutorials/recipes/recipes/Captum_Recipe.html)

**Advance-level Tutorials:**

* **Distributed training**
  * [ ] [PyTorch distributed overview](https://pytorch.org/tutorials/beginner/dist_overview.html)
  * [ ] [Single-machine model parallel best practices](https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html)
  * [ ] [Getting started with distributed data parallel](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
  * [ ] [Writing distributed applications with pytorch](https://pytorch.org/tutorials/intermediate/dist_tuto.html)
  * [ ] [Getting started with distributed RPC framework](https://pytorch.org/tutorials/intermediate/rpc_tutorial.html)
  * [ ] [Implementing a parameter server using distributed RPC framework](https://pytorch.org/tutorials/intermediate/rpc_param_server_tutorial.html)
  * [ ] [Distributed pipeline parallelism using RPC](https://pytorch.org/tutorials/intermediate/dist_pipeline_parallel_tutorial.html)
  * [ ] [Implementing batch RPC processing using asynchronous executions](https://pytorch.org/tutorials/intermediate/rpc_async_execution.html)
  * [ ] [Combining distributed dataparallel with distributed RPC framework](https://pytorch.org/tutorials/advanced/rpc_ddp_tutorial.html)



#### 6.1 State dictionary  in PyTorch

Reference: [WHAT IS A STATE_DICT IN PYTORCH](https://pytorch.org/tutorials/recipes/recipes/what_is_state_dict.html)

A `state_dict` is an integral entity if you are interested in saving or loading models from PyTorch. Because `state_dict` objects are Python dictionaries, they can be easily saved, updated, altered, and restored, adding a great deal of modularity to PyTorch models and optimizers. 

Who has `state_dict` in PyTorch:

* **model**: only layers with learnable parameters (convolutional layers, linear layers, etc.) and registered buffers (batchnorm’s running_mean) have entries in the model’s `state_dict`
* **optimizer object**:  contains information about the optimizer’s state, as well as the hyperparameters used

```python
# Print model's state_dict
print("Model's state_dict:")
for param_tensor in net.state_dict():
    print(param_tensor, "\t", net.state_dict()[param_tensor].size())

print()

# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])
```

#### 6.2 Save and load a model

Reference: [SAVING AND LOADING MODELS FOR INFERENCE IN PYTORCH](https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_models_for_inference.html)

Two ways of saving models in Pytorch:

* (**Recommended**) Save the `state_dict`with `torch.save()`
  * give you the most flexibility for restoring the model later.
* Save the entire model with Python’s [pickle](https://docs.python.org/3/library/pickle.html):
  * Advantage: yields the most intuitive syntax and involves the least amount of code
  * Disadvantage: the serialized data is bound to the specific classes and the exact directory structure used when the model is saved —  **your code can break in various ways when used in other projects or after refactors**

#####  Save and load a model with via `state_dict`

* A common PyTorch convention is to save models using either a `.pt` or `.pth` file extension
* Remember too, that you must call `model.eval()` to set **dropout** and **batch normalization layers** to evaluation mode before running inference. Failing to do this will yield inconsistent inference results.

```python
# Specify a path
PATH = "state_dict_model.pt"

# Save
torch.save(net.state_dict(), PATH)

# Load
model = Net()
model.load_state_dict(torch.load(PATH))
model.eval()
```

#### 6.3 Save and load a general checkpoint

Saving and loading a general checkpoint model for **inference or resuming training** can be helpful for picking up where you last left off. When saving a general checkpoint, you must save the followings:

* **model’s state_dict**
* **optimizer’s state_dict**: contains buffers and parameters that are updated as the model trains
* other items: 
  * the **epoch** you left off on
  * the latest recorded **training loss**
  * **external `torch.nn.Embedding` layers**
  * ...

##### General introduction

To save multiple checkpoints, you must organize them in a dictionary and use `torch.save()` to serialize the dictionary. A common PyTorch convention is to save these checkpoints using the `.tar` file extension. 

To load the items: 

1. first initialize the model and optimizer, 
2. then load the dictionary locally using `torch.load()`

#####  Save general checkpoint

```python
# Additional information
EPOCH = 5
PATH = "model.pt"
LOSS = 0.4
	
torch.save({
            'epoch': EPOCH,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': LOSS,
            }, PATH)
```

#####  Load the general checkpoint

**Reminder:** Remember to first initialize the model and optimizer, then load the dictionary locally.

```python
model = Net()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

# set model to evaluate mode
model.eval()
# - or -
# set model to training mode
model.train()
```

#### 6.3 Save and load multiple models in one file

When saving a model comprised of multiple `torch.nn.Modules`, such as a GAN, a sequence-to-sequence model, or an ensemble of models, you must save a dictionary of each model’s state_dict and corresponding optimizer. You can also save any other items that may aid you in resuming training by simply appending them to the dictionary. 

To load the models, first initialize the models and optimizers, then load the dictionary locally using `torch.load()`. From here, you can easily access the saved items by simply querying the dictionary as you would expect. In this recipe, we will demonstrate how to save multiple models to one file using PyTorch.

##### Save multiple models

```python
# assuming the class Net() has been defined
netA = Net()
netB = Net()

optimizerA = optim.SGD(netA.parameters(), lr=0.001, momentum=0.9)
optimizerB = optim.SGD(netB.parameters(), lr=0.001, momentum=0.9)

# Specify a path to save to
PATH = "model.pt"

torch.save({
            'modelA_state_dict': netA.state_dict(),
            'modelB_state_dict': netB.state_dict(),
            'optimizerA_state_dict': optimizerA.state_dict(),
            'optimizerB_state_dict': optimizerB.state_dict(),
            }, PATH)
```

##### Load multiple models

```python
# initialze model and optimizer
modelA = Net()
modelB = Net()
optimModelA = optim.SGD(modelA.parameters(), lr=0.001, momentum=0.9)
optimModelB = optim.SGD(modelB.parameters(), lr=0.001, momentum=0.9)

# loading
checkpoint = torch.load(PATH)
modelA.load_state_dict(checkpoint['modelA_state_dict'])
modelB.load_state_dict(checkpoint['modelB_state_dict'])
optimizerA.load_state_dict(checkpoint['optimizerA_state_dict'])
optimizerB.load_state_dict(checkpoint['optimizerB_state_dict'])

# set model status: evaluation or training
modelA.eval()
modelB.eval()
# - or -
modelA.train()
modelB.train()
```

