---
layout: post
title:  "PyTorch: save and load models"
date:   2020-08-27 05:35:00 +0800
categories: python
---



# 1. Introduction: state-dict  in PyTorch

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

# 2. Save and load a model

Reference: [SAVING AND LOADING MODELS FOR INFERENCE IN PYTORCH](https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_models_for_inference.html)

Two ways of saving models in Pytorch:

* (**Recommended**) Save the `state_dict`with `torch.save()`
  * give you the most flexibility for restoring the model later.
* Save the entire model with Python’s [pickle](https://docs.python.org/3/library/pickle.html):
  * Advantage: yields the most intuitive syntax and involves the least amount of code
  * Disadvantage: the serialized data is bound to the specific classes and the exact directory structure used when the model is saved —  **your code can break in various ways when used in other projects or after refactors**

##  2.1 Save and load a model via `state_dict`

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

# 3. Save and load a general checkpoint

Saving and loading a general checkpoint model for **inference or resuming training** can be helpful for picking up where you last left off. When saving a general checkpoint, you must save the followings:

* **model’s state_dict**
* **optimizer’s state_dict**: contains buffers and parameters that are updated as the model trains
* other items: 
  * the **epoch** you left off on
  * the latest recorded **training loss**
  * **external `torch.nn.Embedding` layers**
  * ...

## 3.1 General introduction

To save multiple checkpoints, you must organize them in a dictionary and use `torch.save()` to serialize the dictionary. A common PyTorch convention is to save these checkpoints using the `.tar` file extension. 

To load the items: 

1. first initialize the model and optimizer, 
2. then load the dictionary locally using `torch.load()`

##  3.2 Save the general checkpoint

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

##  3.3 Load the general checkpoint

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

# 4. Save and load multiple models in one file

When saving a model comprised of multiple `torch.nn.Modules`, such as a GAN, a sequence-to-sequence model, or an ensemble of models, you must save a dictionary of each model’s state_dict and corresponding optimizer. You can also save any other items that may aid you in resuming training by simply appending them to the dictionary. 

To load the models, first initialize the models and optimizers, then load the dictionary locally using `torch.load()`. From here, you can easily access the saved items by simply querying the dictionary as you would expect. In this recipe, we will demonstrate how to save multiple models to one file using PyTorch.

## 4.1 Save multiple models

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

## 4.2 Load multiple models

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

