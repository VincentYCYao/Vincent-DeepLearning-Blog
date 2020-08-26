---
layout: post
title:  "Explore the pytorch.org"
date:   2020-08-27 05:20:00 +0800
categories: python
---



# Explore the pytorch.org

The [official site of PyTorch](https://pytorch.org) is elegantly designed to provide useful information categorized into the following groups: get started, blog, tutorials, and Docs. 

Below are the resources I am interested in.

## Blogs

* [PyTorch 1.6 now includes Stochastic Weight Averaging](https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/)
* [Introducing native PyTorch automatic mixed precision for faster training on NVIDIA GPUs](https://pytorch.org/blog/accelerating-training-on-nvidia-gpus-with-pytorch-automatic-mixed-precision/)
* [Introduction to Quantization on PyTorch](https://pytorch.org/blog/introduction-to-quantization-on-pytorch/)


## Recipes
Recipes with tag `Basic`would be a good start. Below are recommended recipes:

* (Recommended) **Saving and loading**
  * [what is a state_dic in pytorch](https://pytorch.org/tutorials/recipes/recipes/what_is_state_dict.html)
  * [saving and loading models for inference in pytorch](https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_models_for_inference.html)
  * [saving and loading a general checkpoint in pytorch](https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html)
  * [saving and loading multiple models in one file using pytorch](https://pytorch.org/tutorials/recipes/recipes/saving_multiple_models_in_one_file.html)
  * [saving and loading models across devices in pytorch](https://pytorch.org/tutorials/recipes/recipes/save_load_across_devices.html)

* (Recommended)  **Data customization**
  * [developing custom pytorch dataloader](https://pytorch.org/tutorials/recipes/recipes/custom_dataset_transforms_loader.html)

* **Monitoring the time and memory consumption** 
  * [pytorch profiler](https://pytorch.org/tutorials/recipes/recipes/profiler.html#pytorch-profiler) 

* **Tranfer learning: warmstart a model**
  * [warmstarting model using parameters from a different model in pytorch](https://pytorch.org/tutorials/recipes/recipes/warmstarting_model_using_parameters_from_a_different_model.html) 

* **Model optimization & quantization**
  * [dynamic quantization](https://pytorch.org/tutorials/recipes/recipes/dynamic_quantization.html#dynamic-quantization)

* **Model deployment**
  * [deploying with Flask](https://pytorch.org/tutorials/recipes/deployment_with_flask.html)
    * related tutorial: [Deploying pytorch in python via a rest API with Flask](https://pytorch.org/tutorials/intermediate/flask_rest_api_tutorial.html)
  * [TorchScript for deployment](https://pytorch.org/tutorials/recipes/torchscript_inference.html)
    * related tutorial: [Introduction to TorchScript](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html#introduction-to-torchscript)

* **Interpretability**
  * [model interpretability using Captum](https://pytorch.org/tutorials/recipes/recipes/Captum_Recipe.html)


## Advance Tutorials

* **Distributed training**
  * [PyTorch distributed overview](https://pytorch.org/tutorials/beginner/dist_overview.html)
  * [Single-machine model parallel best practices](https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html)
  * [Getting started with distributed data parallel](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
  * [Writing distributed applications with pytorch](https://pytorch.org/tutorials/intermediate/dist_tuto.html)
  * [Getting started with distributed RPC framework](https://pytorch.org/tutorials/intermediate/rpc_tutorial.html)
  * [Implementing a parameter server using distributed RPC framework](https://pytorch.org/tutorials/intermediate/rpc_param_server_tutorial.html)
  * [Distributed pipeline parallelism using RPC](https://pytorch.org/tutorials/intermediate/dist_pipeline_parallel_tutorial.html)
  * [Implementing batch RPC processing using asynchronous executions](https://pytorch.org/tutorials/intermediate/rpc_async_execution.html)
  * [Combining distributed dataparallel with distributed RPC framework](https://pytorch.org/tutorials/advanced/rpc_ddp_tutorial.html)
