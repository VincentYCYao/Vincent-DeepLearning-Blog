---
layout: post
title:  "PyTorch view op: change data in original place"
date: 2020-08-27 15:00:00 +0800
categories: python
---



Reference: [TENSOR VIEWS](https://pytorch.org/docs/stable/tensor_view.html)

Two classes of **PyTorch op**:

* **normal op**: returns a new tensor as output, e.g. [`add()`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.add)

* **view op**: outputs are views of input tensors

> No data movement occurs when creating a view, view tensor just changes the way it interprets the same data. Taking a view of contiguous tensor could potentially produce a non-contiguous tensor. Users should be pay additional attention as contiguity might have implicit performance impact. [`transpose()`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.transpose) is a common example.
>

# Tensor indexing

> When accessing the contents of a tensor via indexing, PyTorch follows Numpy behaviors that **basic indexing returns views**, while advanced indexing returns a copy. Assignment via either basic or advanced indexing is in-place. See more examples in [Numpy indexing documentation](https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html).

# Other tensor operations

* [`reshape()`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.reshape) and [`reshape_as()`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.reshape_as) can return either a view or new tensor, user code shouldn’t rely on whether it’s view or not
* [`contiguous()`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.contiguous) returns **itself** if input tensor is already contiguous, otherwise it returns a new contiguous tensor by copying data.

# A list of view op

* [`as_strided()`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.as_strided)
* [`detach()`](https://pytorch.org/docs/stable/autograd.html#torch.Tensor.detach)
* [`diagonal()`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.diagonal)
* [`expand()`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.expand)
* [`expand_as()`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.expand_as)
* [`narrow()`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.narrow)
* [`permute()`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.permute)
* [`select()`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.select)
* [`squeeze()`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.squeeze)
* [`transpose()`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.transpose)
* [`t()`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.t)
* [`T`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.T)
* [`real`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.real)
* [`imag`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.imag)
* `view_as_real()`
* `view_as_imag()`
* [`unfold()`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.unfold)
* [`unsqueeze()`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.unsqueeze)
* [`view()`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view)
* [`view_as()`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view_as)
* [`unbind()`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.unbind)
* [`split()`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.split)
* [`chunk()`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.chunk)
* [`indices()`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.indices) (sparse tensor only)
* [`values()`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.values) (sparse tensor only)

# PyTorch internal implementation

For a more detailed walk-through of PyTorch internal implementation, please refer to [ezyang’s blogpost about PyTorch Internals](http://blog.ezyang.com/2019/05/pytorch-internals/).

