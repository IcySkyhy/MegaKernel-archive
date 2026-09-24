#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""PyPTO"""

from typing import Union

from .. import pypto_impl
from .._element import Element
from .._op_wrapper import op_wrapper
from ..error import PyptoError
from ..tensor import Tensor


@op_wrapper
def amin(input: Tensor, dim: int, keepdim: bool = False) -> Tensor:
    """Returns the minimum value of each slice of the input tensor in the given dimension dim.

    Parameters
    ----------
    input : Tensor
        The input tensor.
    dim : int
        The dimension to reduce.
    keepdim : bool, optional
        whether the output tensor has dim retained or not. Default: False.

    Returns
    -------
    Tensor
        If keepdim is True, the return tensor is of the same size as input except in the dimension dim
        where it is of size 1.
        Otherwise, dim is squeezed, resulting in the return tensor having 1 fewer dimension.

    Examples
    --------
    x = pypto.tensor([2, 3], pypto.DT_FP32)
    y = pypto.amin(x, -1, True)

    Input x:[[1.0 2.0 3.0],
             [1.0 2.0 3.0]]
    Output y:[[1.0],
              [1.0]]

    """
    return pypto_impl.Amin(input, dim, keepdim)


@op_wrapper
def amax(input: Tensor, dim: int, keepdim: bool = False) -> Tensor:
    """Returns the maximum value of each slice of the input tensor in the given dimension dim.

    Parameters
    ----------
    input : Tensor
        The input tensor.
    dim : int
        The dimension to reduce.
    keepdim : bool
        whether the output tensor has dim retained or not. Default: False.

    Returns
    -------
    Tensor
        If keepdim is True, the return tensor is of the same size as input except in the dimension dim where it
        is of size 1.
        Otherwise, dim is squeezed, resulting in the return tensor having 1 fewer dimension.

    Examples
    --------
    x = pypto.tensor([2, 3], pypto.DT_FP32)
    y = pypto.amax(x, -1, True)

    Input x:[[1.0 2.0 3.0],
             [1.0 2.0 3.0]]
    Output y:[[3.0],
              [3.0]]

    """
    return pypto_impl.Amax(input, dim, keepdim)


@op_wrapper
def maximum(input: Union[Tensor, Element, int, float], other: Union[Tensor, Element, int, float]) -> Tensor:
    """
    Computes the element-wise maximum of input and other.

    Parameters
    ----------
    input : Tensor
        The first input tensor.
    other : Tensor or Element
        The second input tensor.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise maximum.

    Examples
    --------
    a = pypto.tensor([3], pypto.DT_INT32)
    b = pypto.tensor([3], pypto.DT_INT32)
    out = pypto.maximum(a, b)

    Input a:    [0 2 4]
    Input b:    [3 1 3]
    Output out: [3 2 4]
    """
    if not isinstance(input, pypto_impl.Tensor) and not isinstance(other, pypto_impl.Tensor):
        raise PyptoError(0xF00001, TypeError("one of `input` and `other` should be `Tensor`"))

    if not isinstance(input, pypto_impl.Tensor) and isinstance(other, pypto_impl.Tensor):
        input, other = other, input
    if isinstance(other, (int, float)):
        other = pypto_impl.Element(input.dtype, other)
    return pypto_impl.Maximum(input, other)


@op_wrapper
def minimum(input: Union[Tensor, Element, int, float], other: Union[Tensor, Element, int, float]) -> Tensor:
    """
    Computes the element-wise minimum of input and other.

    Parameters
    ----------
    input : Tensor
        The first input tensor.
    other : Tensor or Element
        The second input tensor.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise minimum.

    Examples
    --------
    a = pypto.tensor([3], pypto.DT_INT32)
    b = pypto.tensor([3], pypto.DT_INT32)
    out = pypto.minimum(a, b)

    Input a:    [0 2 4]
    Input b:    [3 1 3]
    Output out: [0 1 3]
    """
    if not isinstance(input, pypto_impl.Tensor) and not isinstance(other, pypto_impl.Tensor):
        raise PyptoError(0xF00001, TypeError("one of `input` and `other` should be `Tensor`"))

    if not isinstance(input, pypto_impl.Tensor) and isinstance(other, pypto_impl.Tensor):
        input, other = other, input
    if isinstance(other, (int, float)):
        other = pypto_impl.Element(input.dtype, other)
    return pypto_impl.Minimum(input, other)


@op_wrapper
def sum(input: Tensor, dim: int, keepdim: bool = False) -> Tensor:
    """Returns the sum value of each slice of the input tensor in the given dimension dim.

    Parameters
    ----------
    input : Tensor
        The input tensor.
    dim : int
        The dimension to reduce.
    keepdim : bool, optional
        whether the output tensor has dim retained or not. Default: False.

    Returns
    -------
    Tensor
        If keepdim is True, the return tensor is of the same size as input except in the dimension dim where it
        is of size 1.
        Otherwise, dim is squeezed, resulting in the return tensor having 1 fewer dimension.

    Examples
    --------
    x = pypto.tensor([2, 3], pypto.DT_FP32)
    y = pypto.sum(x, -1, True)

    Input x:[[1.0 2.0 3.0],
             [1.0 2.0 3.0]]
    Output y:[[6.0],
              [6.0]]

    """
    return pypto_impl.Sum(input, dim, keepdim)


@op_wrapper
def prod(self: Tensor, dim: int, keepdim: bool = False) -> Tensor:
    """Returns the prod value of each slice of the input tensor in the given dimension dim.

    Parameters
    ----------
    self : Tensor
        The input tensor.
    dim : int
        The dimension to reduce.
    keepdim : bool, optional
        whether the output tensor has dim retained or not. Default: False.

    Returns
    -------
    Tensor
        If keepdim is True, the return tensor is of the same size as input except in the dimension dim where it
        is of size 1.
        Otherwise, dim is squeezed, resulting in the return tensor having 1 fewer dimension.

    Examples
    --------
    x = pypto.tensor([2, 3], pypto.DT_FP32)
    y = pypto.prod(x, -1, True)

    Input x:[[1.0 2.0 3.0],
             [1.0 2.0 3.0]]
    Output y:[[6.0],
              [6.0]]

    """
    return pypto_impl.Prod(self, dim, keepdim)


@op_wrapper
def argmax(input: Tensor, dim: int = -1, keepdim: bool = False) -> Tensor:
    """
    Returns the index of the maximum value in a tensor along a given dimension.

    Parameters
    ----------
    input : Tensor
        The input tensor.
    dim : int, optional
        The dimension along which to find the maximum value. Default is -1 (last dimension).
    keepdim : bool, optional
        Whether the output tensor has dim retained. Default is False.

    Returns
    -------
    Tensor
        A new tensor containing the indices of the maximum values.

    Examples
    --------
    x = pypto.tensor([[1, 3, 2], [4, 1, 5]], pypto.DT_FP32)
    y = pypto.argmax(x, dim=1)

    Input x:  [[1.0, 3.0, 2.0],
              [4.0, 1.0, 5.0]]
    Output y: [1, 2]
    """
    return pypto_impl.ArgMax(input, dim, keepdim)


@op_wrapper
def argmin(input: Tensor, dim: int = -1, keepdim: bool = False) -> Tensor:
    """
    Returns the index of the minimum value in a tensor along a given dimension.

    Parameters
    ----------
    input : Tensor
        The input tensor.
    dim : int, optional
        The dimension along which to find the minimum value. Default is -1 (last dimension).
    keepdim : bool, optional
        Whether the output tensor has dim retained. Default is False.

    Returns
    -------
    Tensor
        A new tensor containing the indices of the minimum values.

    Examples
    --------
    x = pypto.tensor([[1, 3, 2], [4, 1, 5]], pypto.DT_FP32)
    y = pypto.argmin(x, dim=1)

    Input x:  [[1.0, 3.0, 2.0],
              [4.0, 1.0, 5.0]]
    Output y: [0, 1]
    """
    return pypto_impl.ArgMin(input, dim, keepdim)
