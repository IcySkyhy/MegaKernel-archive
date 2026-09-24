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

from typing import Optional, Tuple, Union

from .. import pypto_impl
from .._element import Element
from .._op_wrapper import op_wrapper
from ..enum import OpType, OutType, TopKAlgo
from ..tensor import Tensor


@op_wrapper
def greater(input: Tensor, other: Union[Tensor, float, Element]) -> Tensor:
    """Performs element-wise comparison between `input` and `other`.

    Parameters
    ----------
    input : Tensor
        The first input tensor.
    other : Tensor
        The second input tensor or a scalar value for comparison.

    Returns
    -------
    Tensor
        A boolean tensor that is True where input is greater than other and False elsewhere.
        BOOL tensor with same shape as inputs

    Raises
    ------
    TypeError
        If `other` is neither a Tensor nor a scalar number.


    Examples
    --------
    a = pypto.tensor([3], pypto.DT_FP32)
    b = pypto.tensor([3], pypto.DT_FP32)
    out = pypto.greater(a, b)

    Input a:    [1.0 2.0 3.0]
    Input b:    [2.0 2.0 2.0]
    Output out: [False False True]

    """
    if isinstance(other, float):
        # Tensor vs Scalar comparison
        return pypto_impl.Compare(input, pypto_impl.Element(input.dtype, other), OpType.GT, OutType.BOOL)
    return pypto_impl.Compare(input, other, OpType.GT, OutType.BOOL)


@op_wrapper
def gt(input: Tensor, other: Union[Tensor, float, Element]) -> Tensor:
    """Performs element-wise comparison between `input` and `other`.

    Parameters
    ----------
    input : Tensor
        The first input tensor.
    other : Tensor
        The second input tensor or a scalar value for comparison.

    Returns
    -------
    Tensor
        A boolean tensor that is True where input is greater than other and False elsewhere.
        BOOL tensor with same shape as inputs

    Raises
    ------
    TypeError
        If `other` is neither a Tensor nor a scalar number.


    Examples
    --------
    a = pypto.tensor([3], pypto.DT_FP32)
    b = pypto.tensor([3], pypto.DT_FP32)
    out = pypto.gt(a, b)

    Input a:    [1.0 2.0 3.0]
    Input b:    [2.0 2.0 2.0]
    Output out: [False False True]

    """
    if isinstance(other, float):
        # Tensor vs Scalar comparison
        return pypto_impl.Compare(input, pypto_impl.Element(input.dtype, other), OpType.GT, OutType.BOOL)
    return pypto_impl.Compare(input, other, OpType.GT, OutType.BOOL)


@op_wrapper
def ge(input: Tensor, other: Union[Tensor, float, Element]) -> Tensor:
    """Performs element-wise comparison between `input` and `other`.

    Parameters
    ----------
    input : Tensor
        The first input tensor.
    other : Tensor
        The second input tensor or a scalar value for comparison.

    Returns
    -------
    Tensor
        A boolean tensor that is True where input is greater_equal than other and False elsewhere.
        BOOL tensor with same shape as inputs

    Raises
    ------
    TypeError
        If `other` is neither a Tensor nor a scalar number.


    Examples
    --------
    a = pypto.tensor([3], pypto.DT_FP32)
    b = pypto.tensor([3], pypto.DT_FP32)
    out = pypto.ge(a, b)

    Input a:    [1.0 2.0 3.0]
    Input b:    [2.0 2.0 2.0]
    Output out: [False True True]

    """
    if isinstance(other, float):
        # Tensor vs Scalar comparison
        return pypto_impl.Compare(input, pypto_impl.Element(input.dtype, other), OpType.GE, OutType.BOOL)
    return pypto_impl.Compare(input, other, OpType.GE, OutType.BOOL)


@op_wrapper
def eq(input: Tensor, other: Union[Tensor, float, Element]) -> Tensor:
    """Performs element-wise comparison between `input` and `other`.

    Parameters
    ----------
    input : Tensor
        The first input tensor.
    other : Tensor
        The second input tensor or a scalar value for comparison.

    Returns
    -------
    Tensor
        A boolean tensor that is True where input is equal other and False elsewhere.
        BOOL tensor with same shape as inputs

    Raises
    ------
    TypeError
        If `other` is neither a Tensor nor a scalar number.


    Examples
    --------
    a = pypto.tensor([3], pypto.DT_FP32)
    b = pypto.tensor([3], pypto.DT_FP32)
    out = pypto.eq(a, b)

    Input a:    [1.0 2.0 3.0]
    Input b:    [2.0 2.0 2.0]
    Output out: [False True False]

    """
    if isinstance(other, float):
        # Tensor vs Scalar comparison
        return pypto_impl.Compare(input, pypto_impl.Element(input.dtype, other), OpType.EQ, OutType.BOOL)
    return pypto_impl.Compare(input, other, OpType.EQ, OutType.BOOL)


@op_wrapper
def ne(input: Tensor, other: Union[Tensor, float, Element]) -> Tensor:
    """Performs element-wise comparison between `input` and `other`.

    Parameters
    ----------
    input : Tensor
        The first input tensor.
    other : Tensor
        The second input tensor or a scalar value for comparison.

    Returns
    -------
    Tensor
        A boolean tensor that is True where input is not equal other and False elsewhere.
        BOOL tensor with same shape as inputs

    Raises
    ------
    TypeError
        If `other` is neither a Tensor nor a scalar number.


    Examples
    --------
    a = pypto.tensor([3], pypto.DT_FP32)
    b = pypto.tensor([3], pypto.DT_FP32)
    out = pypto.ne(a, b)

    Input a:    [1.0 2.0 3.0]
    Input b:    [2.0 2.0 2.0]
    Output out: [True False True]

    """
    if isinstance(other, float):
        # Tensor vs Scalar comparison
        return pypto_impl.Compare(input, pypto_impl.Element(input.dtype, other), OpType.NE, OutType.BOOL)
    return pypto_impl.Compare(input, other, OpType.NE, OutType.BOOL)


@op_wrapper
def lt(input: Tensor, other: Union[Tensor, float, Element]) -> Tensor:
    """Performs element-wise comparison between `input` and `other`.

    Parameters
    ----------
    input : Tensor
        The first input tensor.
    other : Tensor
        The second input tensor or a scalar value for comparison.

    Returns
    -------
    Tensor
        A boolean tensor that is True where input is less than other and False elsewhere.
        BOOL tensor with same shape as inputs

    Raises
    ------
    TypeError
        If `other` is neither a Tensor nor a scalar number.


    Examples
    --------
    a = pypto.tensor([3], pypto.DT_FP32)
    b = pypto.tensor([3], pypto.DT_FP32)
    out = pypto.lt(a, b)

    Input a:    [1.0 2.0 3.0]
    Input b:    [2.0 2.0 2.0]
    Output out: [True False False]

    """
    if isinstance(other, float):
        # Tensor vs Scalar comparison
        return pypto_impl.Compare(input, pypto_impl.Element(input.dtype, other), OpType.LT, OutType.BOOL)
    return pypto_impl.Compare(input, other, OpType.LT, OutType.BOOL)


@op_wrapper
def le(input: Tensor, other: Union[Tensor, float, Element]) -> Tensor:
    """Performs element-wise comparison between `input` and `other`.

    Parameters
    ----------
    input : Tensor
        The first input tensor.
    other : Tensor
        The second input tensor or a scalar value for comparison.

    Returns
    -------
    Tensor
        A boolean tensor that is True where input is less equal than other and False elsewhere.
        BOOL tensor with same shape as inputs

    Raises
    ------
    TypeError
        If `other` is neither a Tensor nor a scalar number.


    Examples
    --------
    a = pypto.tensor([3], pypto.DT_FP32)
    b = pypto.tensor([3], pypto.DT_FP32)
    out = pypto.le(a, b)

    Input a:    [1.0 2.0 3.0]
    Input b:    [2.0 2.0 2.0]
    Output out: [True True False]

    """
    if isinstance(other, float):
        # Tensor vs Scalar comparison
        return pypto_impl.Compare(input, pypto_impl.Element(input.dtype, other), OpType.LE, OutType.BOOL)
    return pypto_impl.Compare(input, other, OpType.LE, OutType.BOOL)


@op_wrapper
def topk(
    input: Tensor, k: int, dim: Optional[int] = None, largest: bool = True, algo: TopKAlgo = TopKAlgo.MERGE_SORT
) -> Tuple[Tensor, Tensor]:
    """Returns the k largest elements of the given input tensor along a given dimension.

    Parameters
    ----------
    input : Tensor
        The input tensor.
    k : int
        The k in "top-k".
    dim : int, optional
        The dimension to sort along, if dim is not given, the last dimension of the input is chosen.
    largest : bool
        Controling whether to return the elements in sorted order, if largest is False then the k smallest
        elements are returned.

    Returns
    -------
    tuple
        A tuple of (values, indices) is returned with the values and indices of the largest k elements of
        each row of the input tensor in the given dimension dim.

    Raises
    ------
    RuntimeError
        If the two tensors are not broadcastable to a common shape.

    Examples
    --------
    x = pypto.tensor([2, 3], pypto.DT_FP32)
    y = pypto.topk(x, 2, -1, True)

    Input x:     [[1.0 2.0 3.0],
                  [1.0 2.0 3.0]]
    Output y[0]: [[3.0 2.0],
                  [3.0 2.0]]
    Output y[1]: [[2 1],
                  [2 1]]
    """

    return pypto_impl.TopK(input, k, (-1 if dim is None else dim), largest, algo)


@op_wrapper
def argsort(input: Tensor, dim: Optional[int] = None, descending: bool = False) -> Tensor:
    """
    Returns the indices that sort a tensor along a given dimension.

    Parameters
    ----------
    input : Tensor
        The input tensor.
    dim : int, optional
        The dimension to sort along, if dim is not given, the last dimension of the input is chosen.
    descending : bool
        Controls the order of sorting.
        If `True`, the tensor will be sorted in descending order;
        if `False`, it will be sorted in ascending order.

    Returns
    -------
    Tensor
        The indices tensor along a given dimension in descending or ascending.

    Examples
    --------
    x = pypto.tensor([2, 3], pypto.DT_FP32)
    y = pypto.argsort(x, -1, True)

    Input x:     [[1.0 2.0 3.0],
                  [1.0 2.0 3.0]]
    Output y: [[2 1 0],
               [2 1 0]]
    """
    return pypto_impl.ArgSort(input, (-1 if dim is None else dim), descending)


@op_wrapper
def sort32(input: Tensor, index: Optional[int] = None) -> Tensor:
    """
    Returns interleaved the values and indices that sort every 32 numbers.

    Parameters
    ----------
    input : Tensor
        The input tensor.
    index : int, optional
        The start index of the sorting dimension.

    Returns
    -------
    Tensor
        The output tensor.

    Examples
    --------
    x = pypto.tensor([2, 3], pypto.DT_FP32)
    y = pypto.sort32(x, 0)

    Input x:     [[1.0 2.0 3.0],
                  [1.0 2.0 3.0]]
    Output y: [[3.0 2 2.0 1 1.0 0],
               [3.0 2 2.0 1 1.0 0]]
    """
    return pypto_impl.Sort32(input, (0 if index is None else index))


@op_wrapper
def mrgsort(input: Tensor, mergesize: int) -> Tensor:
    """
    Returns the tensor that merge sort base on merge size.

    Parameters
    ----------
    input : Tensor
        The input tensor.
    mergesize : int
        The start index of the sorting dimension.

    Returns
    -------
    Tensor
        The output tensor.
    """
    return pypto_impl.MrgSort(input, mergesize)
