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

from typing import List, Optional, Sequence, Union, overload

from .. import pypto_impl
from .._element import Element
from .._op_wrapper import op_wrapper
from .._utils import to_syms
from ..enum import DataType
from ..error import PyptoError
from ..symbolic_scalar import SymbolicScalar
from ..tensor import Tensor


def convert_to_element(value) -> pypto_impl.Element:
    if isinstance(value, (int)):
        if value >= -(2**31) and value <= 2**31 - 1:
            return pypto_impl.Element(pypto_impl.DT_INT32, value)
        else:
            return pypto_impl.Element(pypto_impl.DT_INT64, value)
    else:
        return pypto_impl.Element(pypto_impl.DT_FP32, value)


@overload
def arange(end: Union[int, float]) -> Tensor:
    """
    Creates a 1-dimensional tensor containing a sequence of values from 0 (inclusive)
    to 'end'(exclusive), with a step size of 1

    Parameters
    ----------
    end : Number
        The ending value of the sequence (exclusive).

    Returns
    -------
    Tensor
        A 1-dimensional tensor containing the sequence of values.

    Raises
    ------
    ValueError
        If 'end' less than 0.

    Examples
    --------
    a = pypto.arange(4)
    b = pypto.arange(5.5)

    Output a: [0 1 2 3]
    Output b: [0.0 1.0 2.0 3.0 4.0 5.0]
    """
    ...


@overload
def arange(start: Union[int, float], end: Union[int, float]) -> Tensor:
    """
    Creates a 1-dimensional tensor containing a sequence of values from start (inclusive)
    to 'end'(exclusive), with a step size of 1

    Parameters
    ----------
    start : Number
        The starting value of the sequence (inclusive).
    end : Number
        The ending value of the sequence (exclusive).

    Returns
    -------
    Tensor
        A 1-dimensional tensor containing the sequence of values.

    Raises
    ------
    ValueError
        If 'end' less than 'start'.

    Examples
    --------
    a = pypto.arange(-2, 2)
    b = pypto.arange(1.0, 4.0)

    Output a: [-2 -1 0 1]
    Output b: [1.0 2.0 3.0]
    """
    ...


@overload
def arange(start: Union[int, float], end: Union[int, float], step: Union[int, float]) -> Tensor:
    """
    Creates a 1-dimensional tensor containing a sequence of values in the range [start, end) with a given step.

    Parameters
    ----------
    start : Number
        The starting value of the sequence (inclusive).
    end : Number
        The ending value of the sequence (exclusive).
    step : Number
        The step size between consecutive values.

    Returns
    -------
    Tensor
        A 1-dimensional tensor containing the sequence of values.

    Raises
    ------
    ValueError
        If 'step' is zero or directionally incorrect (e.g., step > 0 when start > end).

    Examples
    --------
    a = pypto.arange(1.0, 4.0, 0.5)
    b = pypto.arange(10, 0, -2)

    Output a: [1.0 1.5 2.0 2.5 3.0 3.5]
    Output b: [10 8 6 4 2]
    """

    ...


@op_wrapper
def arange(*args: Union[int, float]) -> Tensor:
    """Creates a 1-dimensional tensor containing a sequence of values in the range [start, end) with a given step.

    This function generates values from 'start' to 'end' (exclusive) in increments of 'step'.
    If only one argument is provided, it is treated as 'end', and 'start' defaults to 0 (INT32) and 'step'
    defaults to 1 (INT32).
    If two arguments are provided, they are treated as 'start' and 'end', and 'step' defailts to 1 (INT32).

    Parameters
    ----------
    start : Number
        The starting value of the sequence (inclusive), default is 0.
    end : Number
        The ending value of the sequence (exclusive).
    step : Number
        The step size between consecutive values, default is 1.

    Returns
    -------
    Tensor
        A 1-dimensional tensor containing the sequence of values.

    Raises
    ------
    ValueError
        If 'step' is zero or directionally incorrect (e.g., step > 0 when start > end).

    Examples
    --------
    a = pypto.arange(1.0, 4.0, 0.5)
    b = pypto.arange(1.0, 4.0)
    c = pypto.arange(4)

    Output a: [1.0 1.5 2.0 2.5 3.0 3.5]
    Output b: [1.0 2.0 3.0]
    Output c: [0 1 2 3]
    """
    if len(args) == 1:
        end = args[0]
        return pypto_impl.Range(
            pypto_impl.Element(pypto_impl.DataType.DT_INT32, 0),
            convert_to_element(end),
            pypto_impl.Element(pypto_impl.DataType.DT_INT32, 1),
        )

    if len(args) == 2:
        start, end = args
        return pypto_impl.Range(
            convert_to_element(start),
            convert_to_element(end),
            pypto_impl.Element(pypto_impl.DataType.DT_INT32, 1),
        )

    if len(args) != 3:
        raise PyptoError(0xF00002, ValueError(f"The length of args should in [1, 2, 3], but got {len(args)}."))

    start, end, step = args
    return pypto_impl.Range(convert_to_element(start), convert_to_element(end), convert_to_element(step))


_UNSIGNED_INT_DTYPES = {DataType.DT_UINT8, DataType.DT_UINT16, DataType.DT_UINT32, DataType.DT_UINT64}


def _check_full_fill_value_unsigned(fill_value, dtype: DataType) -> None:
    if isinstance(fill_value, pypto_impl.Element):
        elem_dtype = fill_value._get_data_type()
        if elem_dtype not in _UNSIGNED_INT_DTYPES:
            return
        elem_val = fill_value._get_signed_data() if not fill_value._is_float() else fill_value._get_float_data()
    elif isinstance(fill_value, (int, float)):
        elem_dtype = dtype
        elem_val = fill_value
    else:
        return
    if elem_dtype in _UNSIGNED_INT_DTYPES and elem_val < 0:
        raise PyptoError(
            0xF00002,
            ValueError(
                f"full() does not support negative fill_value for unsigned integer dtype: "
                f"fill_value is {elem_val} but dtype is {elem_dtype}. "
                f"Negative values cannot be represented in unsigned integer types."
            ),
        )


@op_wrapper
def full(
    size: List[int],
    fill_value: Union[int, float, SymbolicScalar, Element],
    dtype: DataType,
    *,
    valid_shape: Optional[List[Union[int, SymbolicScalar]]] = None,
) -> Tensor:
    """
    Creates a tensor of the specified shape whose every entry equals the scalar elem.

    Parameters
    ----------
    size : List[int]
        target shape; must be non-negative integers
    fill_value : int | float | SymbolicScalar | pypto.element
        scalar value to replicate
    dtype : pypto.DataType
        desired data type; only int/float are supported (DT_FP32, DT_INT32).
        If elem is a SymbolicScalar, dtype must be int32.
    valid_shape : List[int] | List[SymbolicScalar]]
        runtime actual shape

    Returns
    -------
    Tensor of shape shape filled with elem.

    Examples
    --------
    # Valid shapes use keyword argument
    x1 = 1.0
    y1 = pypto.full([2,2], x1, pypto.DT_FP32, valid_shape=[2, 2])

    x2 = pypto.Element(pypto.DT_INT32, 1)
    y2 = pypto.full([2,2], x2, pypto.DT_INT32, valid_shape=[2, 2])

    #  In static graphs, validshape can be ignored
    x3 = 1
    y3 = pypto.full([2,2], x3, pypto.DT_INT32)

    Output y1: [[1.0 1.0], [1.0 1.0]]
    Output y2: [[1 1], [1 1]]
    Output y3: [[1 1], [1 1]]
    """

    if valid_shape is None:
        valid_shape = []
    _check_full_fill_value_unsigned(fill_value, dtype)
    if isinstance(fill_value, pypto_impl.SymbolicScalar):
        return pypto_impl.Full(fill_value, dtype, size, to_syms(valid_shape))
    elif isinstance(fill_value, pypto_impl.Element):
        return pypto_impl.Full(fill_value, dtype, size, to_syms(valid_shape))
    else:
        return pypto_impl.Full(pypto_impl.Element(dtype, fill_value), dtype, size, to_syms(valid_shape))


@op_wrapper
def zeros(*size: Union[int, Sequence[int]], dtype: Optional[DataType] = None) -> Tensor:
    """
    Returns a tensor filled with the scalar value 0, with the shape defined by the variable argument `size`.

    Parameters
    ----------
    size : int or sequence of ints
        Dimensionalities of the tensor. Can be multiple integer arguments or a single sequence.
    dtype : DataType, optional
        The desired data type of returned tensor. Default: DT_FP32.

    Returns
    -------
    Tensor
        A tensor filled with zeros.

    Examples
    --------
    x1 = pypto.zeros(2, 3)
    Output x1 [[0., 0., 0.],
               [0., 0., 0.]])
    """
    if len(size) == 1 and isinstance(size[0], (list, tuple)):
        shape = list(size[0])
    else:
        shape = list(size)

    if dtype is None:
        dtype = pypto_impl.DataType.DT_FP32
    zero_element = pypto_impl.Element(dtype, 0)
    return pypto_impl.Full(zero_element, dtype, shape, to_syms([]))


@op_wrapper
def ones(*size: Union[int, Sequence[int]], dtype: Optional[DataType] = None) -> Tensor:
    """
    Returns a tensor filled with the scalar value 1, with the shape defined by the variable argument `size`.

    Parameters
    ----------
    size : int or sequence of ints
        Dimensionalities of the tensor. Can be multiple integer arguments or a single sequence.
    dtype : DataType, optional
        The desired data type of returned tensor. Default: DT_FP32.

    Returns
    -------
    Tensor
        A tensor filled with ones.

    Examples
    --------
    x1 = pypto.ones(2, 3)

    Output x1 [[1., 1., 1.],
               [1., 1., 1.]]
    """
    if len(size) == 1 and isinstance(size[0], (list, tuple)):
        shape = list(size[0])
    else:
        shape = list(size)

    if dtype is None:
        dtype = pypto_impl.DataType.DT_FP32
    one_element = pypto_impl.Element(dtype, 1)
    return pypto_impl.Full(one_element, dtype, shape, to_syms([]))
