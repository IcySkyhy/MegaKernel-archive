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

from typing import List, Optional, Union

from .. import pypto_impl
from .._op_wrapper import op_wrapper
from ..enum import CastMode, DataType, SaturationMode
from ..symbolic_scalar import SymbolicScalar
from ..tensor import Tensor


@op_wrapper
def permute(input_tensor: Tensor, dims: List[int]) -> Tensor:
    """Returns a tensor that is a permuted version of `input`. The dimensions are reordered according to `dims`.

    Parameters
    ----------
    input_tensor : Tensor
        The input tensor.
    dims : List[int]
        The desired ordering of dimensions. Must be a permutation of all dimensions indices (0 to input.dim() - 1).

    Returns
    -------
    Tensor
        A new tensor with the same data as `input` but with dimensions reordered as specified.

    Raises
    ------
    RuntimeError
        If `dims` is not a valid permutation of input dimensions.
        If the length of `dims` does not match the input tensor's dimension.

    Examples
    --------
    x = pypto.tensor([2, 3, 4], pypto.DT_FP32)
    out = pypto.permute(x, [2, 0, 1])

    Input x:    shape [2, 3, 4]
    Output out: shape [4, 2, 3]  # data permuted accordingly

    # For 2D transpose (equivalent to transpose(x, 0, 1)):
    y = pypto.permute(x, [1, 0])
    """
    return pypto_impl.Permute(input_tensor, dims)


@op_wrapper
def transpose(input: Tensor, dim0: int, dim1: int) -> Tensor:
    """Returns a tensor that is a transposed version of `input`. The given dimensions `dim0` and `dim1` are swapped.

    Parameters
    ----------
    input : Tensor
        The input tensor.
    dim0 : int
        The first dimension to be transposed.
    dim1 : int
        The second dimension to be transposed.

    Returns
    -------
    Tensor
        A new tensor that is a transposed version of `input`.

    Raises
    ------
    RuntimeError
        If dim0 or dim1 is greater than or equal to the input dimension.

    Examples
    --------
    x = pypto.tensor([2, 3], pypto.DT_FP32)
    out = pypto.transpose(x, 0, 1)

    Input x:    [[ 1.0028 -0.9893 0.5809],
                 [-0.1669 0.7299  0.4942]]
    Output out: [[ 1.0028 -0.1669],
                 [-0.9893 0.7299],
                 [ 0.5809 0.4942]]
    """
    return pypto_impl.Transpose(input, [dim0, dim1])


@op_wrapper
def pack(input: Tensor) -> Tensor:
    """Flattens a tensor and reinterprets its raw bytes as uint8 elements.

    Parameters
    ----------
    input : Tensor
        The input tensor.

    Returns
    -------
    Tensor
        A new tensor that is a flattened version of `input`.

    Examples
    --------
    x = pypto.tensor([1, 2], pypto.DT_INT32)
    out = pypto.pack(x)

    Input x:    [[ 0x03040506 0x01050608]]
    Output out: [  6  5  4  3  8  6  5  1  ]
    """
    return pypto_impl.Pack(input)


@op_wrapper
def unpack(input: Tensor, dtype: DataType) -> Tensor:
    """Reinterprets packed uint8 bytes back to the original data type.

    Parameters
    ----------
    input : Tensor
        The input tensor of dtype DT_UINT8 containing packed raw bytes.
    dtype : DataType
        The target data type to unpack to.

    Returns
    -------
    Tensor
        A new tensor with the unpacked elements of the specified dtype.

    See Also
    --------
    pack : Reinterpret raw bytes of a tensor as uint8 elements.

    Examples
    --------
    x = pypto.tensor([6, 5, 4, 3, 8, 6, 5, 1], pypto.DT_UINT8)
    out = pypto.unpack(x, pypto.DT_INT32)

    Input x:  [6  5  4  3  8  6  5  1]
    Output out: [50595078  17105416]
    """
    return pypto_impl.UnPack(input, dtype)


@op_wrapper
def cast(
    input: Tensor, dtype: DataType, mode: CastMode = CastMode.CAST_NONE, satmode: SaturationMode = SaturationMode.OFF
) -> Tensor:
    """Casting the operand to the specified type.

    Parameters
    ----------
    input : Tensor
        The input tensor.
    dtype : DataType
        The desired type.
    mode : CastMode, optional
        The rounding mode for the cast operation. Default is CAST_NONE.
    satmode : SaturationMode, optional
        The saturation mode for float to integer conversions.
        Default is OFF (truncation behavior). Use ON for saturation (clamping).

    Returns
    -------
    Tensor
        Return a tensor after type cast, if this is already of the correct type, no copy is performed and the
        original object is returned

    Raises
    ------
    RuntimeError
        If the two tensors are not broadcastable to a common shape.

    Examples
    --------
    x = pypto.tensor([2], pypto.DT_FP32)
    y = pypto.cast(x, pypto.DT_FP16)

    # With saturation mode
    x = pypto.tensor([300.0, -300.0, 50.0], pypto.DT_FP16)
    y = pypto.cast(x, pypto.DT_INT8, satmode=pypto.SaturationMode.ON)
    # Values will be clamped to [-128, 127] range

    Input  x: [2.0, 3.0] x.dtype: pypto.DT_FP32
    Output y: [2.0, 3.0] y.dtype: pypto.DT_FP16
    """
    if dtype == input.dtype and mode == CastMode.CAST_NONE and satmode == SaturationMode.OFF:
        return input
    else:
        return pypto_impl.Cast(input, dtype, mode, satmode)


@op_wrapper
def expand_clone(
    input: Tensor, shape: List[int], *, valid_shape: Optional[List[Union[int, SymbolicScalar]]] = None
) -> Tensor:
    """
    Broadcast the input tensor along the axis where it is uniquely to 1 to match shape.A deep copy will be performed,
    and a new tensor that actually occupies memory will be returned.

    Parameters
    ----------
    input : Tensor
        The input tensor will be broadcasted.
    shape : List[int]
        Target shape.
    valid_shape : List[int] | List[SymbolicScalar]]
        Keyword argument, used for dynamic graph, represent the actual shapes at runtime.
        They can be ommitted in static graph.

    Examples
    --------
    x = pypto.tensor([1,3], pypto.DT_INT32)
    y = pypto.expand_clone(x, [3,4])

    Input  x: [[1], [2], [3]]
    Output y: [[ 1,  1,  1,  1],
               [ 2,  2,  2,  2],
               [ 3,  3,  3,  3]]

    """
    if valid_shape is None:
        valid_shape = []
    return pypto_impl.Expand(input, shape, valid_shape)
