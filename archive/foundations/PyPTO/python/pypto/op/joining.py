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

from typing import List, Optional, Tuple

from .. import pypto_impl
from .._op_wrapper import op_wrapper
from ..tensor import Tensor


@op_wrapper
def concat(tensors: List[Tensor], dim: int = 0) -> Tensor:
    """
    Concatenate multiple tensors according to the specified dimension.

    Parameters
    ---------
    tensors: Tensors
        tensor to be spliced.

    dim : int
        specified dimensions.

    out: Tensor
        The concatenated tensor
    Examples
    ---------
    x = pypto.tensor([2, 2], pypto.data_type.DT_FP32)  # 2x2 tensor with all 1s
    y = pypto.tensor([2, 2], pypto.data_type.DT_FP32)  # 2x2 tensor with all 0s
    dim = 0
    out = pypto.concat([x, y], dim)

    Input  x : [[1.0 1.0],
                [1.0 1.0]]
           y : [[0.0 0.0],
                [0.0 0.0]]

    Output out:[[1.0 1.0],
                [1.0 1.0],
                [0.0 0.0],
                [0.0 0.0]]
    """
    return pypto_impl.Cat(tensors, dim)


@op_wrapper
def interleave(input: Tensor, other: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Interleave two tensors into two output tensors.

    Parameters
    ---------
    input: Tensor
        The first input tensor.
    other: Tensor
        The second input tensor. It must have the same shape as input.

    Returns
    -------
    tuple
        A tuple of two tensors.
    """
    return pypto_impl.Interleave(input, other)


@op_wrapper
def deinterleave(input: Tensor, other: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
    """
    Deinterleave one or two input tensors into two output tensors.

    Parameters
    ---------
    input: Tensor
        The input tensor. In single-input mode, its last dimension is twice the output last dimension.
    other: Tensor, optional
        The second input tensor. If provided, it must have the same shape as input and the outputs.

    Returns
    -------
    tuple
        A tuple of two tensors.
    """
    if other is None:
        return pypto_impl.DeInterleave(input)
    return pypto_impl.DeInterleave(input, other)
