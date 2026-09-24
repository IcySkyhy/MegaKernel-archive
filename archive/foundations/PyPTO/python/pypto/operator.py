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
__all__ = ["sigmoid", "softmax", "rms_norm"]

import pypto
from pypto import Tensor


def sigmoid(input: Tensor) -> Tensor:
    """Return a tensor containing the element-wise sigmoid values of input.
        The sigmoid function is a common activation function in machine learning,
        defined mathematically as: sigmoid(x) = 1 / (1 + exp(-x))

    Parameters
    ----------
    input: Tensor
        The input tensor to compute.
        The supported data type is DT_FP32.
        Empty tensors are not supported, and the shape size must not exceed 2147483647 (i.e., INT32_MAX).

    Returns
    -------
    Tensor
        A tensor with the same shape and data type as the input, whose elements are
        the results of the input elements mapped to the interval (0, 1) via the sigmoid function.

    Examples
    --------
    x = pypto.tensor([4], pypto.DT_FP32)
    y = pypto.sigmoid(x)

    Input x:[-3.0, 0.0, 2.0, 5.0]
    Output y:[0.0474, 0.5000, 0.8808, 0.9933]
    """
    codegen_options = pypto.get_codegen_options()

    soc_version = ""
    if "soc_version" in codegen_options:
        soc_version = codegen_options["soc_version"]

    dtype = input.dtype
    f_1 = 1.0
    f_nega_1 = -1.0

    if soc_version in ["Kirin9030", "KirinX90"]:
        exp_res = pypto.exp(pypto.mul(input, f_nega_1))
        res = pypto.add(exp_res, f_1)
        ones = pypto.full(res.shape, 1.0, dtype, valid_shape=res.valid_shape)
        res = pypto.div(ones, res, pypto.PrecisionType.INTRINSIC)
    else:
        input = pypto.cast(input, pypto.DT_FP32)
        exp_res = pypto.exp(pypto.mul(input, f_nega_1))
        res = pypto.add(exp_res, f_1)
        ones = pypto.full(res.shape, 1.0, pypto.DT_FP32, valid_shape=res.valid_shape)
        res = pypto.div(ones, res, pypto.PrecisionType.INTRINSIC)

        if dtype != pypto.DT_FP32:
            res = pypto.cast(res, dtype)
    return res


def softmax(input: Tensor, dim: int) -> Tensor:
    """Return a tensor obtained by applying the softmax activation function to the input.
        Mathematically, for an input tensor x along a specified dimension dim,
        the softmax of element x_i is computed as:
        softmax(x_i) = exp(x_i) / sum(exp(x_j) for j in dimension dim

    Parameters
    ----------
    input: Tensor
        The input tensor to compute.
        The supported data type is DT_FP32.
        Empty tensors are not supported, and the shape size must not exceed 2147483647 (i.e., INT32_MAX).
    dim: int
        Specify the dimension for normalization.
        Negative indices are supported (e.g., -1 indicates the last dimension).
        It must be within the range of [-input.dim, input.dim - 1].

    Returns
    -------
    Tensor
        A tensor with the same shape as the input, where the sum of elements along the
        specified dimension is 1, and the data type is determined by dtype or the input type.

    Examples
    --------
    x = pypto.tensor([2, 3], pypto.DT_FP32)
    y = pypto.softmax(x, -1)

    Input x:[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    Output y:[[0.0900, 0.2447, 0.6652], [0.0900, 0.2447, 0.6652]]
    """
    dtype = input.dtype
    input = pypto.cast(input, pypto.DT_FP32)

    rowmax = pypto.amax(input, dim, True)
    sub_res = pypto.sub(input, rowmax)
    exp_res = pypto.exp(sub_res)
    esum = pypto.sum(exp_res, dim, True)
    output = pypto.div(exp_res, esum, pypto.PrecisionType.INTRINSIC)

    if dtype != pypto.DT_FP32:
        output = pypto.cast(output, dtype)
    return output


def rms_norm(input: Tensor, gamma: Tensor = None, epsilon: float = 1e-6) -> Tensor:
    """
    Root Mean Square LayerNorm (RMSNorm) along the last dimension.
    If `gamma` is provided, applies an element-wise scale on the last dim.

    Parameters
    ----------
    input : Tensor
        Input tensor. Any shape (..., C).
    gamma : Tensor | None
        Optional scale of shape (C,).
    epsilon : float
        Numerical stability constant (default: 1e-6).

    Returns
    -------
    Tensor
        Same shape as `input`, cast back to the original dtype.

    Examples
    --------
    x = pypto.tensor([2, 4], pypto.DT_FP32)
    gamma = pypto.tensor([4], pypto.DT_FP32)
    y = pypto.rms_norm(x, gamma)

    Input x: [[1, 2, 3, 4],
              [5, 6, 7, 8]]
          gamma: [1, 1, 1, 1]
    Output y: [[0.3651, 0.7302, 1.0954, 1.4605],
               [0.7580, 0.9097, 1.0613, 1.2129]]
    """
    in_dtype = input.dtype
    x = pypto.cast(input, pypto.DT_FP32)

    n = x.shape[-1]

    y = pypto.sqrt(pypto.sum(x * x * (1.0 / n), -1, keepdim=True) + epsilon)

    ones = pypto.full(y.shape, 1.0, pypto.DT_FP32)
    y = pypto.div(x * ones, y, pypto.PrecisionType.INTRINSIC)

    if gamma is not None:
        rank = input.dim
        shape = [1] * rank
        shape[-1] = gamma.shape[0]
        g = pypto.cast(pypto.reshape(gamma, shape), pypto.DT_FP32)
        y *= g

    if in_dtype != pypto.DT_FP32:
        y = pypto.cast(y, in_dtype)
    return y
