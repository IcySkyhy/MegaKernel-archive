#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""PyPTO"""

from typing import Optional, Tuple

from .. import pypto_impl
from .._op_wrapper import op_wrapper
from ..enum import DataType, DequantScaleRoundingMode
from ..tensor import Tensor


@op_wrapper
def quantize(
    input: Tensor,
    scale: Tensor,
    otype: DataType,
    axis: int,
    zero_points: Optional[Tensor] = None,
) -> Tensor:
    """Quantize fp32 tensor to int8/uint8.

    Converts high-precision floating-point data to low-precision format.

    Parameters
    ----------
    input : Tensor
        Source operand. dtype must be DT_FP32.
        Shape: [..., row, col], 1-4 dimensions supported.
    scale : Tensor
        Scaling factor. dtype must be DT_FP32.
        When axis=-1: shape is [..., row]
        When axis=-2: shape is [..., col]
    otype : DataType
        Output data type. DT_INT8 for symmetric, DT_UINT8 for asymmetric.
    axis : int
        Quantization axis. Supports -1, -2 or relative dimensions.
        When input is 1D, only -1 is supported.
    zero_points : Tensor, optional
        Zero point offset for asymmetric quantization.

    Returns
    -------
    Tensor
        Quantized tensor with specified otype.

    Examples
    --------
    x = pypto.tensor([3, 4], pypto.DT_FP32)
    scale = pypto.tensor([3], pypto.DT_FP32)
    zero_points = pypto.tensor([3], pypto.DT_FP32)

    # Symmetric quantization: fp32 -> int8
    y1 = pypto.quantize(x, scale, pypto.DT_INT8, -1)

    # Asymmetric quantization: fp32 -> uint8
    y2 = pypto.quantize(x, scale, pypto.DT_UINT8, -1, zero_points)
    """
    if zero_points is None:
        zero_points = pypto_impl.Tensor()
    return pypto_impl.Quantize(input, scale, otype, axis, zero_points)


@op_wrapper
def dequantize(
    input: Tensor,
    scale: Tensor,
    otype: DataType,
    axis: int,
    zero_points: Optional[Tensor] = None,
) -> Tensor:
    """Dequantize int8/uint8 tensor to fp32.

    Converts low-precision quantized data back to high-precision floating-point format.

    Parameters
    ----------
    input : Tensor
        Source operand. dtype must be DT_INT8 or DT_UINT8.
        Shape: [..., row, col], 1-4 dimensions supported.
    scale : Tensor
        Scaling factor. dtype must be DT_FP32.
        When axis=-1: shape is [..., row]
        When axis=-2: shape is [..., col]
    otype : DataType
        Output data type. Typically DT_FP32.
    axis : int
        Dequantization axis. Supports -1, -2 or relative dimensions.
        When input is 1D, only -1 is supported.
    zero_points : Tensor, optional
        Zero point offset for asymmetric dequantization.

    Returns
    -------
    Tensor
        Dequantized tensor with specified otype.

    Examples
    --------
    x = pypto.tensor([3, 4], pypto.DT_INT8)
    scale = pypto.tensor([3], pypto.DT_FP32)
    zero_points = pypto.tensor([3], pypto.DT_FP32)

    # Symmetric dequantization: int8/int16 -> fp32
    y1 = pypto.dequantize(x, scale, pypto.DT_FP32, -1)

    # Asymmetric dequantization: int8/int16 -> fp32
    y2 = pypto.dequantize(x, scale, pypto.DT_FP32, -1, zero_points)
    """
    if zero_points is None:
        zero_points = pypto_impl.Tensor()
    return pypto_impl.Dequantize(input, scale, otype, axis, zero_points)


@op_wrapper
def quant_mx(
    input: Tensor,
    quant_dtype: DataType = DataType.DT_FP8E4M3,
    mode: DequantScaleRoundingMode = DequantScaleRoundingMode.ROUND_DOWN,
    axis: int = -1,
    performance_mode: bool = True,
) -> Tuple[Tensor, Tensor]:
    """Quantizes a 1D to 4D FP16/BF16/FP32 ND tensor to MX format.

    `quant_dtype` supports `DT_FP8E4M3` for FP16/BF16/FP32 input and `DT_FP4_E2M1X2`
    for FP16/BF16 input.

    Returns
    -------
    tuple
        A tuple of `(quantized, scale)` where:
        - `quantized` has the same shape as `input` and dtype `quant_dtype`
        - when `axis=-1`, `scale` has shape
          `[*input.shape[:-1], ceil(input.shape[-1] / 64), 2]` and dtype `DT_FP8E8M0`
        - when `axis=-2`, internal raw exp has shape
          `[*input.shape[:-2], input.shape[-2] / 64, input.shape[-1] * 2]` and is returned as
          public `scale` with shape `[*input.shape[:-2], input.shape[-2] / 64, input.shape[-1], 2]`

    Notes
    -----
    `axis` may be the last or second-last dimension using either a positive or negative index.
    `mode` defaults to `ROUND_DOWN` (OCP); `ROUND_UP` selects the NV scale algorithm.
    For `axis=-1`, the quantization-axis tile and view width must be 256-byte aligned.
    For `axis=-2`, the input must have at least two dimensions. The second-last dimension of
    the current input tensor/view shape, runtime valid shape, and tile shape must be a positive
    multiple of 64, with no tail block on that axis. Quantization still uses groups of 32; the
    final scale dimension stores the two consecutive group exponents in each 64-element block.
    FP8 does not require 256-byte alignment on the last dimension. FP4 requires the view and
    tile last dimensions to be multiples of 64 so each packed row is 32-byte aligned.
    `performance_mode` is accepted for compatibility and does not change the DN layout.
    """

    return pypto_impl.QuantMX(input, quant_dtype, mode, axis, performance_mode)
