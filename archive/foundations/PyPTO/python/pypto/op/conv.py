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

from dataclasses import dataclass
from typing import Any, Type

from .. import pypto_impl
from .._op_wrapper import op_wrapper
from ..enum import DataType
from ..error import PyptoError
from ..tensor import Tensor

_VALID_DATA_TYPES = (pypto_impl.DataType.DT_BF16, pypto_impl.DataType.DT_FP16, pypto_impl.DataType.DT_FP32)


@dataclass
class ConvParams:
    out_dtype: Any
    input_conv: Tensor
    weight: Tensor
    strides: list
    paddings: list
    dilations: list
    groups: int
    transposed: bool
    output_paddings: list
    extend_params: Any


@op_wrapper
def conv(
    input_conv,
    weight,
    out_dtype,
    strides,
    paddings,
    dilations,
    *,
    groups=1,
    transposed=False,
    output_paddings=None,
    extend_params=None,
) -> Tensor:
    """
    Performs convolution operation with support for 1D/2D/3D convolution, grouped convolution, transposed convolution
    (not yet supported) and extended features like bias addition, quantization/dequantization, and activation.

    Supports three convolution dimensions:
    1. 1D convolution: Applies to 3-D input tensor (NCL format), operates on width (W) dimension.
    2. 2D convolution: Applies to 4-D input tensor (NCHW format), operates on height/width (HW) dimensions.
    3. 3D convolution: Applies to 5-D input tensor (NCDHW format), operates on depth/height/width (DHW) dimensions.

    Note: Input/weight tensors support multiple format layouts (aligned with TensorFlow/NCHW standards), and transposed
    convolution is currently a placeholder (not implemented). Paddings are specified as bidirectional values (e.g.,
    top/bottom/left/right for 2D conv), while strides/dilations are unidirectional.

    Parameters
    ----------
    input_conv : Tensor
        The input feature map tensor, supporting 3-D (1D conv), 4-D (2D conv), 5-D (3D conv) shapes.
        Supported formats: NCL, NCHW, NCDHW (corresponding to TensorFlow's NLC, NHWC, NDHWC).
        - 1D conv shape: (N, Cin, W)
        - 2D conv shape: (N, Cin, H, W)
        - 3D conv shape: (N, Cin, D, H, W)
    weight : Tensor
        The convolution kernel tensor, matching the convolution dimension (1D/2D/3D) of the input.
        Supported formats: LCN (1D), HWCN (2D), DHWCN (3D) (corresponding to TensorFlow's layout):
        - 1D conv shape: (Cout, Cin, K) (LCN: K=kernel width)
        - 2D conv shape: (Cout, Cin, Kh, Kw) (HWCN: Kh=kernel height, Kw=kernel width)
        - 3D conv shape: (Cout, Cin, Kd, Kh, Kw) (DHWCN: Kd=kernel depth, Kh=kernel height, Kw=kernel width)
    out_dtype : dtype
        The data type of the output tensor. Same as input dtype in normal scenarios; configured for quantization/
        dequantization/requantization when fixpipe is enabled.
    strides : list/tuple of int
        Unidirectional stride values for convolution, with length matching the convolution dimension:
        - 1D conv: [stride_w] (width dimension)
        - 2D conv: [stride_h, stride_w] (height/width dimensions)
        - 3D conv: [stride_d, stride_h, stride_w] (depth/height/width dimensions)
    paddings : list/tuple of int
        Bidirectional padding values for convolution, length is 2×convolution dimension:
        - 1D conv: [padding_w_left, padding_w_right] (width dimension, length=2)
        - 2D conv: [padding_h_top, padding_h_bottom, padding_w_left, padding_w_right] (height/width, length=4)
        - 3D conv: [padding_d_front, padding_d_back, padding_h_top, padding_h_bottom,
                    padding_w_left, padding_w_right] (depth/height/width, length=6)
    dilations : list/tuple of int
        Unidirectional dilation rates for convolution, with length matching the convolution dimension:
        - 1D conv: [dilation_w] (width dimension)
        - 2D conv: [dilation_h, dilation_w] (height/width dimensions)
        - 3D conv: [dilation_d, dilation_h, dilation_w] (depth/height/width dimensions)

    Keyword Arguments
    ----------
    groups : int, default=1
        Number of groups for grouped convolution. Input channels and weight channels must be divisible by groups.
    transposed : bool, default=False
        If True, perform transposed convolution (deconvolution). Currently not supported.
    output_paddings : list/tuple of int, default=[]
        Output padding values for transposed convolution, only used when `transposed=True`.
        Length matches the convolution dimension (1D/2D/3D).
    extend_params : dict, optional
        A dictionary specifying extended computation features for fixpipe (quantization/dequantization pipeline):
        - 'bias': Tensor
            Optional bias tensor to add to the convolution output, shape must be [Cout] (number of output channels).
        - 'scale': float
            Per-tensor scale value for dequantization, quantization, or requantization operations.
        - 'scale_tensor': Tensor
            Per-channel scale tensor for dequantization or requantization operations.
        - 'relu_type': ConvReLuType
            Type of ReLU activation to apply in fixpipe (e.g., ConvReLuType.RELU).

    Returns
    -------
    Tensor
        A new tensor containing the convolution result.
        - 1D conv output shape: (N, Cout, W_out)
        - 2D conv output shape: (N, Cout, H_out, W_out)
        - 3D conv output shape: (N, Cout, D_out, H_out, W_out)

    Raises
    ------
    RuntimeError
        If input/weight dimensions are invalid for the specified convolution type, or if transposed convolution
        is requested (not yet supported).
    ValueError
        If input parameters (strides/paddings/dilations) have invalid lengths, or if groups is not a divisor of
        input/weight channels.

    Examples
    --------
    # 1D convolution (basic)
    input_conv = pypto.tensor((1, 32, 16), pypto.DT_FP16, "input")
    weight = pypto.tensor((32, 32, 1), pypto.DT_FP16, "weight")
    pypto.conv(input_conv, weight, pypto.DT_FP16, strides=[1], paddings=[0,0], dilations=[1])

    # 2D convolution (matching test code)
    input_conv = pypto.tensor((1, 32, 8, 16), pypto.DT_FP16, "input")
    weight = pypto.tensor((32, 32, 1, 1), pypto.DT_FP16, "weight")
    pypto.conv(input_conv, weight, pypto.DT_FP16, strides=[1,1], paddings=[0,0,0,0], dilations=[1,1])

    # 3D convolution (matching test code)
    input_conv = pypto.tensor((1, 96, 2, 16, 16), pypto.DT_FP16, "input")
    weight = pypto.tensor((32, 96, 1, 1, 1), pypto.DT_FP16, "weight")
    pypto.conv(input_conv, weight, pypto.DT_FP16, strides=[1,1,1], paddings=[0,0,0,0,0,0], dilations=[1,1,1])

    # 2D convolution with bias
    input_conv = pypto.tensor((1, 32, 8, 16), pypto.DT_FP16, "input")
    weight = pypto.tensor((32, 32, 1, 1), pypto.DT_FP16, "weight")
    bias = pypto.tensor((32,), pypto.DT_FP16, "bias")
    extend_params = {'bias_tensor': bias}
    pypto.conv(input_conv, weight, pypto.DT_FP16, strides=[1,1], paddings=[0,0,0,0],
               dilations=[1,1], extend_params=extend_params)
    """
    params = ConvParams(
        out_dtype=out_dtype,
        input_conv=input_conv,
        weight=weight,
        strides=strides,
        paddings=paddings,
        dilations=dilations,
        groups=groups,
        transposed=transposed,
        output_paddings=output_paddings,
        extend_params=extend_params,
    )
    __validate_inputs(params)

    if extend_params is not None:
        extend_params = pypto_impl.ConvExtendParam(**__convert_conv_extend_params(extend_params))

    if not transposed:
        return pypto_impl.Conv(
            params.out_dtype,
            params.input_conv,
            params.weight,
            params.strides,
            params.paddings,
            params.dilations,
            extend_params,
            params.groups,
        )
    else:
        raise PyptoError(0xF00003, RuntimeError("Conv transpose true is not supported yet."))


def __validate_type(value: Any, expect_type: Type, arg_name: str = "input") -> None:
    if value is None:
        return
    if not isinstance(value, expect_type):
        raise PyptoError(
            0xF00001,
            TypeError(f"Argument '{arg_name}' must be of type {expect_type.__name__}, but got {type(value).__name__}."),
        )


def __validate_shape(input_conv: Tensor, weight: Tensor, transposed: bool) -> None:
    input_dim = input_conv.Dim()
    weight_dim = weight.Dim()
    if input_dim != weight_dim or input_dim not in {3, 4, 5}:
        raise PyptoError(
            0xF00003,
            RuntimeError(
                "Tensor dimension mismatch. Expect input_dim == weight_dim and both in [3, 4, 5], "
                f"got input_dim: {input_dim}, weight_dim: {weight_dim}."
            ),
        )


def __validate_inputs(params: ConvParams) -> None:
    __validate_type(params.input_conv, pypto_impl.Tensor, "input_conv")
    __validate_type(params.weight, pypto_impl.Tensor, "weight")
    __validate_type(params.out_dtype, DataType, "out_dtype")
    __validate_type(params.strides, list, "strides")
    __validate_type(params.paddings, list, "paddings")
    __validate_type(params.dilations, list, "dilations")
    __validate_type(params.groups, int, "groups")
    __validate_type(params.transposed, bool, "transposed")
    __validate_type(params.output_paddings, list, "output_paddings")
    __validate_type(params.extend_params, dict, "extend_params")
    __validate_shape(params.input_conv, params.weight, False)

    if params.extend_params is not None and 'bias_tensor' in params.extend_params:
        bias = params.extend_params['bias_tensor']
        if bias is not None:
            __validate_type(bias, pypto_impl.Tensor, "bias_tensor")

    if params.input_conv.GetDataType() not in _VALID_DATA_TYPES:
        raise PyptoError(
            0xF00002,
            ValueError(
                "Input tensor data type must in [bf16, fp16, fp32],"
                f"but Input tensor got {params.input_conv.GetDataType()}"
            ),
        )

    if params.weight.GetDataType() not in _VALID_DATA_TYPES:
        raise PyptoError(
            0xF00002,
            ValueError(
                "Weight tensor data type must in [bf16, fp16, fp32],"
                f"but Weight tensor got {params.weight.GetDataType()}"
            ),
        )

    __validate_data_type_consistency(params)


def __validate_data_type_consistency(params: ConvParams) -> None:
    if params.input_conv.GetDataType() != params.weight.GetDataType():
        raise PyptoError(
            0xF00002,
            ValueError(
                f"Input and weight data types must be consistent, "
                f"but got input: {params.input_conv.GetDataType()}, weight: {params.weight.GetDataType()}"
            ),
        )

    if params.out_dtype != params.input_conv.GetDataType():
        raise PyptoError(
            0xF00002,
            ValueError(
                f"Output data type must be consistent with input, "
                f"but got out_dtype: {params.out_dtype}, input: {params.input_conv.GetDataType()}"
            ),
        )

    if params.extend_params is not None and 'bias_tensor' in params.extend_params:
        bias = params.extend_params['bias_tensor']
        if bias is not None and hasattr(bias, 'GetDataType'):
            input_dtype = params.input_conv.GetDataType()
            bias_dtype = bias.GetDataType()
            expected_bias_dtype = input_dtype
            if pypto_impl.GetNPUArch() == 'DAV_2201' and input_dtype == pypto_impl.DataType.DT_BF16:
                expected_bias_dtype = pypto_impl.DataType.DT_FP32
            if bias_dtype != expected_bias_dtype:
                raise PyptoError(
                    0xF00002,
                    ValueError(
                        f"Bias data type must be {expected_bias_dtype}, "
                        f"but got bias: {bias_dtype}, input: {input_dtype}"
                    ),
                )


def __convert_conv_extend_params(extend_params) -> dict:
    extend_params.setdefault('bias_tensor', pypto_impl.Tensor())
    extend_params.setdefault('scale_tensor', pypto_impl.Tensor())
    extend_params.setdefault('relu_type', pypto_impl.ConvReLuType.NO_RELU)
    extend_params.setdefault('scale', 0.0)
    return extend_params
