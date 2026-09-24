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

from typing import List, Optional, Tuple, Union, overload

import numpy as np

import pypto

from .. import pypto_impl
from .._element import Element
from .._op_wrapper import op_wrapper
from ..enum import CastMode, DataType, PrecisionType
from ..error import PyptoError
from ..symbolic_scalar import SymInt
from ..tensor import Tensor

_FLOAT_DTYPES = {DataType.DT_FP32, DataType.DT_FP16, DataType.DT_BF16}

_INT_DTYPE_MAP = {
    DataType.DT_INT8: np.int8,
    DataType.DT_INT16: np.int16,
    DataType.DT_INT32: np.int32,
    DataType.DT_UINT8: np.uint8,
    DataType.DT_UINT16: np.uint16,
}


def _check_scalar_type(op_name, tensor_dtype, other):
    if not isinstance(other, (int, float)):
        return
    if isinstance(other, float) and tensor_dtype not in _FLOAT_DTYPES:
        raise PyptoError(
            0xF00001,
            TypeError(
                f"{op_name}(): float scalar incompatible with integer tensor dtype {tensor_dtype.name}. "
                "Use an integer scalar or cast the tensor to float type."
            ),
        )


def _clip_scalar_to_dtype(tensor_dtype, value):
    if isinstance(value, int) and tensor_dtype in _INT_DTYPE_MAP:
        return int(_INT_DTYPE_MAP[tensor_dtype](value))
    return value


@op_wrapper
def add(input_tensor: Tensor, other: Union[Tensor, float, int]) -> Tensor:
    """Computes the element-wise addition of `input_tensor` and `other`.

    This function calculates the formula: `out = input_tensor + other`.
    It supports broadcasting between the input tensors.

    Parameters
    ----------
    input_tensor : Tensor
        The first input tensor.
    other : Tensor or Number
        The second input tensor or a scalar to be added.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise sum.

    Raises
    ------
    RuntimeError
        If the two tensors are not broadcastable to a common shape.

    See Also
    --------
    sub : The inverse operation, element-wise subtraction.
    mul : Element-wise multiplication.

    Examples
    --------
    a = pypto.tensor([1, 3], pypto.DT_FP32)
    b = pypto.tensor([1, 3], pypto.DT_FP32)
    out = pypto.add(a, b)

    Input a:    [[1.0 2.0 3.0]]
    Input b:    [[2.0 3.0 4.0]]
    Output out: [[3.0 5.0 7.0]]
    """
    if isinstance(other, pypto_impl.Tensor):
        return pypto_impl.Add(input_tensor, other)
    else:
        _check_scalar_type("add", input_tensor.dtype, other)
        other = _clip_scalar_to_dtype(input_tensor.dtype, other)
        return pypto_impl.Add(input_tensor, pypto_impl.Element(input_tensor.dtype, other))


@op_wrapper
def axpy_(y: Tensor, x: Tensor, alpha: Union[int, float] = 1.0) -> Tensor:
    """Computes the element-wise AXPY operation: y = alpha * x + y.

    This function performs an in-place update on tensor y.
    The formula is: `y = alpha * x + y`.
    It supports broadcasting for tensor x (but y cannot broadcast).

    Parameters
    ----------
    y : Tensor
        The destination tensor that will be updated in-place.
    x : Tensor
        The source tensor. Can be broadcast to match y's shape.
    alpha : float, optional
        A scaling factor for the `x` tensor. Default is 1.0.

    Returns
    -------
    Tensor
        The updated y tensor (same as input y, modified in-place).

    Raises
    ------
    RuntimeError
        If y's shape cannot accommodate x's broadcast.
        If dtype combination is not supported (only fp32+fp32, fp16+fp16, bf16+bf16, fp32+fp16).

    Notes
    -----
    - This is an in-place operation. The y tensor is modified directly.
    - Supported dtype combinations:
      - fp32 + fp32
      - fp16 + fp16 (converted to fp32 internally, cast back with CAST_RINT)
      - fp32 + fp16 (mixed precision, x casted to fp32 internally)

    Examples
    --------
    y = pypto.tensor([1, 3], pypto.DT_FP32)
    x = pypto.tensor([1, 3], pypto.DT_FP32)
    y.axpy_(x, alpha=2.0)

    Input y:    [[1.0 2.0 3.0]]
    Input x:    [[2.0 3.0 4.0]]
    alpha:      2.0
    Output:     [[5.0 8.0 11.0]]  (y = 2.0 * x + y)
    """
    if y.dtype == DataType.DT_FP32 and x.dtype == DataType.DT_FP16:
        x = pypto_impl.Cast(x, DataType.DT_FP32, CastMode.CAST_NONE)
    elif y.dtype == DataType.DT_FP16 and x.dtype == y.dtype:
        y_fp32 = pypto_impl.Cast(y, DataType.DT_FP32, CastMode.CAST_NONE)
        x_fp32 = pypto_impl.Cast(x, DataType.DT_FP32, CastMode.CAST_NONE)
        result_fp32 = pypto_impl.Axpy(y_fp32, x_fp32, float(alpha))
        y.Move(pypto_impl.Cast(result_fp32, y.dtype, CastMode.CAST_RINT))
        return y
    y.Move(pypto_impl.Axpy(y, x, float(alpha)))
    return y


@op_wrapper
def sub(input_tensor: Tensor, other: Union[Tensor, float, int]) -> Tensor:
    """Computes the element-wise subtraction of `input_tensor` and `other`.

    This function calculates the formula: `out = input_tensor - other`.
    It supports broadcasting between the input tensors.

    Parameters
    ----------
    input_tensor : Tensor
        The first input tensor.
    other : Tensor or Number
        The second input tensor or a scalar to be subtracted.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise subtraction.

    Raises
    ------
    RuntimeError
        If the two tensors are not broadcastable to a common shape.

    Examples
    --------
    x = pypto.tensor([2, 3], pypto.DT_FP32)
    y = pypto.tensor([2, 3], pypto.DT_FP32)
    out1 = pypto.sub(a, b)

    Input x:      [[9.0 9.0 9.0],
                   [9.0 9.0 9.0]]
    Input y:      [[1.0 2.0 3.0],
                   [1.0 2.0 3.0]]
    Output out1 : [[8.0 7.0 6.0],
                   [8.0 7.0 6.0]]
    """
    if isinstance(other, pypto_impl.Tensor):
        return pypto_impl.Sub(input_tensor, other)
    else:
        _check_scalar_type("sub", input_tensor.dtype, other)
        other = _clip_scalar_to_dtype(input_tensor.dtype, other)
        return pypto_impl.Sub(input_tensor, pypto_impl.Element(input_tensor.dtype, other))


@op_wrapper
def mul(input_tensor: Tensor, other: Union[Tensor, float, int]) -> Tensor:
    """Computes the element-wise multiplication of `input` and `other`.

    This function calculates the formula: `out = input * other`.
    It supports broadcasting between the input tensors.

    Parameters
    ----------
    input : Tensor
        The first input tensor.
    other : Tensor or Number
        The second input tensor or a scalar to be multiplied.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise multiplication.

    Raises
    ------
    RuntimeError
        If the two tensors are not broadcastable to a common shape.

    Examples
    --------
    x = pypto.tensor([2, 3], pypto.DT_FP32)
    y = pypto.tensor([2, 3], pypto.DT_FP32)
    z = pypto.mul(a, b)

    Input x:[[1.0 2.0 3.0],
             [1.0 2.0 3.0]]
    Input y:[[1.0 2.0 3.0],
             [1.0 2.0 3.0]]
    Output z:[[1.0 4.0 9.0],
              [1.0 4.0 9.0]]
    """
    if isinstance(other, pypto_impl.Tensor):
        return pypto_impl.Mul(input_tensor, other)
    else:
        _check_scalar_type("mul", input_tensor.dtype, other)
        other = _clip_scalar_to_dtype(input_tensor.dtype, other)
        return pypto_impl.Mul(input_tensor, pypto_impl.Element(input_tensor.dtype, other))


@op_wrapper
def div(
    input: Tensor, other: Union[Tensor, float, int], precision_type: PrecisionType = PrecisionType.HIGH_PRECISION
) -> Tensor:
    """Computes the element-wise division of `input` and `other`.

    This function calculates the formula: `out = input / other`.
    It supports broadcasting between the input tensors.

    Parameters
    ----------
    input : Tensor
        The first input tensor.
    other : Tensor or Number
        The second input tensor or a scalar to divide.
    precision_type : PrecisionType, optional
        The precision type for division. Default is PrecisionType.HIGH_PRECISION.
        HIGH_PRECISION uses higher precision calculation to reduce precision loss.
        Use PrecisionType.INTRINSIC to directly use chip instructions.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise division.

    Raises
    ------
    RuntimeError
        If the two tensors are not broadcastable to a common shape.

    See Also
    --------
    sub : The inverse operation, element-wise subtraction.
    mul : Element-wise multiplication.

    Examples
    --------
    a = pypto.tensor([1, 3], pypto.DT_FP32)
    b = pypto.tensor([1, 3], pypto.DT_FP32)
    out = pypto.div(a, b)

    Input a:    [[2.0 4.0 6.0]]
    Input b:    [[2.0 2.0 2.0]]
    Output out: [[1.0 2.0 3.0]]

    # Using high precision mode for FP16
    a = pypto.tensor([1, 3], pypto.DT_FP16)
    b = pypto.tensor([1, 3], pypto.DT_FP16)
    out = pypto.div(a, b, pypto.PrecisionType.HIGH_PRECISION)
    """
    if isinstance(other, pypto_impl.Tensor):
        return pypto_impl.Div(input, other, precision_type)
    else:
        _check_scalar_type("div", input.dtype, other)
        other = _clip_scalar_to_dtype(input.dtype, other)
        return pypto_impl.Div(input, pypto_impl.Element(input.dtype, other), precision_type)


@op_wrapper
def hypot(self: Tensor, other: Tensor) -> Tensor:
    """Computes the hypotenuse of a right-angled triangle given its legs.

    This function calculates the element-wise operation: `out = sqrt(self^2 + other^2)`.
    It supports broadcasting between the input tensors.

    Parameters
    ----------
    self : Tensor
        The first input tensor.
    other : Tensor or Number
        The second input tensor.

    Returns
    -------
    Tensor
        A new tensor containing the hypotenuse values.

    Raises
    ------
    RuntimeError
        If the two tensors are not broadcastable to a common shape.

    See Also
    --------
    sqrt : Element-wise square root.
    pow : Element-wise power.

    Examples
    --------
    a = pypto.tensor([3.0, 4.0], pypto.DT_FP32)
    b = pypto.tensor([4.0, 3.0], pypto.DT_FP32)
    out = pypto.hypot(a, b)

    Input a:    [3.0 4.0]
    Input b:    [4.0 3.0]
    Output out: [5.0 5.0]
    """
    return pypto_impl.Hypot(self, other)


@op_wrapper
def fmod(
    input: Tensor,
    other: Union[Tensor, float],
    precision_type: PrecisionType = PrecisionType.HIGH_PRECISION,
) -> Tensor:
    """Computes the element-wise modulus of `input` and `other`.

    This function calculates the formula: `out = input % other`.
    It supports broadcasting between the input tensors.

    Parameters
    ----------
    input : Tensor
        The first input tensor.
    other : Tensor or Number
        The second input tensor or a scalar to modulo operation.
    precision_type : PrecisionType, optional
        The precision type for modulo. Default is PrecisionType.HIGH_PRECISION.
        HIGH_PRECISION uses higher precision calculation to reduce precision loss.
        Use PrecisionType.INTRINSIC to directly use chip instructions.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise modulus.

    Raises
    ------
    RuntimeError
        If the two tensors are not broadcastable to a common shape.

    See Also
    --------
    sub : The inverse operation, element-wise subtraction.
    mul : Element-wise multiplication.

    Examples
    --------
    a = pypto.tensor([1, 3], pypto.DT_FP32)
    b = pypto.tensor([1, 3], pypto.DT_FP32)
    out = pypto.fmod(a, b)

    Input a:    [[2.0 5.0 9.0]]
    Input b:    [[2.0 2.0 2.0]]
    Output out: [[0.0 1.0 1.0]]
    """
    if isinstance(other, pypto_impl.Tensor):
        return pypto_impl.Fmod(input, other, precision_type)
    if isinstance(other, float):
        if np.isnan(other):
            raise PyptoError(0xF00002, ValueError("fmod(): other must not be NaN."))
        if np.isinf(other):
            raise PyptoError(0xF00002, ValueError("fmod(): other must not be inf."))
    _check_scalar_type("fmod", input.dtype, other)
    other = _clip_scalar_to_dtype(input.dtype, other)
    return pypto_impl.Fmod(input, pypto_impl.Element(input.dtype, other), precision_type)


@op_wrapper
def lrelu(other: Tensor, negative_slope: Union[float, Element] = 0.01) -> Tensor:
    """
    Returns a new tensor with the Leaky Rectified Linear Unit (LReLU) function applied element-wise.

    The function is defined as:
    y = x if x >= 0
    y = negative_slope * x if x < 0

    For fp16/bf16 inputs, the computation is performed in fp32 internally and the result is cast back,
    to align with torch_npu's F.leaky_relu precision behavior.

    Parameters
    ----------
    a : Tensor
        The input tensor.
    negative_slope : float, optional
        Controls the angle of the negative slope (default is 0.0 impending small positive value, typically 0.01).
        Internally always treated as fp32. Must be non-negative, and must not be NaN or inf.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise LReLU result.

    Raises
    ------
    ValueError
        If negative_slope is negative, NaN, or inf.

    Examples
    --------
    x = pypto.tensor([-1.0, 2.0, 0.0, -0.5], pypto.DT_FP32)
    y = pypto.lrelu(x)

    Input x:  [-1.0, 2.0, 0.0, -0.5]
    Output y: [-0.01, 2.0, 0.0, -0.005]

    # With custom slope
    y2 = pypto.lrelu(x, negative_slope=0.1)
    Output y2: [-0.1, 2.0, 0.0, -0.05]
    """
    if isinstance(negative_slope, pypto_impl.Element):
        if negative_slope._is_float():
            slope_val = negative_slope._get_float_data()
        else:
            slope_val = negative_slope._get_signed_data()
    else:
        slope_val = negative_slope

    if isinstance(slope_val, float):
        if np.isnan(slope_val):
            raise PyptoError(0xF00002, ValueError("lrelu(): negative_slope must not be NaN."))
        if np.isinf(slope_val):
            raise PyptoError(0xF00002, ValueError("lrelu(): negative_slope must not be inf."))
    if isinstance(slope_val, (int, float)) and slope_val < 0:
        raise PyptoError(0xF00002, ValueError(
            f"lrelu(): negative_slope must be non-negative, but got {slope_val}."))

    negative_slope_base = pypto_impl.Element(pypto_impl.DT_FP32, slope_val)

    orig_dtype = other.dtype
    if orig_dtype in (DataType.DT_FP16, DataType.DT_BF16):
        fp32_input = pypto_impl.Cast(other, DataType.DT_FP32, CastMode.CAST_NONE,
                                     pypto_impl.SaturationMode.OFF)
        fp32_result = pypto_impl.LReLU(fp32_input, negative_slope_base)
        return pypto_impl.Cast(fp32_result, orig_dtype, CastMode.CAST_NONE,
                               pypto_impl.SaturationMode.OFF)

    return pypto_impl.LReLU(other, negative_slope_base)


@op_wrapper
def remainder(
    input: Union[Tensor, int, float],
    other: Union[Tensor, int, float],
    precision_type: PrecisionType = PrecisionType.HIGH_PRECISION,
) -> Tensor:
    """Computes the element-wise remainder of `input` divided by `other`.

    This function calculates the formula: `out = input - floor(input.div(other)) * other`.
    It supports broadcasting between the input tensors.

    Parameters
    ----------
    input : Tensor or Number
        The first input tensor or a scalar.
    other : Tensor or Number
        The second input tensor or a scalar to remainder operation.
    precision_type : PrecisionType, optional
        The precision type for remainder. Default is PrecisionType.HIGH_PRECISION.
        HIGH_PRECISION uses higher precision calculation to reduce precision loss.
        Use PrecisionType.INTRINSIC to directly use chip instructions.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise remainder.

    Raises
    ------
    RuntimeError
        If the two tensors are not broadcastable to a common shape.
        If `other` has a scalar value of zero, a RuntimeError is raised.

    See Also
    --------
    fmod : The modulus operation.

    Examples
    --------
    a = pypto.tensor([1, 3], pypto.DT_FP32)
    b = pypto.tensor([1, 3], pypto.DT_FP32)
    out = pypto.remainder(a, b)

    Input a:    [[7.0 8.0 9.0]]
    Input b:    [[-3.0 -3.0 -3.0]]
    Output out: [[-2.0 -1.0 0.0]]

    # Using high precision mode for FP16
    a = pypto.tensor([1, 3], pypto.DT_FP16)
    b = pypto.tensor([1, 3], pypto.DT_FP16)
    out = pypto.remainder(a, b, pypto.PrecisionType.HIGH_PRECISION)

    Input a:    [[7.0 8.0 9.0]]
    Input b:    [[-3.0 -3.0 -3.0]]
    Output out: [[-2.0 -1.0 0.0]]
    """
    if isinstance(input, pypto_impl.Tensor):
        if isinstance(other, pypto_impl.Tensor):
            return pypto_impl.Remainder(input, other, precision_type)
        if isinstance(other, float) or isinstance(other, int):
            _check_scalar_type("remainder", input.dtype, other)
            other = _clip_scalar_to_dtype(input.dtype, other)
            return pypto_impl.Remainder(input, pypto_impl.Element(input.dtype, other), precision_type)
    if isinstance(other, pypto_impl.Tensor):
        if isinstance(input, float) or isinstance(input, int):
            _check_scalar_type("remainder", other.dtype, input)
            input = _clip_scalar_to_dtype(other.dtype, input)
            return pypto_impl.Remainder(pypto_impl.Element(other.dtype, input), other, precision_type)
    raise PyptoError(0xF00001, TypeError(f"Unsupported operand types for remainder: {type(input)} and {type(other)}"))


@op_wrapper
def bitwise_and(self: Tensor, other: Union[Tensor, int]) -> Tensor:
    """Computes the element-wise bitwise AND of `self` and `other`.

    This function calculates the formula: `out = self & other`.
    It supports broadcasting between the input tensors.

    Parameters
    ----------
    self : Tensor
        The first input tensor.
    other : Tensor or int
        The second input tensor or an integer scalar.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise bitwise AND result.

    Raises
    ------
    RuntimeError
        If the two tensors are not broadcastable to a common shape.

    Examples
    --------
    a = pypto.tensor([0x1234, 0x5678], pypto.DT_INT16)
    b = pypto.tensor([0x0F0F, 0xF0F0], pypto.DT_INT16)
    out = pypto.bitwise_and(a, b)

    Input a:    [5, 3]
    Input b:    [3, 1]
    Output out: [1, 1]
    """
    if isinstance(other, pypto_impl.Tensor):
        return pypto_impl.BitwiseAnd(self, other)
    else:
        if not isinstance(other, int):
            raise PyptoError(
                0xF00001, TypeError(f"Scalar operand for bitwise_and must be an integer, but got {type(other)}.")
            )
        other = _clip_scalar_to_dtype(self.dtype, other)
        return pypto_impl.BitwiseAnd(self, pypto_impl.Element(self.dtype, other))


@op_wrapper
def bitwise_or(input1: Tensor, input2: Union[Tensor, int]) -> Tensor:
    """Computes the element-wise bitwise OR of `input1` and `input2`.

    This function calculates the formula: `out = input1 | input2`.
    It supports broadcasting between the input tensors.

    Parameters
    ----------
    input1 : Tensor
        The first input tensor.
    input2 : Tensor or int
        The second input tensor or an integer scalar.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise bitwise OR result.

    Raises
    ------
    RuntimeError
        If the two tensors are not broadcastable to a common shape.

    Examples
    --------
    a = pypto.tensor([5, 3], pypto.DT_INT16)
    b = pypto.tensor([3, 1], pypto.DT_INT16)
    out = pypto.bitwise_or(a, b)

    Input a:    [5, 3]
    Input b:    [3, 1]
    Output out: [7, 3]
    """
    if isinstance(input2, pypto_impl.Tensor):
        return pypto_impl.BitwiseOr(input1, input2)
    else:
        if not isinstance(input2, int):
            raise PyptoError(
                0xF00001, TypeError(f"Scalar operand for bitwise_or must be an integer, but got {type(input2)}.")
            )
        input2 = _clip_scalar_to_dtype(input1.dtype, input2)
        return pypto_impl.BitwiseOr(input1, pypto_impl.Element(input1.dtype, input2))


@op_wrapper
def bitwise_xor(first: Tensor, second: Union[Tensor, int]) -> Tensor:
    """Computes the element-wise bitwise XOR of `first` and `second`.

    This function calculates the formula: `out = first ^ second`.
    It supports broadcasting between the input tensors.

    Parameters
    ----------
    first : Tensor
        The first input tensor.
    second : Tensor or int
        The second input tensor or an integer scalar.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise bitwise XOR result.

    Raises
    ------
    RuntimeError
        If the two tensors are not broadcastable to a common shape.

    Examples
    --------
    a = pypto.tensor([5, 3], pypto.DT_INT16)
    b = pypto.tensor([3, 1], pypto.DT_INT16)
    out = pypto.bitwise_xor(a, b)

    Input a:    [5, 3]   # binary: [101, 011]
    Input b:    [3, 1]   # binary: [011, 001]
    Output out: [6, 2]   # binary: [110, 010]
    """
    if isinstance(second, pypto_impl.Tensor):
        return pypto_impl.BitwiseXor(first, second)
    else:
        if not isinstance(second, int):
            raise PyptoError(
                0xF00001, TypeError(f"Scalar operand for bitwise_xor must be an integer, but got {type(second)}.")
            )
        second = _clip_scalar_to_dtype(first.dtype, second)
        return pypto_impl.BitwiseXor(first, pypto_impl.Element(first.dtype, second))


@op_wrapper
def pow(
    base: Tensor, other: Union[Tensor, int, float], precision_type: PrecisionType = PrecisionType.HIGH_PRECISION
) -> Tensor:
    """Computes the element-wise power of `base` raised to `other`.

    This function calculates the formula: `out = base ** other`.

    Parameters
    ----------
    base : Tensor
        The base input tensor.
    other : Tensor or Number
        The exponent to which each element in `base` will be raised.
    precision_type : PrecisionType, optional
        The precision type for pow. Default is PrecisionType.HIGH_PRECISION.
        HIGH_PRECISION uses higher precision calculation to reduce precision loss.
        Use PrecisionType.INTRINSIC to directly use chip instructions.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise power operation results.

    Examples
    --------
    x = pypto.tensor([2, 2], pypto.DT_FP32)
    a = 2
    b = pypto.tensor([2, 2], pypto.DT_FP32)
    y = pypto.pow(x, a, PrecisionType.HIGH_PRECISION)
    z = pypto.pow(x, b, PrecisionType.HIGH_PRECISION)

    Input x:[[ 1.0 2.0],
             [-3.0 4.0]]
          b:[[2.0 2.0],
             [1.0 1.0]]
    Output y:[[1.0  4.0],
              [9.0 16.0]]
           z:[[ 1.0 4.0],
              [-3.0 4.0]]
    """
    if not isinstance(other, (pypto_impl.Tensor, int, float)):
        raise PyptoError(0xF00001, TypeError(f"other must be Tensor, int or float but got {type(other)}."))
    if isinstance(other, pypto_impl.Tensor):
        return pypto_impl.Pow(base, other, precision_type)
    if isinstance(other, int):
        return pypto_impl.Pow(base, pypto_impl.Element(DataType.DT_INT32, other), precision_type)
    return pypto_impl.Pow(base, pypto_impl.Element(DataType.DT_DOUBLE, other), precision_type)


@op_wrapper
def exp(input: Tensor, precision_type: PrecisionType = PrecisionType.INTRINSIC) -> Tensor:
    """Computes the element-wise exponential of `input`.

    This function calculates the formula: `out = e ** input`.

    Parameters
    ----------
    input : Tensor
        The input tensor.
    precision_type : PrecisionType, optional
        The precision type for exponential. Default is PrecisionType.INTRINSIC.
        INTRINSIC directly uses chip instructions for faster computation.
        Use PrecisionType.HIGH_PRECISION to use higher precision calculation to reduce precision loss.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise exponential.

    See Also
    -------
    sqrt : Element-wise square-root

    Examples
    --------
    x = pypto.tensor([3], pypto.DT_FP32)
    y = pypto.exp(x)

    Input x: [0.0    1.0    2.0]
    Output y:[1.0000 2.7183 7.3891]

    # Using high precision mode for FP16
    x = pypto.tensor([3], pypto.DT_FP16)
    y = pypto.exp(x, pypto.PrecisionType.HIGH_PRECISION)
    """
    return pypto_impl.Exp(input, precision_type)


@op_wrapper
def exp2(input: Tensor) -> Tensor:
    """Computes the element-wise exponential of `input`.

    This function calculates the formula: `out = e ** input`.

    Parameters
    ----------
    input : Tensor
        The input tensor.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise exponential.

    See Also
    -------
    sqrt : Element-wise square-root

    Examples
    --------
    x = pypto.tensor([3], pypto.DT_FP32)
    y = pypto.exp2(x)

    Input x: [0.0    1.0    2.0]
    Output y:[1.0000 2.0000 4.0000]
    """
    valid_shape = input.GetValidShape()
    if input.dtype in (pypto.DT_INT8, pypto.DT_UINT8):
        input = pypto.cast(input, pypto.DT_FP16)
        input = pypto.cast(input, pypto.DT_FP32)
    elif input.dtype in (pypto.DT_INT16, pypto.DT_INT32):
        input = pypto.cast(input, pypto.DT_FP32)

    two_element = pypto_impl.Element(input.dtype, 2)
    x = pypto_impl.Full(two_element, input.dtype, input.shape, valid_shape)
    return pow(x, input)


@op_wrapper
def expm1(input: Tensor) -> Tensor:
    """Computes the element-wise exponential of `input` minus 1.

    This function calculates the formula: `out = e ** input - 1`.

    Parameters
    ----------
    input : Tensor
        The input tensor.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise expm1 results.

    See Also
    -------
    exp : Element-wise exponential function

    Examples
    --------
    x = pypto.tensor([3], pypto.DT_FP32)
    y = pypto.expm1(x)

    Input x: [0.0     1.0    2.0]
    Output y:[0.0000 1.7183 6.3891]
    """

    return pypto_impl.Expm1(input)


@op_wrapper
def atan(self: Tensor) -> Tensor:
    """Computes the element-wise arctangent of `self`.

    This function calculates the trigonometric arctangent for each element in the input tensor.
    Mathematically, it computes: `out = arctan(self)`.

    Parameters
    ----------
    self : Tensor
        The input tensor.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise arctangent value.

    Examples
    --------
    x = pypto.tensor([3], pypto.DT_FP32)
    y = pypto.atan(x)

    Input  x: [0.0    1.0    -1.0   ]
    Output y: [0.0000 0.7854 -0.7854]
    """
    return pypto_impl.Atan(self)


@op_wrapper
def atan2(y: Tensor, x: Tensor) -> Tensor:
    """Computes the element-wise arctangent of `y / x` considering the quadrant.

    This function calculates the trigonometric arctangent for each pair of element in the input tensors.
    Mathematically, it computes: `out = arctan2(y, x)`
        where `x` corresponds to x-coordinate and `y` corresponds to y-coordinate.
    The result is in the range (-pi, pi].

    Parameters
    ----------
    y : Tensor
        The y-coordinate tensor.
    x : Tensor
        The x-coordinate tensor.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise arctangent value with correct quadrant.

    Examples
    --------
    y = pypto.tensor([4], pypto.DT_FP32)
    x = pypto.tensor([4], pypto.DT_FP32)
    z = pypto.atan2(y, x)

    Input  y: [1.0    -1.0     1.0    -1.0   ]
    Input  x: [1.0     1.0    -1.0    -1.0   ]
    Output z: [0.7854  2.3562 -2.3562 -0.7854]
    """
    return pypto_impl.Atan2(y, x)


@op_wrapper
def sin(self: Tensor) -> Tensor:
    """Computes the element-wise sine of `self`.

    This function calculates the trigonometric sine for each element in the input tensor.
    Mathematically, it computes: `out = sin(self)`.

    Parameters
    ----------
    self : Tensor
        The input tensor.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise exponential.

    See Also
    -------
    sqrt : Element-wise square-root

    Examples
    --------
    x = pypto.tensor([3], pypto.DT_FP32)
    y = pypto.sin(x)

    Input x: [0.0    1.5708 3.1416]
    Output y:[0.0000 1.0000 0.0000]
    """
    return pypto_impl.Sin(self)


@op_wrapper
def cos(self: Tensor) -> Tensor:
    """Computes the element-wise cosine of `self`.

    This function calculates the trigonometric cosine for each element in the input tensor.
    Mathematically, it computes: `out = cos(self)`.

    Parameters
    ----------
    self : Tensor
        The input tensor.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise exponential.

    See Also
    -------
    sqrt : Element-wise square-root

    Examples
    --------
    x = pypto.tensor([3], pypto.DT_FP32)
    y = pypto.cos(x)

    Input x: [0.0    1.5708 3.1416]
    Output y:[1.0000 0.0000 -1.0000]
    """
    return pypto_impl.Cos(self)


@op_wrapper
def erf(self: Tensor) -> Tensor:
    """Computes the element-wise error function of `self`.

    This function calculates the error function for each element in the input tensor.
    Mathematically, it computes: `out = erf(self)`.

    Parameters
    ----------
    self : Tensor
        The input tensor.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise error function values.

    See Also
    -------
    sqrt : Element-wise square-root

    Examples
    --------
    x = pypto.tensor([0.0, 1.0, 2.0], pypto.DT_FP32)
    y = pypto.erf(x)

    Input x: [0.0000 1.0000 2.0000]
    Output y:[0.0000 0.8427 0.9953]
    """
    return pypto_impl.Erf(self)


@op_wrapper
def sinh(input: Tensor) -> Tensor:
    """Computes the element-wise hyperbolic sine of `input`.

    This function calculates the formula: `out = (e ** input - e ** (-input)) / 2`.

    Parameters
    ----------
    input : Tensor
        The input tensor.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise hyperbolic sine results.

    See Also
    -------
    cosh : Element-wise hyperbolic cosine function

    Examples
    --------
    x = pypto.tensor([3], pypto.DT_FP32)
    y = pypto.sinh(x)

    Input x: [0.0     1.0     2.0]
    Output y:[0.0000  1.1752  3.6269]
    """

    return pypto_impl.Sinh(input)


@op_wrapper
def cosh(input: Tensor) -> Tensor:
    """Computes the element-wise hyperbolic cosine of `input`.

    This function calculates the formula: `out = (e ** input + e ** (-input)) / 2`.

    Parameters
    ----------
    input : Tensor
        The input tensor.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise hyperbolic cosine results.

    See Also
    -------
    sinh : Element-wise hyperbolic sine function

    Examples
    --------
    x = pypto.tensor([3], pypto.DT_FP32)
    y = pypto.cosh(x)

    Input x: [0.0     1.0     2.0]
    Output y:[1.0000  1.5431  3.7622]
    """

    return pypto_impl.Cosh(input)


@op_wrapper
def erfc(input: Tensor) -> Tensor:
    """Computes the element-wise complementary error function of `input`.

    This function calculates the complementary error function: `erfc(x) = 1 - erf(x)`.
    It is often used in probability theory, statistics, and partial differential equations.

    Parameters
    ----------
    input : Tensor
        The input tensor.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise complementary error function results.

    See Also
    -------
    exp : Element-wise exponential function
    log : Element-wise logarithm function

    Examples
    --------
    x = pypto.tensor([3], pypto.DT_FP32)
    y = pypto.erfc(x)

    Input x: [0.0     1.0     2.0]
    Output y:[1.0000  0.1573  0.0047]
    """

    return pypto_impl.Erfc(input)


@op_wrapper
def asin(self: Tensor) -> Tensor:
    """Computes the element-wise arcsine of `self`.

    This function calculates the inverse trigonometric sine for each element in the input tensor.
    Mathematically, it computes: `out = asin(self)`. The result is in the range [-pi/2, pi/2].
    Inputs with magnitude greater than 1 produce NaN.

    Parameters
    ----------
    self : Tensor
        The input tensor. Values are expected to be in [-1, 1].

    Returns
    -------
    Tensor
        A new tensor containing the element-wise arcsine.

    See Also
    -------
    acos : Element-wise arccosine function
    sin : Element-wise sine function

    Examples
    --------
    x = pypto.tensor([3], pypto.DT_FP32)
    y = pypto.asin(x)

    Input x: [-1.0    0.0    1.0]
    Output y:[-1.5708 0.0000 1.5708]
    """
    return pypto_impl.Asin(self)


@op_wrapper
def acos(self: Tensor) -> Tensor:
    """Computes the element-wise arccosine of `self`.

    This function calculates the inverse trigonometric cosine for each element in the input tensor.
    Mathematically, it computes: `out = acos(self)`. The result is in the range [0, pi].
    Inputs with magnitude greater than 1 produce NaN.

    Parameters
    ----------
    self : Tensor
        The input tensor. Values are expected to be in [-1, 1].

    Returns
    -------
    Tensor
        A new tensor containing the element-wise arccosine.

    See Also
    -------
    asin : Element-wise arcsine function
    cos : Element-wise cosine function

    Examples
    --------
    x = pypto.tensor([3], pypto.DT_FP32)
    y = pypto.acos(x)

    Input x: [-1.0    0.0    1.0]
    Output y:[3.1416  1.5708 0.0000]
    """
    return pypto_impl.Acos(self)


@op_wrapper
def asinh(input: Tensor) -> Tensor:
    """Computes the element-wise inverse hyperbolic sine of `input`.

    This function calculates the formula: `out = ln(input + sqrt(input ** 2 + 1))`.

    Parameters
    ----------
    input : Tensor
        The input tensor.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise inverse hyperbolic sine results.

    See Also
    -------
    acosh : Element-wise inverse hyperbolic cosine function

    Examples
    --------
    x = pypto.tensor([4], pypto.DT_FP32)
    y = pypto.asinh(x)

    Input x: [0.0      1.0      2.0      -1.0]
    Output y:[0.0000   0.8814   1.4436   -0.8814]
    """

    return pypto_impl.ASinh(input)


@op_wrapper
def acosh(input: Tensor) -> Tensor:
    """Computes the element-wise inverse hyperbolic cosine of `input`.

    This function calculates the formula: `out = ln(input + sqrt(input ** 2 - 1))`.
    The input must satisfy `input >= 1`.

    Parameters
    ----------
    input : Tensor
        The input tensor.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise inverse hyperbolic cosine results.

    See Also
    -------
    asinh : Element-wise inverse hyperbolic sine function

    Examples
    --------
    x = pypto.tensor([4], pypto.DT_FP32)
    y = pypto.acosh(x)

    Input x: [1.0      2.0      3.0      4.0]
    Output y:[0.0000   1.3170   1.7627   2.0634]
    """

    return pypto_impl.ACosh(input)


@op_wrapper
def atanh(input: Tensor) -> Tensor:
    """Computes the element-wise inverse hyperbolic tangent of `input`.

    This function calculates the formula: `out = 0.5 * ln((1 + input) / (1 - input))`.

    Parameters
    ----------
    input : Tensor
        The input tensor. Must be in the range (-1, 1) for real-valued output.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise inverse hyperbolic tangent results.

    See Also
    --------
    tanh : Element-wise hyperbolic tangent function

    Examples
    --------
    x = pypto.tensor([3], pypto.DT_FP32)
    y = pypto.atanh(x)

    Input x: [0.0     0.5     -0.5]
    Output y:[0.0     0.5493  -0.5493]
    """

    return pypto_impl.Atanh(input)


@op_wrapper
def sign(a: Tensor) -> Tensor:
    """Computes the element-wise exponential of `input`.

    This function return a tensor with the signs of the elements of input.

    Parameters
    ----------
    input : Tensor
        The input tensor.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise exponential.

    Examples
    --------
    x = pypto.tensor([3], pypto.DT_FP32)
    y = pypto.sign(x)

    Input x: [-5.0    0.0    5.0    10.0]
    Output y:[-1.    0.    1.    1.]
    """
    return pypto_impl.Sign(a)


@op_wrapper
def signbit(a: Tensor) -> Tensor:
    """Checks if the sign bit of each element of input is set (i.e., is negative).

    This function returns a tensor with the sign bits of the elements of input.

    Parameters
    ----------
    input : Tensor
        The input tensor.

    Returns
    -------
    Tensor
        A new bool tensor containing True where the sign bit is set, False otherwise.

    Examples
    --------
    x = pypto.tensor([-5.0, 0.0, 5.0, -2.0], pypto.DT_FP32)
    y = pypto.signbit(x)

    Input x: [-5.0, 0.0, 5.0, -2.0]
    Output y:[True, False, False, True]
    """
    return pypto_impl.Signbit(a)


@op_wrapper
def tanh(a: Tensor) -> Tensor:
    """
    Computes the hyperbolic tangent of each element in input.

    Parameters
    ----------
    a : Tensor
        The input tensor.

    Returns
    -------
    Tensor
        A new tensor containing the hyperbolic tangent of each element in input.

    Examples
    --------
    x = pypto.tensor([2], pypto.DT_FP32)
    y = pypto.tanh(x)

    Input x: [0.0, 1.0, -1.0]
    Output y: [0.0, 0.7615941559557649, -0.7615941559557649]
    """
    return pypto_impl.Tanh(a)


@op_wrapper
def abs(a: Tensor) -> Tensor:
    """
    Computes the absolute value of each element in input.

    Parameters
    ----------
    input : Tensor
        The input tensor.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise absolute.

    Examples
    --------
    x = pypto.tensor([3], pypto.DT_INT32)
    y = pypto.abs(x)

    Input x:  [-1, -2, 3]
    Output y: [ 1,  2, 3]
    """
    return pypto_impl.Abs(a)


@op_wrapper
def reciprocal(a: Tensor, precision_type: PrecisionType = PrecisionType.HIGH_PRECISION) -> Tensor:
    """
    Returns a new tensor with the reciprocal of the elements of input

    Parameters
    ----------
    input : Tensor
        The input tensor.
    precision_type : PrecisionType, optional
        The precision type for reciprocal. Default is PrecisionType.HIGH_PRECISION.
        HIGH_PRECISION uses higher precision calculation to reduce precision loss.
        Use PrecisionType.INTRINSIC to directly use chip instructions for faster computation.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise reciprocal.

    Examples
    --------
    x = pypto.tensor([4], pypto.DT_FP32)
    y = pypto.reciprocal(x)

    Input x:  [-0.4595, -2.1219, -1.4314,  0.7298]
    Output y: [-2.1763, -0.4713, -0.6986,  1.3702]

    # Using high precision mode
    x = pypto.tensor([4], pypto.DT_FP16)
    y = pypto.reciprocal(x, pypto.PrecisionType.HIGH_PRECISION)
    """
    return pypto_impl.Reciprocal(a, precision_type)


@op_wrapper
def relu(a: Tensor) -> Tensor:
    """
    Returns a new tensor with the rectified linear unit function applied element-wise.

    The function is defined as:
    y = max(0, x)

    Parameters
    ----------
    input : Tensor
        The input tensor.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise relu result.

    Examples
    --------
    x = pypto.tensor([-1.0, 2.0, 0.0, -0.5], pypto.DT_FP32)
    y = pypto.relu(x)

    Input x:  [-1.0, 2.0, 0.0, -0.5]
    Output y: [ 0.0, 2.0, 0.0,  0.0]
    """
    return pypto_impl.Relu(a)


@op_wrapper
def logical_not(input: Tensor) -> Tensor:
    """
    Computes the element-wise logical NOT of 'input'

    This function calculates the formula: 'out = input == 0? True : False'.

    Parameters
    ----------
    input : Tensor
        The input tensor

    Returns
    -------
    Tensor
        A tensor of bool with the same shape as input

    Examples
    --------
    a = pypto.tensor([5], pypto.DT_INT32)
    out = pypto.logical_not(a)

    Input a:    [0 1 2 3 4]
    Output out: [True False False False False]

    """
    return pypto_impl.LogicalNot(input)


@op_wrapper
def logical_and(input: Tensor, other: Tensor) -> Tensor:
    """Computes the element-wise logical AND of `input` and `other`.

    This function calculates the formula: `out = input && other`.

    Parameters
    ----------
    input : Tensor
        The first input tensor.
    other : Tensor
        The second input tensor. Should be broadcastable to the shape of `input`.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise logical AND operation results.

    Examples
    --------
    x = pypto.tensor([True, False], pypto.DT_BOOL)
    y = pypto.tensor([True, True], pypto.DT_BOOL)
    z = pypto.logical_and(x, y)

    Input x: [True, False]
    Input y: [True, True]
    Output z: [True, False]

    # 支持广播
    x = pypto.tensor([[True, False], [False, True]], pypto.DT_BOOL)
    y = pypto.tensor([True, False], pypto.DT_BOOL)
    z = pypto.logical_and(x, y)

    Input x:  [[True, False], [False, True]]
    Input y:  [True, False]
    Output z: [[True, False], [False, False]]
    """
    return pypto_impl.LogicalAnd(input, other)


@op_wrapper
def round(input: Tensor, decimals: int = 0) -> Tensor:
    """Rounds elements of `input` to the nearest number of decimal places.

    Parameters
    ----------
    input : Tensor
        The input tensor.
    decimals : int
        Number of decimal places to round to (default: 0).
        If decimals is negative, it specifies the number of positions to the left of the decimal point.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise round.

    Examples
    --------
    x = pypto.tensor([2, 2], pypto.DT_FP32)
    y = pypto.round(x, decimals=1)

    Input x: [[1.21, 2.35], [3.65, 4.76]]
    Output y: [[1.2, 2.4], [3.6, 4.8]]
    """
    return pypto_impl.Round(input, decimals)


@op_wrapper
def rsqrt(input: Tensor, precision_type: PrecisionType = PrecisionType.INTRINSIC) -> Tensor:
    """Computes the element-wise reciprocal of the square-root of `input`

    This function calculates the formula: `out = 1 / sqrt(input)`.

    Parameters
    ----------
    input : Tensor
        The input tensor.
    precision_type : PrecisionType, optional
        The precision type for reciprocal square-root. Default is PrecisionType.INTRINSIC.
        INTRINSIC directly uses chip instructions for faster computation.
        Use PrecisionType.HIGH_PRECISION to use higher precision calculation to reduce precision loss.

    Returns
    -------
    Tensor
        A new tensor with the reciprocal of the square-root of each of the element of input.

    Raises
    ------
    TODO

    See Also
    --------
    sqrt : square-root of each of the element

    Examples
    --------
    x = pypto.tensor([2, 2], pypto.DT_FP32)
    y = pypto.rsqrt(x)

    Input x: [[1.0  4.0],
              [16.0 9.0]]
    Output y:[[1.0  0.5],
              [0.25 0.33333]]

    # Using high precision mode
    x = pypto.tensor([2, 2], pypto.DT_FP16)
    y = pypto.rsqrt(x, pypto.PrecisionType.HIGH_PRECISION)
    """
    return pypto_impl.Rsqrt(input, precision_type)


@op_wrapper
def ceil(input: Tensor) -> Tensor:
    """Computes the element-wise ceiling of `input` (upward rounding to the nearest integer)

    This function calculates the formula: `out = ceil(input)`.
    The ceiling of a number is the smallest integer greater than or equal to the number.

    Parameters
    ----------
    input : Tensor
        The input tensor containing numerical values to be ceiling-rounded.

    Returns
    -------
    Tensor
        A new tensor with the ceiling value of each element of the input tensor.

    Raises
    ------
    TODO

    See Also
    --------
    ceil : ceil rounding (upward to the nearest integer)

    Examples
    --------
    x = pypto.tensor([2.1, -2.1, 5.0, 3.9], pypto.DT_FP32)
    y = pypto.ceil(x)

    Input x: [[1.2  4.7],
              [-1.1  9.0]]
    Output y:[[2.0  5.0],
              [-1.0  9.0]]
    """
    return pypto_impl.Ceil(input)


@op_wrapper
def floor(input: Tensor) -> Tensor:
    """Computes the element-wise squareroot of `input`.

    This function calculates the formula: `out = √input`.

    Parameters
    ----------
    input : Tensor
        The input tensor.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise squareroot.

    See Also
    --------
    floor : floor rounding (downward to the nearest integer)

    Examples
    --------
    x = pypto.tensor([5], pypto.DT_FP32)
    y = pypto.floor(x)

    Input x:  [1.2 4.2 9.8 6.9 25.5]
    Output y: [1.0 4.0 9.0 6.0  25.0]
    """
    return pypto_impl.Floor(input)


@op_wrapper
def trunc(input: Tensor) -> Tensor:
    """Computes the element-wise squareroot of `input`.

    This function calculates the formula: `out = √input`.

    Parameters
    ----------
    input : Tensor
        The input tensor.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise squareroot.

    See Also
    --------
    trunc : trunc rounding (towards zero to the nearest integer)

    Examples
    --------
    x = pypto.tensor([5], pypto.DT_FP32)
    y = pypto.trunc(x)

    Input x:  [1.3 4.2 9.8 16.4 25.8]
    Output y: [1.0 4.0 10.0 16.0 26.0]
    """
    return pypto_impl.Trunc(input)


@op_wrapper
def sqrt(input: Tensor, precision_type: PrecisionType = PrecisionType.INTRINSIC) -> Tensor:
    """Computes the element-wise squareroot of `input`.

    This function calculates the formula: `out = √input`.

    Parameters
    ----------
    input : Tensor
        The input tensor.
    precision_type : PrecisionType, optional
        The precision type for square root. Default is PrecisionType.INTRINSIC.
        INTRINSIC directly uses chip instructions for faster computation.
        Use PrecisionType.HIGH_PRECISION to use higher precision calculation to reduce precision loss.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise squareroot.

    See Also
    --------
    exp : Element-wise exponential function.

    Examples
    --------
    x = pypto.tensor([5], pypto.DT_FP32)
    y = pypto.sqrt(x)

    Input x:  [1.0 4.0 9.0 16.0 25.0]
    Output y: [1.0 2.0 3.0 4.0  5.0]

    # Using high precision mode for FP16
    x = pypto.tensor([5], pypto.DT_FP16)
    y = pypto.sqrt(x, pypto.PrecisionType.HIGH_PRECISION)
    """
    return pypto_impl.Sqrt(input, precision_type)


@op_wrapper
def neg(a: Tensor) -> Tensor:
    """
    Returns a new tensor with the negative of the elements of input.

    Parameters
    ----------
    input : Tensor
        The input tensor.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise neg.

    Examples
    --------
    x = pypto.tensor([5], pypto.DT_FP32)
    y = pypto.neg(x)

    Input x: [ 0.0090, -0.2262, -0.0682, -0.2866,  0.3940]
    Output y:[-0.0090,  0.2262,  0.0682,  0.2866, -0.3940]
    """
    return pypto_impl.Neg(a)


@op_wrapper
def log(input: Tensor, precision_type: PrecisionType = PrecisionType.INTRINSIC) -> Tensor:
    """Computes the element-wise log of `input`.

    This function calculates the formula: `out = log(input)`.

    Parameters
    ----------
    input : Tensor
        The input tensor.
    precision_type : PrecisionType, optional
        The precision type for logarithm. Default is PrecisionType.INTRINSIC.
        INTRINSIC directly uses chip instructions for faster computation.
        Use PrecisionType.HIGH_PRECISION to use higher precision calculation to reduce precision loss.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise log.

    See Also
    -------
    sqrt : Element-wise square-root

    Examples
    --------
    x = pypto.tensor([3], pypto.DT_FP32)
    y = pypto.log(x)

    Input x: [1.0     2.0    3.0]
    Output y:[0.0000 0.6931 1.0986]

    # Using high precision mode for FP16
    x = pypto.tensor([3], pypto.DT_FP16)
    y = pypto.log(x, pypto.PrecisionType.HIGH_PRECISION)
    """

    return pypto_impl.Log(input, pypto_impl.LogBaseType.LOG_E, precision_type)


@op_wrapper
def log2(input: Tensor, precision_type: PrecisionType = PrecisionType.INTRINSIC) -> Tensor:
    """Computes the element-wise base-2 logarithm of `input`.

    This function calculates the formula: `out = log_2(input)`.

    Parameters
    ----------
    input : Tensor
        The input tensor. Must be positive (input > 0).
    precision_type : PrecisionType, optional
        The precision type for logarithm. Default is PrecisionType.INTRINSIC.
        INTRINSIC directly uses chip instructions for faster computation.
        Use PrecisionType.HIGH_PRECISION to use higher precision calculation to reduce precision loss.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise base-2 logarithm.

    See Also
    --------
    sqrt : Element-wise square-root.

    Examples
    --------
    >>> x = pypto.tensor([1.0, 2.0, 4.0], pypto.DT_FP32)
    >>> y = pypto.log2(x)
    # Input x: [1.0     2.0     4.0]
    # Output y: [0.0000 1.0000 2.0000]
    """
    return pypto_impl.Log(input, pypto_impl.LogBaseType.LOG_2, precision_type)


@op_wrapper
def log10(input: Tensor, precision_type: PrecisionType = PrecisionType.INTRINSIC) -> Tensor:
    """Computes the element-wise base-10 logarithm of `input`.

    This function calculates the formula: `out = log_10(input)`.

    Parameters
    ----------
    input : Tensor
        The input tensor. Must be positive (input > 0).
    precision_type : PrecisionType, optional
        The precision type for logarithm. Default is PrecisionType.INTRINSIC.
        INTRINSIC directly uses chip instructions for faster computation.
        Use PrecisionType.HIGH_PRECISION to use higher precision calculation to reduce precision loss.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise base-10 logarithm.

    See Also
    --------
    sqrt : Element-wise square-root.

    Examples
    --------
    >>> x = pypto.tensor([1.0, 10.0, 100.0], pypto.DT_FP32)
    >>> y = pypto.log10(x)
    # Input x: [1.0      10.0     100.0]
    # Output y: [0.0000   1.0000   2.0000]
    """
    return pypto_impl.Log(input, pypto_impl.LogBaseType.LOG_10, precision_type)


@op_wrapper
def log1p(input: Tensor) -> Tensor:
    """Computes the element-wise natural logarithm of (1 + input).

    This function calculates the formula: `out = log(1 + input)`, where `log`
    denotes the natural logarithm (base e).

    Parameters
    ----------
    input : Tensor
        The input tensor. Must satisfy `input > -1`.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise natural logarithm of (1 + input).

    See Also
    --------
    log : Element-wise natural logarithm.
    add : Element-wise addition.

    Examples
    --------
    >>> x = pypto.tensor([0.0, 1.0, 2.0], pypto.DT_FP32)
    >>> y = pypto.log1p(x)
    # Input x: [0.0     1.0     2.0]
    # Output y: [0.0000 0.6931 1.0986]
    """
    return pypto_impl.Log1p(input)


@op_wrapper
def tan(x: Tensor) -> Tensor:
    """Computes the element-wise tangent of `x`.

    This function calculates the formula: `out = tan(x)`.

    Parameters
    ----------
    x : Tensor
        The input tensor. Supports float16, bfloat16, float32.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise tangent.

    See Also
    --------
    sin : Element-wise sine.
    cos : Element-wise cosine.

    Examples
    --------
    >>> x = pypto.tensor([0.0, 0.785, 1.57], pypto.DT_FP32)
    >>> y = pypto.tan(x)
    # Input x: [0.0     0.785   1.57]
    # Output y: [0.0000  1.0     large]
    """
    return pypto_impl.Tan(x)


@op_wrapper
def clip(
    input: Tensor,
    min: Optional[Union[Tensor, Element, float, int]] = None,
    max: Optional[Union[Tensor, Element, float, int]] = None,
):
    """
    Make the values in `input` greater than `min` and less than `max`.

    Parameters
    ----------
    input : Tensor
        The first input tensor.
    min : Tensor or Element
        The minimum value.
    max: Tensor or Element
        The maximum value

    Returns
    -------
    Tensor
        A new tensor containing the values greater than `min` and less than `max`.

    Examples
    --------
    x = pypto.tensor([2, 3], pypto.DT_INT32)
    min = 1
    max = 3
    out = pypto.clip(x, min, max)

    Input x:    [[0 2 4], [3, 4, 6]]
    Output out: [[1 2 3], [3, 3, 3]]
    """
    if min is None and max is None:
        return input

    element_types = (pypto_impl.Element, int, float)
    is_element_mode = isinstance(min, element_types) or isinstance(max, element_types)
    default = pypto_impl.Tensor() if not is_element_mode else pypto_impl.Element(pypto_impl.DataType.DT_BOTTOM, 0)
    if min is None:
        min = default
    if max is None:
        max = default

    if not isinstance(min, pypto_impl.Element) and isinstance(min, element_types):
        min = pypto_impl.Element(input.GetDataType(), min)

    if not isinstance(max, pypto_impl.Element) and isinstance(max, element_types):
        max = pypto_impl.Element(input.GetDataType(), max)

    return pypto_impl.Clip(input, min, max)


@op_wrapper
def cumsum(input: Tensor, dim: int) -> Tensor:
    """
    This function returns the cumulative sum over a given axis.
    Parameters
    ---------
    input: Tensor
        tensor to be calculated.
    dim : int
        specified dimension.
    out: Tensor
        The tensor after calculating the cumulative sum.
    Examples
    ---------
    x = pypto.tensor([2, 3], pypto.data_type.DT_INT32)
    dim = 0
    out = pypto.cumsum(x, dim)
    Input  x : [[0 1 2],
                [3 4 5]]
    Output out:[[0 1 2],
                [3 5 7]]
    """
    return pypto_impl.cumsum(input, dim)


@op_wrapper
def cumprod(input: Tensor, dim: int) -> Tensor:
    """
    This function returns the cumulative prod over a given axis.
    Parameters
    ---------
    input: Tensor
        tensor to be calculated.
    dim : int
        specified dimension.
    out: Tensor
        The tensor after calculating the cumulative prod.
    Examples
    ---------
    x = pypto.tensor([2, 3], pypto.data_type.DT_FP32)
    dim = 0
    out = pypto.cumprod(x, dim)
    Input  x : [[0 1 2],
                [3 4 5]]
    Output out:[[0 1 2],
                [0 4 10]]
    """
    return pypto_impl.cumprod(input, dim)


@op_wrapper
def bitwise_right_shift(input: Union[Tensor, int], other: Union[Tensor, int]) -> Tensor:
    """Computes the element-wise bitwise right shift of `input` and `other`.

    This function calculates the formula: `out = input >> other`.
    It supports broadcasting between the input tensors.

    Parameters
    ----------
    input : Tensor
        The first input tensor.
    other : Tensor or Number
        The second input tensor or a scalar to bitwise right shift.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise right shift.

    Raises
    ------
    RuntimeError
        If the two tensors are not broadcastable to a common shape.

    See Also
    --------
    sub : The inverse operation, element-wise subtraction.
    mul : Element-wise multiplication.

    Examples
    --------
    a = pypto.tensor([1, 3], pypto.DT_INT16)
    b = pypto.tensor([1, 3], pypto.DT_INT16)
    out = pypto.bitwise_right_shift(a, b)

    Input a:    [[1 2 3]]
    Input b:    [[1 1 1]]
    Output out: [[0 1 1]]
    """
    if isinstance(input, pypto_impl.Tensor) and isinstance(other, pypto_impl.Tensor):
        return pypto_impl.BitwiseRightShift(input, other)
    elif isinstance(input, pypto_impl.Tensor):
        return pypto_impl.BitwiseRightShift(input, pypto_impl.Element(input.dtype, other))
    else:
        return pypto_impl.BitwiseRightShift(pypto_impl.Element(other.dtype, input), other)


@op_wrapper
def bitwise_left_shift(input: Union[Tensor, int], other: Union[Tensor, int]) -> Tensor:
    """Computes the element-wise bitwise left shift of `input` and `other`.

    This function calculates the formula: `out = input << other`.
    It supports broadcasting between the input tensors.

    Parameters
    ----------
    input : Tensor
        The first input tensor.
    other : Tensor or Number
        The second input tensor or a scalar to bitwise left shift.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise left shift.

    Raises
    ------
    RuntimeError
        If the two tensors are not broadcastable to a common shape.

    See Also
    --------
    sub : The inverse operation, element-wise subtraction.
    mul : Element-wise multiplication.

    Examples
    --------
    a = pypto.tensor([1, 3], pypto.DT_INT16)
    b = pypto.tensor([1, 3], pypto.DT_INT16)
    out = pypto.bitwise_left_shift(a, b)

    Input a:    [[1 2 3]]
    Input b:    [[1 1 1]]
    Output out: [[2 4 6]]
    """
    if isinstance(input, pypto_impl.Tensor) and isinstance(other, pypto_impl.Tensor):
        return pypto_impl.BitwiseLeftShift(input, other)
    elif isinstance(input, pypto_impl.Tensor):
        return pypto_impl.BitwiseLeftShift(input, pypto_impl.Element(input.dtype, other))
    else:
        return pypto_impl.BitwiseLeftShift(pypto_impl.Element(other.dtype, input), other)


@op_wrapper
def bitwise_not(self: Tensor) -> Tensor:
    """
    Computes the element-wise bitwise NOT of 'self'

    This function calculates the formula: 'out = ~self'.
    For each element in the self tensor, performs a bitwise NOT operation.

    Parameters
    ----------
    self : Tensor
        The input tensor (should be of integer type)

    Returns
    -------
    Tensor
        A tensor with the same shape and dtype as input

    Examples
    --------
    a = pypto.tensor([0, 1, 2, 3, 4], pypto.DT_INT32)
    out = pypto.bitwise_not(a)

    Self a:    [0 1 2 3 4]  (in binary: [000, 001, 010, 011, 100])
    Output out: [-1 -2 -3 -4 -5]  (in binary: [111, 110, 101, 100, 011])

    """
    return pypto_impl.BitwiseNot(self)


@op_wrapper
def triu(input: Tensor, diagonal: SymInt = 0) -> Tensor:
    """
    Return the upper traingular part of a matrix or a banch of matrices `input`, the other elements of
    the result are set to 0.
    Parameters
    ---------
    input: Tensor
        The tensor to be calculated.
    diagonal : SymInt
        The diagonal to consider.
    out: Tensor
        The tensor after calculation.
    Examples
    ---------
    x = pypto.tensor([3, 3], pypto.data_type.DT_INT32)
    diagonal = 0
    out = pypto.triu(x, diagonal)
    Input  x : [[1 2 3],
                [4 5 6],
                [7 8 9]]
    Output out:[[1 2 3],
                [0 5 6],
                [0 0 9]]
    """
    return pypto_impl.TriU(input, diagonal)


@op_wrapper
def tril(input: Tensor, diagonal: SymInt = 0) -> Tensor:
    """
    Return the lower traingular part of a matrix or a banch of matrices `input`, the other elements of
    the result are set to 0.
    Parameters
    ---------
    input: Tensor
        The tensor to be calculated.
    diagonal : SymInt
        The diagonal to consider.
    out: Tensor
        The tensor after calculation.
    Examples
    ---------
    x = pypto.tensor([3, 3], pypto.data_type.DT_INT32)
    diagonal = 0
    out = pypto.tril(x, diagonal)
    Input  x : [[1 2 3],
                [4 5 6],
                [7 8 9]]
    Output out:[[1 0 0],
                [4 5 0],
                [7 8 9]]
    """
    return pypto_impl.TriL(input, diagonal)


@op_wrapper
def copysign(input: Tensor, other: Tensor) -> Tensor:
    """
    Create a new floating-point tensor with the magnitude of input and the sign of other, elementwise.
    Parameters
    ---------
    input: Tensor
        The tensor of magnitudes.
    other : Tensor
         The tensor that contains value(s) whose signbit(s) are applied to the magnitudes in input.
    out: Tensor
        The output tensor.
    Examples
    ---------
    x = pypto.tensor([3, 3], pypto.data_type.DT_FP32)
    y = pypto.tensor([3, 3], pypto.data_type.DT_FP32)
    out = pypto.copysign(x, y)
    Input  x : [[1 -2  3],
                [4  5 -6],
                [-7 8  9]]
    Input  y : [[-1 6 -8],
                [1 -1  0],
                [7 -8  9]]
    Output out:[[-1 2 -3],
                [4 -5  6],
                [7 -8  9]]
    """
    return pypto_impl.CopySign(input, other)


@op_wrapper
def isfinite(self: Tensor) -> Tensor:
    """
    Judge whether the value in Tensor `self` is inf/nan/-inf, if it is, the
        return value will be false, otherwise it will be true.

    Parameters
    --------
    self: Tensor
        The input tensor

    Examples
    --------
    self = pypto.tensor([3, 3], pypto.data_type.DT_FP32)
    out = pypto.isfinite(self)
    Input  self: [[1 nan 3],
                  [inf 1 1],
                  [1, 1, -inf]]
    Output out:  [[True False True],
                  [False True True],
                  [True True False]]
    """
    return pypto_impl.isfinite(self)


@op_wrapper
def isnan(self: Tensor) -> Tensor:
    """
    Judge whether the value in Tensor `self` is nan, if it is, the return
        value will be true, otherwise it will be false.

    The supported input dtypes are fp16/bf16/fp32 and the output dtype is bool.
    Both +inf/-inf and +0.0/-0.0 are treated as non-nan and return false.

    Parameters
    --------
    self: Tensor
        The input tensor

    Examples
    --------
    self = pypto.tensor([3, 3], pypto.data_type.DT_FP32)
    out = pypto.isnan(self)
    Input  self: [[1 nan 3],
                  [inf 1 1],
                  [1, 1, -inf]]
    Output out:  [[False True False],
                  [False False False],
                  [False False False]]
    """
    return pypto_impl.isnan(self)


@op_wrapper
def cbrt(self: Tensor) -> Tensor:
    """
    Computes the element-wise cube root of 'self'

    This function calculates the formula: 'out = self^(1/3)'.
    For each element in the self tensor, performs a cube root operation.

    Parameters
    ----------
    self : Tensor
        The input tensor

    Returns
    -------
    Tensor
        A tensor with the same shape and dtype as input

    Examples
    --------
    x = pypto.tensor([1, 2], pypto.DT_FP32)
    out = pypto.cbrt(x)

    Input  x:[[8, -8]]
    Output y:[[2, -2]]
    """
    return copysign(pow(abs(self), 1.0 / 3.0), self)


@op_wrapper
def gcd(input: Tensor, other: Union[Tensor, int]) -> Tensor:
    """
    This function returns the greatest common divisor of the corresponding elements of input and other.
    Parameters
    ---------
    input: Tensor
        tensor to be calculated.
    other: Tensor or int
        tensor to be calculated.
    out: Tensor
        The tensor after calculating the greatest common divisor of the corresponding elements of input and other.
    Examples
    ---------
    x = pypto.tensor([2, 3], pypto.data_type.DT_INT32)
    y = pypto.tensor([2, 3], pypto.data_type.DT_INT32)
    out = pypto.gcd(x, y)
    Input  x : [[1 1 2],
                [3 4 5]]
           y : [[6 6 6],
                [6 6 6]]
    Output out:[[1 1 2],
                [3 2 1]]
    """
    if isinstance(other, pypto_impl.Tensor):
        return pypto_impl.Gcd(input, other)
    else:
        return pypto_impl.Gcd(input, pypto_impl.Element(input.dtype, other))


@overload
def var(input: Tensor, dim: List[int], correction: float) -> Tensor:
    """
    Computes the variance  of 'input' over the dimensions.

    This function calculates the formula: 'out = 1 / max(0, N - correction) * sum((x_i - (sum(x_i) / N))^2)'.

    Parameters
    ----------
    input : Tensor
        The input Tensor to be calculated.
    dim : List
        Dimensions involved in calculating variance.
    correction : float
        The difference between sample size and sample degree if freedom. Usually take 0 or 1.

    Returns
    -------
    Tensor
        A tensor with the input shape of the dimensions after reduce

    Examples
    --------
    x = pypto.tensor([[2, 3], pypto.DT_FP32)
    y = pypto.var(x, [1], 0)

    Input  x:[[1., 2., 3.],
              [4., 5., 6.]]
    Output y:[0.6667, 0.6667]
    """
    ...


@overload
def var(
    input: Tensor, dim: Union[int, List[int], Tuple[int]] = None, *, correction: float = 1, keepdim: bool = False
) -> Tensor:
    """
    Computes the variance  of 'input' over the dimensions.

    This function calculates the formula: 'out = 1 / max(0, N - correction) * sum((x_i - (sum(x_i) / N))^2)'.

    Parameters
    ----------
    input : Tensor
        The input Tensor to be calculated.
    dim : Union[int, List[int], Tuple[int]]
        Dimensions involved in calculating variance. Default is None, means all demensions.
    correction : float, optional
        The difference between sample size and sample degree if freedom. Default take 1.
    keepdim : bool, optional
        whether the output tensor has dim retained or not. Default: False.

    Returns
    -------
    Tensor
        A tensor with the input shape of the dimensions after reduce

    Examples
    --------
    x = pypto.tensor([[2, 3], pypto.DT_FP32)
    y = pypto.var(x, 1, correction=1, keepdim=True)

    Input  x:[[1., 2., 3.],
              [4., 5., 6.]]
    Output y:[[1.],
              [1.]]
    """
    ...


@op_wrapper
def var(
    input: Tensor, dim: Union[int, List[int], Tuple[int]] = None, correction: float = 1, keepdim: bool = False
) -> Tensor:
    """
    Computes the variance  of 'input' over the dimensions.

    This function calculates the formula: 'out = 1 / max(0, N - correction) * sum((x_i - (sum(x_i) / N))^2)'.

    Parameters
    ----------
    input : Tensor
        The input Tensor to be calculated.
    dim : Union[int, List[int], Tuple[int]]
        Dimensions involved in calculating variance. Default is None, means all demensions.
    correction : float, optional
        The difference between sample size and sample degree if freedom. Default take 1.
    keepdim : bool, optional
        whether the output tensor has dim retained or not. Default: False.

    Returns
    -------
    Tensor
        A tensor with the input shape of the dimensions after reduce

    Examples
    --------
    x = pypto.tensor([[2, 3], pypto.DT_FP32)
    y = pypto.var(x, 1, correction=1, keepdim=True)

    Input  x:[[1., 2., 3.],
              [4., 5., 6.]]
    Output y:[[1.],
              [1.]]
    """
    inner_dim = None
    if isinstance(dim, int):
        inner_dim = [dim]
    elif dim is None or len(dim) == 0:
        inner_dim = []
    elif isinstance(dim, (list, tuple)):
        inner_dim = list(dim)
    else:
        raise PyptoError(
            0xF00001, TypeError("the type of dim is not supported. 'int' or 'Lise[int]' or 'Tuple[int]' is needed.")
        )

    return pypto_impl.Var(input, inner_dim, correction, keepdim)


@op_wrapper
def ceil_div(
    self: Tensor,
    other: Union[Tensor, int],
) -> Tensor:
    """
    Calculate the ceiling division of two tensors.
    Parameters
    ---------
    self: Tensor
        The dividend tensor.
    other: Tensor or int
        The divisor tensor or scalar.
    out: Tensor
        The tensor after calculating the ceiling division of the corresponding elements of self and other.
    Examples
    ---------
    x = pypto.tensor([2, 3], pypto.data_type.DT_INT32)
    y = pypto.tensor([2, 3], pypto.data_type.DT_INT32)
    out = pypto.ceildiv(x, y)
    Input  x : [[1 6 6],
                [4 6 6]]
           y : [[1 1 2],
                [3 4 5]]
    Output out:[[1 6 3],
                [2 2 2]]
    """
    if isinstance(other, pypto_impl.Tensor):
        return pypto_impl.CeilDiv(self, other)
    else:
        _check_scalar_type("ceil_div", self.dtype, other)
        other = _clip_scalar_to_dtype(self.dtype, other)
        return pypto_impl.CeilDiv(self, pypto_impl.Element(self.dtype, other))


@op_wrapper
def floor_div(
    input: Tensor,
    other: Union[Tensor, int, float],
) -> Tensor:
    """
    Calculate the floor division of two tensors.
    Parameters
    ---------
    input: Tensor
        The dividend tensor.
        Supported dtypes: DT_FP16, DT_BF16, DT_FP32, DT_INT32, DT_INT8, DT_UINT8.
    other: Tensor or int or float
        The divisor tensor or scalar.
    out: Tensor
        The tensor containing the floor division results of the corresponding elements in `input` and `other`.
    Examples
    ---------
    x = pypto.tensor([2, 3], pypto.DT_INT32)
    y = pypto.tensor([2, 3], pypto.DT_INT32)
    out = pypto.floor_div(x, y)
    Input  x : [[1 6 6],
                [4 6 6]]
           y : [[1 1 2],
                [3 4 5]]
    Output out:[[1 6 3],
                [1 1 1]]
    """
    if isinstance(other, pypto_impl.Tensor):
        return pypto_impl.FloorDiv(input, other)
    else:
        _check_scalar_type("floor_div", input.dtype, other)
        other = _clip_scalar_to_dtype(input.dtype, other)
        return pypto_impl.FloorDiv(input, pypto_impl.Element(input.dtype, other))


@op_wrapper
def prelu(self: Tensor, weight: Tensor) -> Tensor:
    """
    Applies the element-wise parametric rectified linear unit (PReLU) function.

    The function is defined as:
    f(x) = max(0, x) + weight * min(0, x)

    Parameters
    ----------
    input : Tensor
        The input tensor.
    weight : Tensor
        The learnable parameter tensor. For a 4D input tensor, weight should be a 1D tensor
        with size equal to the number of channels (second dimension).

    Returns
    -------
    Tensor
        A new tensor containing the element-wise PReLU activation results.

    Examples
    --------
    x = pypto.tensor([-1.0, 2.0, -0.5, 3.0], dtype="float32")
    weight = pypto.tensor([0.25], dtype="float32")
    y = pypto.prelu(x, weight)

    Input x:  [-1.0, 2.0, -0.5, 3.0]
    Weight:   [0.25]
    Output y: [-0.25, 2.0, -0.125, 3.0]
    """
    return pypto_impl.PReLU(self, weight)
