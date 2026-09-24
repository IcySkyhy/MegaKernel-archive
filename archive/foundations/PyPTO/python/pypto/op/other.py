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

from typing import Sequence, Union

from .. import pypto_impl
from .._element import Element
from .._op_wrapper import op_wrapper
from ..enum import DataType
from ..error import PyptoError
from ..tensor import Tensor

_FLOAT_DTYPES = {DataType.DT_FP32, DataType.DT_FP16, DataType.DT_BF16}
_UNSIGNED_INT_DTYPES = {DataType.DT_UINT8, DataType.DT_UINT16, DataType.DT_UINT32, DataType.DT_UINT64}


def _check_pad_value(x: Tensor, value: Union[float, int]) -> None:
    if not isinstance(value, float):
        return
    if x.dtype in _FLOAT_DTYPES:
        return
    raise PyptoError(
        0xF00002,
        ValueError(
            f"value type mismatch: tensor dtype is {repr(x.dtype)} but value is float. "
            f"Integer tensor requires int value."
        ),
    )


def _where_arg_dtype(val):
    if isinstance(val, pypto_impl.Tensor):
        return val.dtype
    if isinstance(val, pypto_impl.Element):
        return val._get_data_type()
    if isinstance(val, float):
        return DataType.DT_FP32
    return None


def _check_where_dtype_consistency(input_val, other_val):
    input_dtype = _where_arg_dtype(input_val)
    other_dtype = _where_arg_dtype(other_val)
    if input_dtype is None or other_dtype is None:
        return
    if input_dtype != other_dtype:
        raise PyptoError(
            0xF00002,
            TypeError(
                f"where() input and other data type inconsistent: "
                f"input dtype is {input_dtype}, other dtype is {other_dtype}. "
                f"Please ensure both inputs have the same dtype, or use Element to "
                f"specify matching dtype."
            ),
        )


def _check_where_scalar_unsigned(arg_name: str, scalar_val, tensor_arg, tensor_name: str):
    if isinstance(scalar_val, pypto_impl.Element):
        elem_dtype = scalar_val._get_data_type()
        if elem_dtype not in _UNSIGNED_INT_DTYPES:
            return
        elem_val = scalar_val._get_signed_data() if not scalar_val._is_float() else scalar_val._get_float_data()
        if elem_val < 0:
            raise PyptoError(
                0xF00002,
                ValueError(
                    f"where() does not support negative scalar for unsigned integer dtype: "
                    f"'{arg_name}' is {elem_val} but Element dtype is {elem_dtype}. "
                    f"Negative values cannot be represented in unsigned integer types."
                ),
            )
        return
    if not isinstance(scalar_val, int) or scalar_val >= 0:
        return
    if not isinstance(tensor_arg, pypto_impl.Tensor):
        return
    if tensor_arg.dtype not in _UNSIGNED_INT_DTYPES:
        return
    raise PyptoError(
        0xF00002,
        ValueError(
            f"where() does not support negative scalar for unsigned integer tensor: "
            f"'{arg_name}' is {scalar_val} but '{tensor_name}' tensor dtype is {tensor_arg.dtype}. "
            f"Negative values cannot be represented in unsigned integer types."
        ),
    )


@op_wrapper
def where(condition: Tensor, input: Union[Tensor, float, Element], other: Union[Tensor, float, Element]) -> Tensor:
    """
    Return a tensor of elements selected from either `input` or `other`, depending on `condition`.

    This function implements element-wise selection:
    'out[i] = input[i] if condition[i] else other[i]'.
    It supports broadcasting among `condition`, `input`, and `other`.

    Parameters
    ----------
    condition : Tensor of bool
        A boolean tensor indicating which elements to select from `input` (True) or `other` (False).
    input : Tensor or float or Element
        A tensor or scalar value to be selected where `condition` is True.
        When a float scalar is used, the paired tensor argument must be DT_FP32.
        For other dtypes, use Element instead, e.g. pypto.Element(pypto.DT_FP16, 1.0).
    other : Tensor or float or Element
        A tensor or scalar value to be selected where `condition` is False.
        When a float scalar is used, the paired tensor argument must be DT_FP32.
        For other dtypes, use Element instead, e.g. pypto.Element(pypto.DT_FP16, 0.0).

    Returns
    -------
    Tensor
        A tensor with the same shape as the broadcasted `condition`, containing elements
        from `input` where `condition` is True, and from `other` otherwise.
        The data type is determined by type promotion rules between `input` and `other`.

    Raises
    ------
    RuntimeError
        If `condition`, `input`, and `other` cannot be broadcasted to a common shape.
    TypeError
        If `condition` is not a boolean tensor.
    TypeError
        If a float scalar is used with a non-fp32 tensor. Use Element instead.

    See Also
    --------
    logical_not : Computes element-wise logical NOT.
    add : Element-wise addition with optional scaling.

    Examples
    --------
    cond = pypto.tensor([4], pypto.DT_BOOL)
    x = pypto.tensor([4], pypto.DT_FP32)
    y = pypto.tensor([4], pypto.DT_FP32)
    out1 = pypto.where(cond, x, y)

    Input cond:  [True False True False]
    Input x:     [1.0  2.0  3.0  4.0]
    Input y:     [10.0 20.0 30.0 40.0]
    Output out1: [1.0  20.0 3.0  40.0]

    # Using scalar inputs (float scalar is converted to DT_FP32 Element)
    out2 = pypto.where(cond, 1.0, 0.0)

    Output out2: [1.0 0.0 1.0 0.0]

    # Using Element for non-fp32 dtypes
    x_fp16 = pypto.tensor([4], pypto.DT_FP16)
    out3 = pypto.where(cond, x_fp16, pypto.Element(pypto.DT_FP16, 0.0))

    # Broadcasting example
    cond = pypto.tensor([2, 2], pypto.DT_BOOL)
    x = pypto.tensor([1, 2], pypto.DT_FP32)  # Will be broadcasted
    y = 0.0
    out4 = pypto.where(cond, x, y)

    Input cond:  [[True False], [False True]]
    Input x:     [1.0 2.0]
    Input y:     0.0

    Output out4: [[1.0 0.0],
                  [0.0 2.0]])
    """
    _check_where_dtype_consistency(input, other)
    _check_where_scalar_unsigned("input", input, other, "other")
    _check_where_scalar_unsigned("other", other, input, "input")

    if condition.dtype == DataType.DT_UINT8:
        raise PyptoError(
            0xF00002,
            TypeError(
                "where() does not support DT_UINT8 condition from Python API. "
                "condition dtype must be DT_BOOL. DT_UINT8 is only supported in the C++ API "
                "for packed boolean masks (1 uint8 = 8 bools)."
            ),
        )

    if isinstance(input, pypto_impl.Tensor) or isinstance(input, pypto_impl.Element):
        input_base = input
    else:
        input_base = pypto_impl.Element(pypto_impl.DT_FP32, input)

    if isinstance(other, pypto_impl.Tensor) or isinstance(other, pypto_impl.Element):
        other_base = other
    else:
        other_base = pypto_impl.Element(pypto_impl.DT_FP32, other)
    return pypto_impl.Where(condition, input_base, other_base)


@op_wrapper
def pad(x: Tensor, padding: Sequence[int], mode: str = "constant", value: Union[float, int] = 0) -> Tensor:
    """
    Pads the input tensor.

    Padding size:
    The padding size by which to pad some dimensions of input are described starting from
    the last dimension and moving forward.
    For padding has format (d_last_dim, d_last_dim-1, ..., d_last_dim-k).
    Current implementation supports padding the last 2 dimensions (Right and Bottom) with
    constant values.

    Parameters
    ----------
    x : Tensor
        The input tensor to be padded.
    padding : tuple or list of int
        m-elements tuple, where m/2 <= input dimensions and m is even.
        Format is (pad_left, pad_right, pad_top, pad_bottom, ...).
        Note: Currently only supports pad_left=0 and pad_top=0 (Right/Bottom padding only).
        All padding values must be non-negative. Negative padding values are NOT supported.
    mode : str, optional
        'constant', 'reflect', 'replicate' or 'circular'. Default: 'constant'
        Note: Currently only 'constant' is supported.
    value : Union[float, int], optional
        fill value for 'constant' padding. Default: 0
        Note: For float types (DT_FP32, DT_FP16, DT_BF16), the value supports
        arbitrary floating-point values including -inf, inf.
        For integer types (DT_INT8, DT_INT16, DT_INT32, DT_UINT8, DT_UINT16, DT_UINT32),
        the value must be an integer.
        The data type of the padding value will automatically be converted to match
        that of the input Tensor.

    Returns
    -------
    Tensor
        Padded tensor.

    Raises
    ------
    TypeError
        If input is not a Tensor.
        If pad is not a sequence of integers.
    ValueError
        If pad length is not even.
        If any padding value is negative.

    Examples
    --------
    t4d = pypto.tensor([1, 1, 2, 2], pypto.DT_FP32)
    # Pad last dim by (0, 1) -> Right pad 1
    # Pad 2nd to last dim by (0, 1) -> Bottom pad 1
    p1 = (0, 1, 0, 1)
    out = pypto.pad(t4d, p1, "constant", 0.0)

    Input t4d: [[[[0.0, 1.0],
                [2.0, 3.0]]]]

    Output out: [[[[0.0, 1.0, 0.0],
                [2.0, 3.0, 0.0],
                [0.0, 0.0, 0.0]]]]

    """

    pad_list = list(padding)
    _check_pad_value(x, value)
    if isinstance(value, int):
        return pypto_impl.Pad(x, pad_list, mode, pypto_impl.Element(x.dtype, value))
    else:
        return pypto_impl.Pad(x, pad_list, mode, pypto_impl.Element(x.dtype, float(value)))


@op_wrapper
def fillpad(x: Tensor, mode: str = "constant", value: Union[float, int] = 0) -> Tensor:
    """
    Fills the padding region of the input tensor with a constant value.

    Unlike `pad`, this function does not change the tensor shape. It fills the
    padding region (the area beyond valid_shape) with the specified value.

    Parameters
    ----------
    x : Tensor
        The input tensor with padding region to be filled.
    mode : str, optional
        'constant'. Default: 'constant'
        Note: Currently only 'constant' is supported.
    value : Union[float, int], optional
        fill value for 'constant' padding. Default: 0
        Note: For float types (DT_FP32, DT_FP16, DT_BF16), the value supports
        arbitrary floating-point values including -inf, inf.
        For integer types (DT_INT8, DT_INT16, DT_INT32, DT_UINT8, DT_UINT16, DT_UINT32),
        the value must be an integer.
        The data type of the padding value will automatically be converted to match
        that of the input Tensor.

    Returns
    -------
    Tensor
        Tensor with padding region filled.

    Raises
    ------
    TypeError
        If input is not a Tensor.

    Examples
    --------
    # Create a tensor with valid_shape smaller than actual shape
    a = pypto.tensor([4, 4], pypto.DT_FP32)
    # Assume valid_shape is [2, 2], the rest [2:4, 2:4] is padding region
    out = pypto.fillpad(a, "constant", "-inf")

    Input a: [[1.0, 2.0, 0.0, 0.0],
              [3.0, 4.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0]]

    Output out: [[1.0, 2.0, -inf, -inf],
                 [3.0, 4.0, -inf, -inf],
                 [-inf, -inf, -inf, -inf],
                 [-inf, -inf, -inf, -inf]]

    """

    _check_pad_value(x, value)
    if isinstance(value, int):
        return pypto_impl.FillPad(x, mode, pypto_impl.Element(x.dtype, value))
    else:
        return pypto_impl.FillPad(x, mode, pypto_impl.Element(x.dtype, float(value)))


@op_wrapper
def one_hot(input: Tensor, num_classes: int) -> Tensor:
    """
    Converts a tensor of indices to one-hot encoded tensor.

    Parameters
    ----------
    input : Tensor
        LongTensor containing class indices of any shape (*)
    num_classes : int
        Total number of classes.

    Returns
    -------
    Tensor
        One-hot encoded tensor(LongTensor) of shape (*, num_classes) where:
        - 1 is placed at the index specified by input value
        - 0 is placed everywhere else

    Examples
    --------
    a = pypto.tensor([3], pypto.DT_INT32)
    out = pypto.one_hot(a, 5)

    Input a:    [0 2 4]
    Input num_classes:  5
    Output out: [[1, 0, 0, 0, 0],
                 [0, 0, 1, 0, 0],
                 [0, 0, 0, 0, 1]]

    """
    if not isinstance(input, pypto_impl.Tensor):
        raise PyptoError(0xF00001, TypeError("input must be a `Tensor`"))
    if not isinstance(num_classes, int):
        raise PyptoError(0xF00001, TypeError("num_classes must be an `int`"))
    if num_classes == -1:
        raise PyptoError(0xF00003, RuntimeError("num_classes must be specified"))
    if num_classes <= 0:
        raise PyptoError(0xF00003, RuntimeError("num_classes must be a positive integer"))
    return pypto_impl.OneHot(input, num_classes)


@op_wrapper
def expand_exp_dif(input: Tensor, other: Tensor) -> Tensor:
    """Computes the exp dif of `input` and `other`.

    This function calculates the formula: `out = e ** (input - other)`.

    Parameters
    ----------
    input : Tensor
        The first input tensor.
    other : Tensor
        The second input tensor.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise expand exp dif.

    Examples
    --------
    x = pypto.tensor([2, 3], pypto.DT_FP32)
    y = pypto.tensor([1, 3], pypto.DT_FP32)
    z = pypto.expand_exp_dif(x, y)

    Input x:      [[1, 2, 3], [4, 5, 6]]
    Input y:      [[1, 2, 3]]
    Output z :    [[ 1.      ,  1.      ,  1.      ],
                   [20.085537, 20.085537, 20.085537]]

    x = pypto.tensor([2, 3], pypto.DT_FP32)
    y = pypto.tensor([2, 1], pypto.DT_FP32)
    z = pypto.expand_exp_dif(x, y)

    Input x:      [[1, 2, 3], [4, 5, 6]]
    Input x:      [[1], [2]]
    Output z :    [[ 1.       ,  2.718282 ,  7.3890557],
                   [ 7.3890557, 20.085537 , 54.59815  ]]
    """
    return pypto_impl.ExpandExpDif(input, other)
