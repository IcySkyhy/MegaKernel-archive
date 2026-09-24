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
""" """

from typing import Dict, List, Optional, Sequence, Tuple, Union, overload

from . import pypto_impl
from ._op_wrapper import op_wrapper
from ._utils import to_syms
from .enum import AtomicRMWMode, DataType
from .error import FeError
from .symbolic_scalar import SymbolicScalar
from .tensor import Tensor

_view_original_shapes: Dict[int, list] = {}
_view_input_ids: Dict[int, int] = {}
_view_input_names: Dict[int, str] = {}
_gathermask_checks: list = []


@overload
def assemble(
    src: Tensor,
    offsets: List[Union[int, SymbolicScalar]],
    dst: Tensor,
    parallel: bool = False,
) -> None:
    """
    Assembles a small Tensor into a larger Tensor based on specified offsets.

    Parameters
    ---------
    src: Tensor
        The small input tensor to be assembled into the larger tensor

    offsets : List[int] or List[SymbolicScalar]
        List of offset values indicating where the input tensor should be placed in the output tensor.
        It is required that the offsets is smaller than the shape of out.

    dst: Tensor
        The larger output tensor that will contain the assembled input tensor

    parallel : bool, optional
        Whether assemble can be executed in parallel. Defaults to False.

    Examples
    ---------
    x = pypto.tensor([2, 2], pypto.data_type.DT_FP32)
    out = pypto.tensor([4, 4], pypto.data_type.DT_FP32)
    offsets = [0, 0]
    pypto.assemble(x, offsets, out)

    Input x:[[1 1],
            [1,1]]
          out:[[0 0 0 0],
               [0 0 0 0],
               [0 0 0 0],
               [0 0 0 0]]

    Output out:[[1 1 0 0]
                [1 1 0 0]
                [0 0 0 0]
                [0 0 0 0]]
    """
    ...


@overload
def assemble(
    srcs: Sequence[Tuple[Tensor, List[Union[int, SymbolicScalar]]]],
    dst: Tensor,
    parallel: bool = False,
) -> None:
    """
    Assembles multiple small Tensors into a larger Tensor based on specified offsets.

    Parameters
    ---------
    srcs: Sequence[Tuple[Tensor, List[Union[int, SymbolicScalar]]]]
        Collect multiple small tensors into a larger output tensor.

    dst: Tensor
        The larger output tensor

    parallel : bool, optional
        Whether assemble can be executed in parallel. Defaults to False.

    Examples
    ---------
    x = pypto.tensor([2, 2], pypto.DT_FP32)
    y = pypto.tensor([2, 2], pypto.DT_FP32)
    out = pypto.tensor([4, 4], pypto.DT_FP32)
    inputs = [(x, [0, 0]), (y, [2, 2])]
    pypto.assemble([(x, [0, 0]), (y, [2, 2])], out, parallel=True)

    Input x:[[1 1],
            [1,1]]
          y:[[1, 1],
            [1, 1]]
          out:[[0 0 0 0],
               [0 0 0 0],
               [0 0 0 0],
               [0 0 0 0]]

    Output out:[[1 1 0 0]
                [1 1 0 0]
                [0 0 1 1]
                [0 0 1 1]]
    """
    ...


def assemble(*args, parallel: bool = False) -> None:
    if isinstance(args[0], Sequence):
        srcs, dst = args
        if len(srcs) == 0:
            return
        srcs = [(src.base(), to_syms(offsets)) for src, offsets in srcs]
        pypto_impl.Assemble(srcs, dst.base(), parallel)
    else:
        src, offsets, dst = args
        pypto_impl.Assemble(src.base(), to_syms(offsets), dst.base(), parallel)


def min(a: "SymbolicScalar | int", b: "SymbolicScalar | int") -> "SymbolicScalar":
    if isinstance(a, int):
        a = SymbolicScalar(a)
    return a.min(b)


def max(a: "SymbolicScalar | int", b: "SymbolicScalar | int") -> "SymbolicScalar":
    if isinstance(a, int):
        a = SymbolicScalar(a)
    return a.max(b)


def atomic_add(src: Tensor, offsets: List[Union[int, SymbolicScalar]], dst: Tensor):
    """
    Atomic add operation on the input tensor.

    Parameters
    ---------
    src: Tensor
        The input tensor to be added.

    offsets : List[int | SymbolicScalar]
        List of offset values to be added to the input tensor.

    dst: Tensor
        The output tensor that will contain the atomic add result.
    """
    pypto_impl.AtomicRMW(src.base(), to_syms(offsets), dst.base(), AtomicRMWMode.ADD)


@op_wrapper
def reshape(
    input: Tensor,
    shape: List[Union[int, SymbolicScalar]],
    *,
    valid_shape: Optional[List[Union[int, SymbolicScalar]]] = None,
    inplace: bool = False,
) -> Tensor:
    """
    Reshape the input Tensor into a new tensor with the specific shape.

    Parameters
    ---------
    input: pypto.Tensor
        The input tensor to be reshaped.

    shape : List[int]
        The new shape of the tensor. The total number of elements must match the input tensor.
        If shape is [-1], all dimensions will be flattened to 1D.

    valid_shape : List[int], optional
        An optional parameter specifying the valid shape for partial reshapeing or padding.
        If provided, it may be used to define the effective part of the new shape.

    inplace : bool, optional
        An optional parameter determines memory sharing behavior between input and out tensors.
        If True, performs the reshape operation in-place, sharing the same storage with the input tensor.
        If False, creates a new tensor with the reshaped shape.

    Return
    ------
    pypto.Tensor
        A new tensor with the specific shape.

    Examples
    ---------
    x = pypto.tensor([2, 2], pypto.DT_FP32)
    y = pypto.reshape(x, [4, 1], valid_shape=[2, 1])
    z = pypto.add(y, 1.0)

    input x: [[1, 2],
              [3, 4]]
    output y: [[1],
               [2],
               [3],
               [4]
           z: [[2],
               [3],
               [3],
               [4]]

    # inplace
    x = pypto.tensor([2, 2], pypto.DT_FP32)
    y = pypto.reshape(x, [1, 4], inplace=True)

    input x: [[1, 2],
              [3, 4]]
    output y: [1, 2, 3, 4]

    # flatten to 1D
    x = pypto.tensor([4, 3, 2, 2], pypto.DT_FP32)
    y = pypto.reshape(x, [-1])
    # output y shape: [48]
    """
    if valid_shape is None and inplace:
        out = pypto_impl.Reshape(input, shape, inplace)
    else:
        if not all(isinstance(s, int) for s in shape):
            raise FeError(
                TypeError(
                    f"reshape() requires integer shape when using non-inplace reshape, "
                    f"but got [{', '.join(type(s).__name__ for s in shape)}]. "
                    f"Use 'inplace=True' for dynamic shapes."
                )
            )
        out = pypto_impl.Reshape(input, shape, [] if valid_shape is None else valid_shape, inplace)
    return out


@op_wrapper
def clone(input: Tensor) -> Tensor:
    """
    Clone the input Tensor into a new tensor with the same shape.

    Parameters
    ---------
    input: pypto.Tensor
        The input tensor to be cloned.

    Return
    ------
    pypto.Tensor
        A new tensor with the same shape.

    Examples
    ---------
    x = pypto.tensor([2, 2], pypto.DT_FP32)
    y = pypto.clone(y)

    input x: [[1, 2],
              [3, 4]]
    output y: [[1, 2],
              [3, 4]]
    """
    return pypto_impl.Assign(input)


@op_wrapper
def unsqueeze(input: Tensor, dim: int) -> Tensor:
    """Add a new dimension of size 1 to a tensor at a specified position.

    This operation increases the tensor's dimensionality while preserving
    the total number of elements (since the new dimension has size 1).

    Parameters
    ----------
    input : Tensor
        The input tensor to which a new dimension will be added.
        Supported data types are: DT_FP32, DT_FP16, DT_BF16.
        Empty tensors are not supported, and the shape size must not exceed 2147483647 (i.e., INT32_MAX).

    dim : int
        The position(index) where the new dimension is inserted.
        It must be within the range of [-input.dim - 1, input.dim]

    Returns
    -------
    Tensor
        A new tensor with the same data as the input tensor, but with an additional dimension
        of size 1 inserted at the specified dim position.

    Examples
    --------
    x = pypto.tensor([2, 3], pypto.DT_FP32)
    y = pypto.unsqueeze(x, 0)

    Input x:[[1, 2, 3],
             [4, 5, 6]]
    Output y:[[[1, 2, 3],
               [4, 5, 6]]]

    """
    return pypto_impl.Unsqueeze(input, dim)


@op_wrapper
def view(
    input: Tensor,
    shape: List[int] = None,
    offsets: List[Union[int, SymbolicScalar]] = None,
    *,
    valid_shape: Optional[List[Union[int, SymbolicScalar]]] = None,
    dtype: DataType = None,
) -> Tensor:
    """Extract a partial view from the input tensor for subsequent computations.
       WARNING: view has a very different behavior from torch.view, it is more like slice.

    Parameters
    ----------
    input: Tensor
        The input tensor to extract a partial view.
        The supported data types are: DT_FP32, DT_FP16, DT_BF16.Empty Tensors are not supported,
        and the Shape Size must not exceed 2147483647 (i.e., INT32_MAX).
    shape: List[int]
        Get the shape of the view.
        The Shape Size must not exceed 2147483647 (i.e., INT32_MAX).
    offsets: List[int]
        Get the offset of each dimension relative to the input when obtaining the view.
        It is required that the offsets are smaller than the shape of the input.
    valid_shape: List[int] = None
        Optional parameter to retrieve the effective data size of the schematic block.
        It is required that the valid_shape is smaller than the shape of the input.
    dtype: DataType
        The target data type for bitwise splitting operations.
        The supported data types are: DT_FP32, DT_FP16, DT_BF16, DT_INT8.

    Returns
    -------
    Tensor
        A partial view from the input tensor with the size of shape.

    Examples
    --------
    x = pypto.tensor([4, 8], pypto.DT_FP32)
    shape = [4, 4]
    offsets = [0, 4]
    y = pypto.view(x, shape, offsets)

    Input x:[[1 1 2 2 3 3 4 4],
             [1 1 2 2 3 3 4 4],
             [1 1 2 2 3 3 4 4],
             [1 1 2 2 3 3 4 4]]
    Output y:[[3 3 4 4],
              [3 3 4 4],
              [3 3 4 4],
              [3 3 4 4]]

    # add valid_shape
    x = pypto.tensor([4, 8], pypto.DT_FP32)
    shape = [4, 4]
    offsets = [2, 4]
    valid_shape = [2, 4]
    y = pypto.view(x, shape, offsets, valid_shape=valid_shape)

    Input x:[[1 1 2 2 3 3 4 4],
             [1 1 2 2 3 3 4 4],
             [1 1 2 2 5 5 6 6],
             [1 1 2 2 5 5 6 6]]
    Output y:[[5 5 6 6],
              [5 5 6 6],
              [0 0 0 0],
              [0 0 0 0]]

    # use view_type
    x = pypto.tensor([4, 8], pypto.DT_FP32)
    y = pypto.view(x, dtype=pypto.DT_INT8)
    """
    if dtype is not None:
        return pypto_impl.View(input, dtype)
    elif valid_shape is None:
        result = pypto_impl.View(input, shape, offsets)
    else:
        result = pypto_impl.View(input, shape, to_syms(valid_shape), to_syms(offsets))

    def _get_tensor_id(t):
        if hasattr(t, '_base'):
            return t._base.Id()
        if hasattr(t, 'Id'):
            return t.Id()
        return t.id()

    def _get_tensor_name(t):
        if hasattr(t, '_base'):
            base = t._base
        else:
            base = t
        if hasattr(base, 'GetName'):
            return base.GetName()
        return ''

    _view_original_shapes[result.Id()] = list(shape)
    _view_input_ids[result.Id()] = _get_tensor_id(input)
    _view_input_names[result.Id()] = _get_tensor_name(input)
    return result
