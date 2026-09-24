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

from typing import List

import pypto

from .. import pypto_impl
from .._op_wrapper import op_wrapper
from ..enum import DataType
from ..error import PyptoError
from ..symbolic_scalar import SymInt
from ..tensor import Tensor


def uniform_impl(
    key: int, counter0: SymInt, counter1: int, shape: List[int], rounds: int = 10, dtype: "DataType" = None
) -> Tensor:
    """
    Generates uniform random numbers using the Philox algorithm.

    Philox is a counter-based random number generator that produces deterministic
    uniform random sequences based on a key and counter.

    Parameters
    ----------
    key : int
        A uint64 value that serves as the key for the uniform random number sequence.
    counter0 : SymInt
        The first uint64 value of the 128-bit counter. Can be a symbolic scalar.
    counter1 : int
        The second uint64 value of the 128-bit counter.
    shape : List[int]
        The shape of the output tensor (must be 1-dimensional).
    rounds : int, optional
        The number of rounds for the Philox algorithm (7 or 10). Default is 10.
    dtype : DataType, optional
        The data type of the output tensor. Supports DT_FP32, DT_FP16, DT_BF16.
        Default is DT_FP32.

    Returns
    -------
    Tensor
        A tensor of uniform random numbers with the specified shape and dtype.

    Examples
    --------
    key = 12345678901234
    counter0 = 0
    counter1 = 0
    shape = [1024]
    output = pypto.uniform(key, counter0, counter1, shape, rounds=10)
    """
    if len(shape) != 1:
        raise PyptoError(0xF00002, ValueError(f"shape must be 1-dimensional, got {len(shape)} dimensions"))
    if rounds not in [7, 10]:
        raise PyptoError(0xF00002, ValueError(f"rounds must be 7 or 10, got {rounds}"))

    if dtype is None:
        dtype = pypto_impl.DataType.DT_FP32

    valid_dtypes = [pypto_impl.DataType.DT_FP32, pypto_impl.DataType.DT_FP16, pypto_impl.DataType.DT_BF16]
    if dtype not in valid_dtypes:
        raise PyptoError(0xF00002, ValueError(f"dtype must be one of DT_FP32, DT_FP16, DT_BF16, got {dtype}"))

    return pypto_impl.Uniform(
        pypto_impl.Element(pypto_impl.DataType.DT_UINT64, key),
        counter0,
        pypto_impl.Element(pypto_impl.DataType.DT_UINT64, counter1),
        shape,
        pypto_impl.Element(pypto_impl.DataType.DT_UINT16, rounds),
        dtype,
    )


@op_wrapper
def uniform(shape, key, counter, alg, dtype) -> Tensor:
    """Return a tensor containing uniformly distributed random values.

    Parameters
    ----------
    shape: List[int]
        The shape of the output tensor.
    key: List[int]
        The key for random number generation.
    counter: List[int]
        The counter for random number generation.
    alg: List[int]
        The algorithm to use for random number generation, support 1(Philox) and 3(auto_select, select Philox).
    dtype: DataType
        The data type of the output tensor.
        The supported data types are DT_FP32, DT_FP16, and DT_BF16.

    Returns
    -------
    Tensor
        A tensor with the specified shape and data type, containing
        uniformly distributed random values in the range [0, 1).


    Examples
    --------
    shape = [4, 4]
    key = [1234]
    counter = [0, 1]
    alg = [1]

    y = pypto.uniform(shape, key, counter, alg, dtype)

    Output y:[[0.1689806  0.9725481  0.90036285 0.16582811]
              [0.1454581  0.48029935 0.02495587 0.99239147]
              [0.02835405 0.10649502 0.45283175 0.87260246]
              [0.6877538  0.24809706 0.95886254 0.24039495]]
    """
    if len(shape) < 1 or len(shape) > 4:
        raise PyptoError(0xF00002, ValueError(f"output shape dim should be in [1, 4], but got {len(shape)}."))

    if len(key) != 1:
        raise PyptoError(0xF00002, ValueError(f"input key number should be 1, but got {len(key)}."))

    if len(counter) != 2:
        raise PyptoError(0xF00002, ValueError(f"input counter number should be 2, but got {len(counter)}."))

    if len(alg) != 1:
        raise PyptoError(0xF00002, ValueError(f"input alg number should be 1, but got {len(alg)}."))

    alg = alg[0]
    if alg != 1 and alg != 3:
        raise PyptoError(0xF00002, ValueError("alg only support Philox."))

    tile_shapes = pypto.get_vec_tile_shapes()
    tile_shape_one_dim = 1
    for dim_num in tile_shapes:
        tile_shape_one_dim *= dim_num
    pypto.set_vec_tile_shapes(tile_shape_one_dim)

    shape_one_dim = 1
    for dim_num in shape:
        shape_one_dim *= dim_num

    counter0, counter1 = 0, counter[1]
    uniform_res = uniform_impl(key[0], counter0, counter1, [shape_one_dim], rounds=10, dtype=dtype)
    pypto.set_vec_tile_shapes(*tile_shapes)
    return pypto.reshape(uniform_res, shape)


@op_wrapper
def normal(shape, key, counter, alg, dtype) -> Tensor:
    """Return a tensor containing normally distributed random values.

    Parameters
    ----------
    shape: List[int]
        The shape of the output tensor.
    key: List
        The key for random number generation.
    counter: List[int]
        The counter for random number generation.
        Note: counter[0] is hardcoded to 0 internally for alignment with
        CANN ACLNN L2 API (aclnnNormalFloatFloat), which constructs the
        Philox counter as [0, offset]. Only counter[1] (offset) is used.
    alg: List[int]
        The algorithm to use for random number generation, support 1(Philox) and 3(auto_select, select Philox).
    dtype: DataType
        The data type of the output tensor.
        The supported data types are DT_FP32, DT_FP16, and DT_BF16.

    Returns
    -------
    Tensor
        A tensor with the specified shape and data type, containing
        uniformly distributed random values in the range [0, 1).


    Examples
    --------
    shape = [4, 4]
    key = [1234]
    counter = [0, 1]
    alg = [1]

    y = pypto.normal(shape, key, counter, alg, dtype)

    Output y:[[-0.32364845  1.8577391   0.39556974  0.2311697 ]
              [ 0.24243996 -1.9485782  -0.12983137  2.7137496 ]
              [ 1.6558666   2.0938187  -0.90338254  0.8765667 ]
              [ 0.86518306  0.01034508  0.2893259   0.01748212]]
    """

    def box_muller(input: Tensor) -> Tensor:
        tensor_len = input.shape[0]
        u1 = pypto.gathermask(input, 1)
        u2 = pypto.gathermask(input, 2)

        eps = 1.0e-7
        m_pi = 3.14159265358979323846
        u1 = pypto.maximum(u1, eps)
        v1 = pypto.mul(u2, 2.0 * m_pi)
        v2 = pypto.sqrt(pypto.mul(pypto.log(u1), -2.0))
        f0 = pypto.sin(v1)
        f1 = pypto.cos(v1)
        f2 = pypto.mul(f0, v2)
        f3 = pypto.mul(f1, v2)
        f4 = pypto.zeros([tensor_len], dtype=pypto.DT_FP32)
        u1_index = pypto.arange(0, tensor_len, 2)
        u2_index = pypto.add(u1_index, 1)
        pypto.index_put_(f4, (u1_index,), f2)
        pypto.index_put_(f4, (u2_index,), f3)
        return f4

    if len(shape) < 1 or len(shape) > 4:
        raise PyptoError(0xF00002, ValueError(f"output shape dim should be in [1, 4], but got {len(shape)}."))

    if len(key) != 1:
        raise PyptoError(0xF00002, ValueError(f"input key number should be 1, but got {len(key)}."))

    if len(counter) != 2:
        raise PyptoError(0xF00002, ValueError(f"input counter number should be 2, but got {len(counter)}."))

    if len(alg) != 1:
        raise PyptoError(0xF00002, ValueError(f"input alg number should be 1, but got {len(alg)}."))

    alg = alg[0]
    if alg != 1 and alg != 3:
        raise PyptoError(0xF00002, ValueError("alg only support Philox."))

    tile_shapes = pypto.get_vec_tile_shapes()
    tile_shape_one_dim = 1
    for dim_num in tile_shapes:
        tile_shape_one_dim *= dim_num
    pypto.set_vec_tile_shapes(tile_shape_one_dim)

    shape_one_dim = 1
    for dim_num in shape:
        shape_one_dim *= dim_num

    counter0, counter1 = 0, counter[1]
    uniform_res = uniform_impl(
        key[0], counter0, counter1, [(shape_one_dim + 1) // 2 * 2], rounds=10, dtype=pypto.DT_FP32
    )

    normal_res = box_muller(uniform_res)
    if shape_one_dim % 2 != 0:
        normal_res = normal_res[:-1]

    if dtype != pypto.DT_FP32:
        normal_res = pypto.cast(normal_res, dtype)
    else:
        normal_res = pypto.add(normal_res, 0)
    pypto.set_vec_tile_shapes(*tile_shapes)
    return pypto.reshape(normal_res, shape)
