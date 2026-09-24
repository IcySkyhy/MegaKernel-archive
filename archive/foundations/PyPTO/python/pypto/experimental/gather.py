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
""" """

from .. import pypto_impl
from .._op_wrapper import op_wrapper
from ..tensor import Tensor


@op_wrapper
def gather_in_l1(
    src: Tensor, indices: Tensor, block_table: Tensor, block_size: int, size: int, is_b_matrix: bool, is_trans: bool
) -> Tensor:
    """gather_in_l1."""

    return pypto_impl.gather_in_l1(src, indices, block_table, block_size, size, is_b_matrix, is_trans)


@op_wrapper
def gather_in_ub(param: Tensor, indices: Tensor, block_table: Tensor, block_size: int, axis: int) -> Tensor:
    """gather_in_ub."""
    """
    Custom Operator for Sparse Attention Mechanism:
    Extracts selected key-value (KV) vectors from the PagedAttention KV cache based on token indices.

    This operator assumes that the KV cache is stored in GM (Global Memory),
    and the extracted results are written to UB (Unified Buffer).

    Parameters:
    -----------
    param : Tensor
        Input tensor representing the KV cache in GM.
        Only 2-D tensors are supported, with shape [token_num, hidden_size].
    indices : Tensor
        Input tensor containing the indices of selected tokens (e.g., TopK results).
        Only 2-D tensors are supported, with shape [1, k].
    blockTable : Tensor
        Input tensor representing the page table in PagedAttention.
        Only 2-D tensors are supported, with shape [1, block_table_size].
    blockSize : int
        Input scalar indicating the number of tokens per block in PagedAttention.
    axis : int
        Input scalar specifying the dimension along which the operation is applied.
        Only -2 (second-to-last dimension) is currently supported.
    out: Tensor
        Contains the KV vectors (either key or value, depending on the input param) corresponding to
        the k selected tokens specified by indices with shape [k, hidden_size].

    Examples
    --------
    param = pypto.tensor([6,4], pypto.DT_FP16, "src")
    offsets = pypto.tensor([1,3], pypto.DT_INT32, "offsets")
    pageTable = pypto.tensor([1,3], pypto.DT_INT32, "pageTable")
    blockSize = 2
    out = pypto.experimental.gather_in_ub(param, offsets, pageTable, blockSize, -2)

    Input param:
    [
        [  0,  1,  2,  3],  # 0
        [ 10, 11, 12, 13],  # 1
        [ 20, 21, 22, 23],  # 2
        [ 30, 31, 32, 33],  # 3
        [ 40, 41, 42, 43],  # 4
        [ 50, 51, 52, 53],  # 5
    ]
    Input indices:  [0, 4, 3]
    Input blockTable:  [0, 2, 1]

    Output out:
    [
        [  0,  1,  2,  3],
        [ 20, 21, 22, 23],
        [ 50, 51, 52, 53],
    ]
    """
    return pypto_impl.gather_in_ub(param, indices, block_table, block_size, axis)
