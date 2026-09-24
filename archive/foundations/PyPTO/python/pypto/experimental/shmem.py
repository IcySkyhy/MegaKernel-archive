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

from typing import List, Optional, Union

from .. import pypto_impl
from .._op_wrapper import op_wrapper
from .._utils import to_syms
from ..enum import AtomicType, DataType
from ..symbolic_scalar import SymbolicScalar
from ..tensor import ShmemTensor, Tensor


@op_wrapper
def shmem_store(
    src: Tensor,
    offsets: List[Union[int, SymbolicScalar]],
    dst: ShmemTensor,
    dst_pe: Union[int, SymbolicScalar],
    *,
    put_op: AtomicType = AtomicType.SET,
    pred: List[Tensor] = None,
) -> Tensor:
    """Stores local UB data to remote device Global Memory.

    Parameters
    ----------
    src : Tensor
        The source tensor in local UB.
    offsets : list of int or SymbolicScalar
        The offsets in the destination tensor.
    dst : ShmemTensor
        The destination tensor on the remote device (symmetric memory).
    dst_pe : int or SymbolicScalar
        The pe of the destination device.
    put_op : AtomicType
        The type of atomic operation to apply during the data transfer.
    pred : list[Tensor]
        Predicate tokens used as control dependencies.

    Returns
    -------
    Tensor
        Output predicate tokens.

    Examples
    --------
    Store the computation result from UB to pe 2
    result = pypto.matmul(A_tile, B_tile, pypto.DT_FP16)
    out = pypto.experimental.shmem_store(
        result,
        [0, 0],
        sym_buffer,
        2,
        pred=pred_token,
    )
    """
    if pred is None:
        dummy = pypto_impl.Full(pypto_impl.Element(DataType.DT_INT32, 1), DataType.DT_INT32, [1, 1], to_syms([]))
    else:
        src_id = src.Id()
        valid_pred = [p for p in pred if p.Id() != src_id]
        if len(valid_pred) == 0:
            dummy = pypto_impl.Full(pypto_impl.Element(DataType.DT_INT32, 1), DataType.DT_INT32, [1, 1], to_syms([]))
        elif len(valid_pred) == 1:
            dummy = valid_pred[0]
        else:
            dummy = pypto_impl.Nop(valid_pred)
    dst_tile = pypto_impl.ShmemView(dst, src.shape, offsets)
    return pypto_impl.ShmemStore(src, dst_tile, dst_pe, put_op, dummy)


@op_wrapper
def shmem_load(
    src: ShmemTensor,
    src_pe: Union[int, SymbolicScalar],
    shape: List[int] = None,
    offsets: List[Union[int, SymbolicScalar]] = None,
    *,
    pred: List[Tensor] = None,
    valid_shape: Optional[List[Union[int, SymbolicScalar]]] = None,
) -> Tensor:
    """Loads data from remote device Global Memory to local UB.

    Parameters
    ----------
    src : ShmemTensor
        The source tensor on the remote device (symmetric memory).
    src_pe : int or SymbolicScalar
        The pe of the source device.
    shape : list of int
        The shape of the source tensor.
    offsets : list of int or SymbolicScalar
        The offsets of the source tensor.
    pred : list[Tensor]
        Predicate token used as a control dependency.
    valid_shape: list of int or SymbolicScalar = None
        Optional parameter to retrieve the effective data size of the schematic block.
        It is required that the valid_shape is smaller than the shape of the input.

    Returns
    -------
    Tensor
        The tensor in local UB (can be used directly for computation).

    Examples
    --------
    Load a [64, 128] tile from pe 1 into UB
    wait_until_out = pypto.distributed.shmem_wait_until(
        shmem_signal,
        0,
        4,
        shape,
        offset,
        cmp=pypto.OpType.EQ,
        clear_signal=True,
        pred=None,
    )
    tile = pypto.experimental.shmem_load(
        shmem_data,
        1,
        [128, 256],
        [0, 0],
        pred=wait_until_out,
        valid_shape=None,
    )
    The tile is now in UB and can be used directly for computation
    result = pypto.exp(tile)
    """
    if pred is None:
        dummy = pypto_impl.Full(pypto_impl.Element(DataType.DT_INT32, 1), DataType.DT_INT32, [1, 1], to_syms([]))
    else:
        dummy = pred[0] if len(pred) == 1 else pypto_impl.Nop(pred)
    if shape is not None and offsets is not None:
        if valid_shape is not None:
            src = pypto_impl.ShmemView(src, shape, to_syms(valid_shape), to_syms(offsets))
        else:
            src = pypto_impl.ShmemView(src, shape, offsets)
    return pypto_impl.ShmemLoad(src, src_pe, dummy)
