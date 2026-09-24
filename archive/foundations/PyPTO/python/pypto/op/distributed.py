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

from typing import Optional, Union

from pypto import pypto_impl
from pypto._controller import loop
from pypto._op_wrapper import op_wrapper
from pypto._utils import to_syms
from pypto.enum import AtomicType, DataType, OpType
from pypto.symbolic_scalar import SymbolicScalar
from pypto.tensor import ShmemTensor, Tensor


@op_wrapper
def create_shmem_tensor(
    group_name: str,
    n_pes: int,
    dtype: DataType,
    shape: list[int],
) -> ShmemTensor:
    """Creates a symmetric tensor in shared memory.

    Parameters
    ----------
    group_name : str
        The name of the communication group.
    n_pes : int
        The total number of processes in the communication group.
    dtype : DataType
        The data type of the tensor.
    shape : list of int
        The shape of the tensor to create.

    Returns
    -------
    data : ShmemTensor
        A shared memory tensor used by communication operators to transport data.

    Examples
    --------
    data = pypto.distributed.create_shmem_tensor(
        "tp",
        8,
        pypto.DT_FP16,
        [64, 128],
    )
    """
    t = ShmemTensor()
    for _ in loop(1, name="CREATE_SHMEM_TENSOR", idx_name="_"):
        pypto_impl.CreateShmemTensor(group_name, n_pes, dtype, shape, t.base())
    return t


@op_wrapper
def create_shmem_signal(group_name: str, n_pes: int) -> ShmemTensor:
    """Creates a signal tensor in shared memory.

    Parameters
    ----------
    group_name : str
        The name of the communication group.
    n_pes : int
        The total number of processes in the communication group.

    Returns
    -------
    signal : ShmemTensor
        A signal tensor to coordinate process execution.

    Examples
    --------
    signal = pypto.distributed.create_shmem_signal("tp", 8)
    """
    t = ShmemTensor()
    for _ in loop(1, name="CREATE_SHMEM_SIGNAL", idx_name="_"):
        pypto_impl.CreateShmemSignal(group_name, n_pes, t.base())
    return t


@op_wrapper
def shmem_view(
    src: ShmemTensor,
    shape: list[int],
    offsets: list[Union[int, SymbolicScalar]],
    *,
    valid_shape: Optional[list[Union[int, SymbolicScalar]]] = None,
) -> ShmemTensor:
    """Extract a partial view from the input shmem tensor for subsequent computations.
       the behavior is similar to pypto.view

    Parameters
    ----------
    src: ShmemTensor
        The input tensor to extract a partial view, it must be a shmem tensor
        The constraints of the tensor shape and data type is same as pypto.view
    shape: list of int
        Get the shape of the shmem view.
        The constraints of this parameter is same as pypto.view
    offsets: list of int or SymbolicScalar
        Get the offset of each dimension relative to the input when obtaining the shmem view.
        The constraints of this parameter is same as pypto.view
    valid_shape: list of int or SymbolicScalar = None
        Optional parameter to retrieve the effective data size of the schematic block.
        The constraints of this parameter is same as pypto.view

    Returns
    -------
    ShmemTensor
        A partial view from the input tensor with the size of shape.

    Examples
    --------
    x = pypto.distributed.create_shmem_tensor("tp", 4, pypto.DT_FP32, [4, 8])
    shape = [4, 4]
    offsets = [0, 4]
    valid_shape = [2, 4]
    y = pypto.distributed.shmem_view(x, shape, offsets, valid_shape=valid_shape)
    """
    if valid_shape is None:
        return pypto_impl.ShmemView(src, shape, offsets)
    else:
        return pypto_impl.ShmemView(src, shape, to_syms(valid_shape), to_syms(offsets))


@op_wrapper
def shmem_put(
    src: Tensor,
    offsets: list[Union[int, SymbolicScalar]],
    dst: ShmemTensor,
    dst_pe: Union[int, SymbolicScalar],
    *,
    put_op: AtomicType = AtomicType.SET,
    pred: list[Tensor] = None,
) -> Tensor:
    """Asynchronously sends local GM data to a remote GM.

    Parameters
    ----------
    src: Tensor
        The source tensor located in local GM.
    offsets : list of int or SymbolicScalar
        The offset in the destination shared memory GM.
    dst: ShmemTensor
        The destination tensor in shared memory GM (symmetric memory).
    dst_pe : int or SymbolicScalar
        The pe of the destination device.
    put_op : AtomicType
        The type of atomic operation to apply during the data transfer.
    pred : list[Tensor]
        Predicate tensors used to control the execution dependency of the operation.

    Returns
    -------
    Tensor
        Output predicate tensors representing the completion dependency of the operation.

    Examples
    --------
    Send local GM data to dst_pe 1
    tile = pypto.distributed.shmem_put(
        local_tensor,
        [0, 0],
        shmem_tensor,
        1,
        put_op=pypto.AtomicType.SET,
        pred=pred_token,
    )
    """
    dummy = __normalize_pred(pred)
    dst = pypto_impl.ShmemView(dst, src.shape, offsets)
    return pypto_impl.ShmemPut(src, dst, dst_pe, put_op, dummy)


@op_wrapper
def shmem_get(
    src: ShmemTensor,
    src_pe: Union[int, SymbolicScalar],
    shape: list[int] = None,
    offsets: list[Union[int, SymbolicScalar]] = None,
    *,
    valid_shape: Optional[list[Union[int, SymbolicScalar]]] = None,
    pred: list[Tensor] = None,
) -> Tensor:
    """Asynchronously fetches data from a remote GM to local GM.

    Parameters
    ----------
    src : ShmemTensor
        The source tensor in remote GM.
    src_pe : int or SymbolicScalar
        The src_pe of the source device.
    shape : list of int
        The shape of the destination tensor.
    offsets : list of int or SymbolicScalar
        The offset of the destination tensor in local GM.
    valid_shape: list[int] = None
        Optional parameter to retrieve the effective data size of the schematic block.
        It is required that the valid_shape is smaller than the shape of the input.
    pred : list[Tensor]
        Predicate tensors used as control dependencies.

    Returns
    -------
    Tensor
        The destination tensor in local GM.


    Examples
    --------
    Load data from a remote device to local GM
    local_tensor = pypto.distributed.shmem_get(
        shmem_data,
        1,
        [128, 256],
        [0, 0],
        valid_shape=valid_shape,
        pred=pred_token,
    )
    """
    dummy = __normalize_pred(pred)
    if shape is not None and offsets is not None:
        if valid_shape is not None:
            src = pypto_impl.ShmemView(src, shape, valid_shape, offsets)
        else:
            src = pypto_impl.ShmemView(src, shape, offsets)
    return pypto_impl.ShmemGet(src, src_pe, dummy)


@op_wrapper
def shmem_signal(
    src: ShmemTensor,
    src_pe: Union[int, SymbolicScalar],
    signal: int,
    shape: list[int] = None,
    offsets: list[Union[int, SymbolicScalar]] = None,
    *,
    target_pe: Union[int, SymbolicScalar],
    sig_op: AtomicType = AtomicType.SET,
    pred: list[Tensor] = None,
) -> Tensor:
    """Send a signal to the target_pe, and the signal is triggered by an operation on some shmem tensor,
        which is described by src and src_pe.

    Parameters
    ----------
    src : ShmemTensor
        The shared memory tensor which triggers the signal.
    src_pe : int or SymbolicScalar
        The shmem tensor belongs to.
    signal: int
        The value of the signal sent to shared memory,
    shape : list of int
        The shapes of the shared memory signal;
    offsets : list of int or SymbolicScalar
        The offsets of the signal in shared memory.
    target_pe: int or SymbolicScalar
        The target device to receive this signal
    sig_op : AtomicType
        The type of atomic operation.
    pred : list[Tensor]
        Predicate tensors used as control dependencies.

    Returns
    -------
    Tensor
        The destination tensor in local GM.

    Examples
    --------
    Load data from a remote device to local GM
    dummy = pypto.distributed.shmem_signal(
        shmem_signal,
        1,
        2,
        [128, 256],
        [0, 0],
        target_pe=3,
        sig_op=pypto.AtomicType.SET,
        pred=pred_token,
    )
    """
    dummy = __normalize_pred(pred)
    if shape is not None and offsets is not None:
        src = pypto_impl.ShmemView(src, shape, offsets)
    if type(target_pe) is int and target_pe == -1:
        return pypto_impl.ShmemSignalAll(src, src_pe, signal, sig_op, dummy)
    else:
        return pypto_impl.ShmemSignal(src, src_pe, target_pe, signal, sig_op, dummy)


@op_wrapper
def shmem_wait_until(
    src: ShmemTensor,
    src_pe: Union[int, SymbolicScalar],
    cmp_value: int = 0,
    shape: list[int] = None,
    offsets: list[Union[int, SymbolicScalar]] = None,
    *,
    cmp: OpType = OpType.EQ,
    clear_signal: bool = False,
    pred: list[Tensor] = None,
) -> Tensor:
    """Wait for a signal at my_pe, and the signal is triggered by an operation on some shmem tensor,
        which is described by src and src_pe.

    Parameters
    ----------
    src : ShmemTensor
        The shared memory tensor which will trigger a signal
    src_pe: int or SymbolicScalar
        The shared memory tensor belongs to.
    cmp: OpType
        Comparison operation type for condition checking. only EQ is supported currently.
    cmp_value : int
        The value to wait for.
    shape : list of int
        The shapes of the shared memory signal;
    offsets : list of int or SymbolicScalar
        The offsets of the shmem signal.
    clear_signal : bool
        Whether to reset the signal after waiting.
    pred : list[Tensor]
        Predicate tensors used as control dependencies.

    Returns
    -------
    Tensor
        Output predicate tensors.

    Examples
    --------
    dummy = pypto.distributed.shmem_wait_until(
        shmem_signal,
        1,
        4,
        [128, 256],
        [0, 0],
        cmp=OpType.EQ,
        clear_signal=False,
        pred=pred_token,
    )
    """
    dummy = __normalize_pred(pred)
    if shape is not None and offsets is not None:
        src = pypto_impl.ShmemView(src, shape, offsets)
    return pypto_impl.ShmemWaitUntil(src, src_pe, cmp, cmp_value, clear_signal, dummy)


@op_wrapper
def shmem_barrier_all(
    src: ShmemTensor,
    pred: list[Tensor] = None,
) -> Tensor:
    """Synchronizes multiple devices within a communication group.

    Parameters
    ----------
    src : ShmemTensor
        The shared memory barrier signal.
    pred : list[Tensor]
        Predicate tensors used as control dependencies.

    Returns
    -------
    Tensor
        Output predicate tensors.


    Examples
    --------
    Synchronize devices within the communication group
    result = pypto.matmul(A_tile, B_tile, pypto.DT_FP16)
    dummy = pypto.distributed.shmem_barrier_all(shmem_barrier_signal, pred)
    """
    dummy = __normalize_pred(pred)
    return pypto_impl.ShmemBarrier(src, dummy)


@op_wrapper
def shmem_clear_data(
    src: ShmemTensor,
    shape: list[int] = None,
    offsets: list[Union[int, SymbolicScalar]] = None,
    *,
    pred: list[Tensor] = None,
) -> Tensor:
    """Clear the data of shmem tensor owned by my_pe.

    Parameters
    ----------
    src : ShmemTensor
        The shmem tensor to be cleared
    shape : list of int
        The shape of the shmem tensor to be cleared
    offsets : list of int or SymbolicScalar
        The offsets of the shmem tensor to be cleared
    pred : list[Tensor]
        Predicate tensors used as control dependencies.

    Returns
    -------
    Tensor
        Output predicate tensors.

    Examples
    --------
    Clear a shmem data tensor in shared memory
    data_clear_dummy = pypto.distributed.shmem_clear_data(
        shmem_data,
        [1, 128, 256],
        [0, 0, 0],
        pred=pred_token,
    )
    """
    dummy = __normalize_pred(pred)
    if shape is not None and offsets is not None:
        src = pypto_impl.ShmemView(src, shape, offsets)
    return pypto_impl.ShmemClearData(src, dummy)


@op_wrapper
def shmem_clear_signal(src: ShmemTensor, *, pred: list[Tensor] = None) -> Tensor:
    """Clear all the signal value of shmem tensor on my_pe.

    Parameters
    ----------
    src : ShmemTensor
        The signal of shmem tensor to be cleared;
    pred : list[Tensor]
        Predicate tensors used as control dependencies.

    Returns
    -------
    Tensor
        Output predicate tensors.

    Examples
    --------
    Clear a shmem signal tensor in shared memory
    data_clear_dummy = pypto.distributed.shmem_clear_signal(
        shmem_signal,
        pred=pred_token,
    )
    """
    dummy = __normalize_pred(pred)
    return pypto_impl.ShmemClearSignal(src, dummy)


@op_wrapper
def my_symbolic_pe(group_name: str) -> SymbolicScalar:
    """Gets the symbolic PE.

    Parameters
    ----------
    group_name : str
        The name of the communication group.

    Returns
    -------
    symbolic scalar
        Represents my_pe.

    Examples
    --------
       my_pe = pypto.distributed.my_symbolic_pe(group_name)
    """
    return pypto_impl.GetSymbolicScalarPeId(group_name)


def __normalize_pred(pred: Union[list[Tensor], None]) -> Tensor:
    if pred is None:
        return pypto_impl.Full(pypto_impl.Element(DataType.DT_INT32, 1), DataType.DT_INT32, [1, 1], to_syms([]))
    if len(pred) == 1:
        return pred[0]
    return pypto_impl.Nop(pred)
