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

from functools import wraps
from typing import TYPE_CHECKING, List, Optional

from ._utils import get_torch_npu
from .enum import DataType, TileOpFormat
from .error import FeError, VerifyError
from .tensor import Tensor

if TYPE_CHECKING:
    import torch


def _count_calls(func):
    count = 0

    @wraps(func)
    def wrapper(
        tensor,
        name: str = "",
        *,
        dynamic_axis: Optional[List[int]] = None,
        tensor_format: Optional[TileOpFormat] = None,
        dtype: Optional[DataType] = None,
    ):
        nonlocal count
        count += 1
        if name == "":
            name = f"TENSOR_{count}"
        return func(tensor, name, dynamic_axis, tensor_format, dtype)

    return wrapper


def _set_shape_nz_aligned(tensor, ori_shape):
    """
    Set shape alignment requirements for NZ format tensors.

    NZ format is a data layout format specific to Ascend AI processors,
    requiring specific alignment:
    - Inner axis: aligned to 32-byte boundary
    - Outer axis: aligned to 16 elements

    Args:
        tensor: The tensor object (used to get element size)
        ori_shape: Original tensor shape as a list/tuple

    Returns:
        list: Aligned shape for NZ format
    """
    if len(ori_shape) <= 1:
        return ori_shape
    block_align_bytes = 32
    block_align_size = 16
    dtype_bytes = tensor.element_size()
    if dtype_bytes <= 0:
        return ori_shape
    c0size = block_align_bytes // dtype_bytes
    if c0size <= 0:
        return ori_shape
    dyn_shape = ori_shape.copy()
    dyn_shape[-1] = (ori_shape[-1] + c0size - 1) // c0size * c0size
    dyn_shape[-2] = (ori_shape[-2] + block_align_size - 1) // block_align_size * block_align_size

    return dyn_shape


@_count_calls
def from_torch(
    tensor,
    name: str = "",
    dynamic_axis: Optional[List[int]] = None,
    tensor_format: Optional[TileOpFormat] = None,
    dtype: Optional[DataType] = None,
):
    """
    convert the input into a PyPTO Tensor

    Parameters
    ----------
    tensor: object
        The input tensor to be converted. Currently, supports PyTorch tensors.
    name: str
        The name of the resulting PyPTO Tensor.
    dynamic_axis: List[int]
        Specifies which axes of the tensor should be marked as dynamic.
    tensor_format: TileOpFormat
        Specifies the format of the resulting PyPTO Tensor.
    dtype: DataType
        Specifies the data type of the resulting PyPTO Tensor.

    Returns
    -------
    Tensor
        A PyPTO Tensor object containing the following properties:
        - shape: The dimensions of the tensor.
        - name: The specified name of the tensor.
        - data_ptr: The memory address of the tensor data.
        - format: The format of the tensor (e.g., TILEOP_ND or TILEOP_NZ).
        - dtype: The dtype of the tensor.

    Examples
    --------
    >>> x= torch.randn(2, 3)
    >>> x_pto = pypto.from_torch(x)
    >>> print(x_pto.shape)
    [2, 3]
    >>> y = torch.randn(2, 3)
    >>> y_pto = pypto.from_torch(y, "input_tensor", dynamic_axis=[0])
    >>> print(y_pto.shape)
    [SymbolicScalar(RUNTIME_GetInputShapeDim(ARG_input_tensor,0)), 3]
    >>> y = torch.randn(2, 3)
    >>> y_pto = pypto.from_torch(y, "input_tensor", tensor_format=pypto.TileOpFormat.TILEOP_NZ)
    >>> print(y_pto.format)
    TileOpFormat.TILEOP_NZ
    >>> y = torch.randn(2, 3)
    >>> y_pto = pypto.from_torch(y, "input_tensor", dtype=pypto.DataType.DT_INT32)
    >>> print(y_pto.dtype)
    DataType.DT_INT32
    """
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise FeError(TypeError("input type is not currently supported."))

    if not tensor.is_contiguous():
        raise FeError(RuntimeError("not all tensors are contiguous"))

    dtype = _dtype_from(tensor.dtype) if dtype is None else dtype
    if tensor_format is None:
        tensor_format = TileOpFormat.TILEOP_ND
        if tensor.device.type == "npu":
            torch_npu = get_torch_npu()

            if torch_npu is not None and torch_npu.get_npu_format(tensor) == 29:
                tensor_format = TileOpFormat.TILEOP_NZ

    if tensor.dim() == 0:
        return Tensor(
            shape=tuple([1]),
            dtype=dtype,
            name=name,
            data_ptr=tensor.data_ptr(),
            format=tensor_format,
            device=tensor.device,
        )
    dyn_shape = list(tensor.shape)
    if tensor_format == TileOpFormat.TILEOP_NZ:
        dyn_shape = _set_shape_nz_aligned(tensor, dyn_shape)
    if dtype == DataType.DT_FP4_E1M2 or dtype == DataType.DT_FP4_E2M1:
        dyn_shape[-1] *= 2
    if dynamic_axis is not None:
        for axis in dynamic_axis:
            dyn_shape[axis] = -1
    return Tensor(
        shape=dyn_shape,
        dtype=dtype,
        name=name,
        data_ptr=tensor.data_ptr(),
        format=tensor_format,
        device=tensor.device,
        ori_shape=list(tensor.shape),
    )


_dtype_dict = {
    "torch.float16": DataType.DT_FP16,
    "torch.bfloat16": DataType.DT_BF16,
    "torch.float32": DataType.DT_FP32,
    "torch.float64": DataType.DT_DOUBLE,
    "torch.int8": DataType.DT_INT8,
    "torch.uint8": DataType.DT_UINT8,
    "torch.int16": DataType.DT_INT16,
    "torch.uint16": DataType.DT_UINT16,
    "torch.int32": DataType.DT_INT32,
    "torch.uint32": DataType.DT_UINT32,
    "torch.int64": DataType.DT_INT64,
    "torch.uint64": DataType.DT_UINT64,
    "torch.bool": DataType.DT_BOOL,
    "torch.float8_e4m3fn": DataType.DT_FP8E4M3,
    "torch.float8_e5m2": DataType.DT_FP8E5M2,
    "torch.float8_e8m0fnu": DataType.DT_FP8E8M0,
    "torch.float4_e2m1fn_x2": DataType.DT_FP4_E2M1X2,
}


def _dtype_from(dtype: 'torch.dtype') -> DataType:
    pto_dtype = _dtype_dict.get(dtype.__str__())
    if pto_dtype is None:
        raise FeError(ValueError(f"Input torch.dtype is not supported. Got {dtype}"))
    return pto_dtype


def _torch_dtype_from(dtype: DataType) -> "torch.dtype":
    """
    Convert the input into a torch.dtype.
    Dynamically checks for all FP8 support to ensure maximum compatibility
    across different PyTorch versions (including older versions without FP8).

    Parameters
    ----------
    dtype: DataType
        The input pypto.DataType to be converted.

    Returns
    -------
    torch.dtype
        The corresponding torch.dtype.

    Raises
    ------
    ValueError
        If the dtype is not supported or if a specific FP8 type is requested
        but not available in the current PyTorch version.
    """
    import torch

    _torch_dtype_dict = {
        DataType.DT_FP16: torch.float16,
        DataType.DT_BF16: torch.bfloat16,
        DataType.DT_FP32: torch.float32,
        DataType.DT_DOUBLE: torch.float64,
        DataType.DT_INT8: torch.int8,
        DataType.DT_UINT8: torch.uint8,
        DataType.DT_INT16: torch.int16,
        DataType.DT_UINT16: torch.uint16,
        DataType.DT_INT32: torch.int32,
        DataType.DT_UINT32: torch.uint32,
        DataType.DT_INT64: torch.int64,
        DataType.DT_UINT64: torch.uint64,
        DataType.DT_BOOL: torch.bool,
        DataType.DT_HF8: torch.uint8,
    }

    fp8_mappings = [
        (DataType.DT_FP8E4M3, 'float8_e4m3fn'),
        (DataType.DT_FP8E5M2, 'float8_e5m2'),
        (DataType.DT_FP8E8M0, 'float8_e8m0fnu'),
    ]

    for pto_dtype, attr_name in fp8_mappings:
        torch_attr = getattr(torch, attr_name, None)
        if torch_attr is not None:
            _torch_dtype_dict[pto_dtype] = torch_attr

    torch_dtype = _torch_dtype_dict.get(dtype)
    if torch_dtype is None:
        if dtype == DataType.DT_FP8E8M0:
            raise VerifyError(
                ValueError(
                    f"DataType.DT_FP8E8M0 requires 'torch.float8_e8m0fnu', which is NOT available "
                    f"in your current PyTorch version ({torch.__version__}).\n"
                    "Action: Please upgrade your torch-npu / PyTorch to a version supporting this specific FP8 format."
                )
            )
        elif dtype == DataType.DT_FP8E4M3:
            raise VerifyError(
                ValueError(
                    f"DataType.DT_FP8E4M3 requires 'torch.float8_e4m3fn', which is NOT available "
                    f"in your current PyTorch version ({torch.__version__}).\n"
                    "Action: Please upgrade your torch-npu / PyTorch to a version supporting this specific FP8 format."
                )
            )
        elif dtype == DataType.DT_FP8E5M2:
            raise VerifyError(
                ValueError(
                    f"DataType.DT_FP8E5M2 requires 'torch.float8_e5m2', which is NOT available "
                    f"in your current PyTorch version ({torch.__version__}).\n"
                    "Action: Please upgrade your torch-npu / PyTorch to a version supporting this specific FP8 format."
                )
            )
        else:
            raise VerifyError(ValueError(f"Input pypto.DataType is not supported or mapped to None. Got {dtype}"))
    return torch_dtype


def _gen_pto_tensor(input_tensors):
    import torch

    torch_tensors = []
    pto_tensors = []
    for t in input_tensors:
        torch_dtype = _torch_dtype_from(t.dtype)
        tshape = t.shape if all([isinstance(s, int) for s in t.shape]) else t.ori_shape
        torch_tensor = torch.zeros(tshape, dtype=torch_dtype)
        pto_tensor = Tensor(
            shape=tshape,
            dtype=t.dtype,
            name=t.name,
            data_ptr=torch_tensor.data_ptr(),
            format=t.format,
            device=torch_tensor.device,
            ori_shape=tshape,
        )

        torch_tensors.append(torch_tensor)
        pto_tensors.append(pto_tensor)
    return pto_tensors, torch_tensors
