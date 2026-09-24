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

import functools
import sys
from typing import List, Sequence, Union

from . import pypto_impl
from .enum import DataType
from .error import FeError
from .symbolic_scalar import SymbolicScalar, SymInt

_torch_npu = None
_torch_npu_checked = False
_dtensor_type = None
_dtensor_type_checked = False


def get_torch_npu():
    """Return the torch_npu module if available, otherwise None."""
    global _torch_npu, _torch_npu_checked
    if not _torch_npu_checked:
        try:
            import torch_npu

            _torch_npu = torch_npu
        except ImportError:
            pass
        _torch_npu_checked = True
    return _torch_npu


def get_dtensor_type():
    """Return torch.distributed._tensor.DTensor type if available, otherwise None."""
    global _dtensor_type, _dtensor_type_checked
    if not _dtensor_type_checked:
        try:
            from torch.distributed._tensor import DTensor

            _dtensor_type = DTensor
        except ImportError:
            pass
        _dtensor_type_checked = True
    return _dtensor_type


def get_npu_tensor_format(tensor):
    """Return 'NZ' if tensor is in NPU NZ format (format code 29), else 'ND'."""
    torch_npu = get_torch_npu()
    if torch_npu is not None and torch_npu.get_npu_format(tensor) == 29:
        return "NZ"
    return "ND"


def to_sym(value) -> pypto_impl.SymbolicScalar:
    if isinstance(value, int):
        return pypto_impl.SymbolicScalar(value)
    if isinstance(value, pypto_impl.SymbolicScalar):
        return value
    raise FeError(ValueError("Invalid value type"))


def to_syms(value: Union[Sequence[int], Sequence[SymbolicScalar]]) -> List[pypto_impl.SymbolicScalar]:
    return [to_sym(v) for v in value]


def ceildiv(a: SymInt, b: SymInt) -> SymInt:
    return (a + b - 1) // b


# only outer takes effect void avoid tensor.py hide source_location of user code
_source_location_depth = 0


def set_source_location(level: int = 1, filename=None, lineno=None):
    global _source_location_depth
    if _source_location_depth == 0:
        if filename is None:
            frame = sys._getframe(level + 1)
            filename = frame.f_code.co_filename
            lineno = frame.f_lineno
        pypto_impl.SetSpan(filename, lineno)
    _source_location_depth += 1


def clear_source_location():
    global _source_location_depth
    _source_location_depth -= 1
    if _source_location_depth == 0:
        pypto_impl.ClearSpan()


def source_location(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        set_source_location()
        out = func(*args, **kwargs)
        clear_source_location()
        return out

    return wrapper


def bytes_of(dtype: DataType) -> int:
    """return the number of bytes of the current datatype

    Parameters
    ----------
    dtype: pypto.DataType
        datatype to be determined the number of bytes

    Returns
    -------
    int: the size of bytes the datatype contains

    Examples
    --------
    >>> print(pypto.bytes_of(pypto.DT_FP32))
        4
    """
    # implementation
    return pypto_impl.BytesOf(dtype)
