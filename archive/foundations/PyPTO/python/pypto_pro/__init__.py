# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
PyPTO - Python Tensor Operations Library

This package provides Python bindings for the PyPTO C++ library.
"""

__all__ = [
    "codegen",
    "ir",
    "language",
    "InternalError",
    "DataType",
    "DT_BOOL",
    "DT_INT4",
    "DT_INT8",
    "DT_INT16",
    "DT_INT32",
    "DT_INT64",
    "DT_UINT4",
    "DT_UINT8",
    "DT_UINT16",
    "DT_UINT32",
    "DT_UINT64",
    "DT_FP4",
    "DT_FP8E4M3FN",
    "DT_FP8E5M2",
    "DT_FP8E8M0",
    "DT_FP4E2M1",
    "DT_FP4E1M2",
    "DT_FP16",
    "DT_FP32",
    "DT_BF16",
    "DT_HF4",
    "DT_HF8",
]

from ._bootstrap import DataType, InternalError, codegen # noqa: I001
from . import ir as ir
from . import language as language

DT_BF16 = DataType.BF16
DT_BOOL = DataType.BOOL
DT_FP4 = DataType.FP4
DT_FP8E4M3FN = DataType.FP8E4M3FN
DT_FP8E5M2 = DataType.FP8E5M2
DT_FP8E8M0 = DataType.FP8E8M0
DT_FP4E2M1 = DataType.FP4E2M1
DT_FP4E1M2 = DataType.FP4E1M2
DT_FP16 = DataType.FP16
DT_FP32 = DataType.FP32
DT_HF4 = DataType.HF4
DT_HF8 = DataType.HF8
DT_INT4 = DataType.INT4
DT_INT8 = DataType.INT8
DT_INT16 = DataType.INT16
DT_INT32 = DataType.INT32
DT_INT64 = DataType.INT64
DT_UINT4 = DataType.UINT4
DT_UINT8 = DataType.UINT8
DT_UINT16 = DataType.UINT16
DT_UINT32 = DataType.UINT32
DT_UINT64 = DataType.UINT64
