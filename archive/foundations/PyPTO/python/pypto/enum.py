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

import enum

from . import pypto_impl


def _enum_repr(self):
    return f"{self.__class__.__name__}.{self.name}"  # remove pypto_impl. prefix


DataType = pypto_impl.DataType
TileOpFormat = pypto_impl.TileOpFormat
CachePolicy = pypto_impl.CachePolicy
ReduceMode = pypto_impl.ReduceMode
CastMode = pypto_impl.CastMode
OpType = pypto_impl.OpType
OutType = pypto_impl.OutType
TopKAlgo = pypto_impl.TopKAlgo
ReLuType = pypto_impl.ReLuType
TransMode = pypto_impl.TransMode
ScatterMode = pypto_impl.ScatterMode
SaturationMode = pypto_impl.SaturationMode
AtomicRMWMode = pypto_impl.AtomicRMWMode
AtomicType = pypto_impl.AtomicType
DequantScaleRoundingMode = pypto_impl.DequantScaleRoundingMode
PrecisionType = pypto_impl.PrecisionType

_ENUM_CLASSES = [
    DataType,
    TileOpFormat,
    CachePolicy,
    ReduceMode,
    CastMode,
    OpType,
    OutType,
    TopKAlgo,
    SaturationMode,
    DequantScaleRoundingMode,
]
for cls in _ENUM_CLASSES:
    cls.__repr__ = _enum_repr
    cls.__str__ = _enum_repr

DT_INT4 = DataType.DT_INT4
DT_INT8 = DataType.DT_INT8
DT_INT16 = DataType.DT_INT16
DT_INT32 = DataType.DT_INT32
DT_INT64 = DataType.DT_INT64
DT_FP8 = DataType.DT_FP8
DT_FP16 = DataType.DT_FP16
DT_FP32 = DataType.DT_FP32
DT_BF16 = DataType.DT_BF16
DT_HF4 = DataType.DT_HF4
DT_HF8 = DataType.DT_HF8
DT_FP8E4M3 = DataType.DT_FP8E4M3
DT_FP8E5M2 = DataType.DT_FP8E5M2
DT_FP8E8M0 = DataType.DT_FP8E8M0
DT_FP4_E2M1 = DataType.DT_FP4_E2M1
DT_FP4_E2M1X2 = DataType.DT_FP4_E2M1X2
DT_FP4_E1M2 = DataType.DT_FP4_E1M2
DT_FP4_E1M2X2 = DataType.DT_FP4_E1M2X2

DT_UINT8 = DataType.DT_UINT8
DT_UINT16 = DataType.DT_UINT16
DT_UINT32 = DataType.DT_UINT32
DT_UINT64 = DataType.DT_UINT64
DT_BOOL = DataType.DT_BOOL
DT_DOUBLE = DataType.DT_DOUBLE
DT_BOTTOM = DataType.DT_BOTTOM
ROUND_UP = DequantScaleRoundingMode.ROUND_UP
ROUND_DOWN = DequantScaleRoundingMode.ROUND_DOWN


class StatusType(enum.Enum):
    DYN = "DYN"
    DYNAMIC = "DYNAMIC"
    STATIC = "STATIC"


DYN = StatusType.DYN
DYNAMIC = StatusType.DYNAMIC
STATIC = StatusType.STATIC
