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

import pypto


def test_dtype():
    # Make sure all data types are defined
    assert pypto.bytes_of(pypto.DT_INT4) == 1
    assert pypto.bytes_of(pypto.DT_INT8) == 1
    assert pypto.bytes_of(pypto.DT_INT16) == 2
    assert pypto.bytes_of(pypto.DT_INT32) == 4
    assert pypto.bytes_of(pypto.DT_INT64) == 8
    assert pypto.bytes_of(pypto.DT_FP8) == 1
    assert pypto.bytes_of(pypto.DT_FP16) == 2
    assert pypto.bytes_of(pypto.DT_FP32) == 4
    assert pypto.bytes_of(pypto.DT_BF16) == 2
    assert pypto.bytes_of(pypto.DT_HF4) == 1
    assert pypto.bytes_of(pypto.DT_HF8) == 1
    assert pypto.bytes_of(pypto.DT_FP8E4M3) == 1
    assert pypto.bytes_of(pypto.DT_FP8E5M2) == 1
    assert pypto.bytes_of(pypto.DT_FP8E8M0) == 1
    assert pypto.bytes_of(pypto.DT_UINT8) == 1
    assert pypto.bytes_of(pypto.DT_UINT16) == 2
    assert pypto.bytes_of(pypto.DT_UINT32) == 4
    assert pypto.bytes_of(pypto.DT_UINT64) == 8
    assert pypto.bytes_of(pypto.DT_BOOL) == 1
    assert pypto.bytes_of(pypto.DT_DOUBLE) == 8

    assert str(pypto.DT_INT4) == "DataType.DT_INT4"


def test_tile_op_format():
    assert str(pypto.TileOpFormat.TILEOP_ND) == "TileOpFormat.TILEOP_ND"
    assert str(pypto.TileOpFormat.TILEOP_NZ) == "TileOpFormat.TILEOP_NZ"


def test_cache_policy():
    assert str(pypto.CachePolicy.NONE_CACHEABLE) == "CachePolicy.NONE_CACHEABLE"


def test_reduce_mode():
    assert str(pypto.ReduceMode.ATOMIC_ADD) == "ReduceMode.ATOMIC_ADD"


def test_cast_mode():
    assert str(pypto.CastMode.CAST_RINT) == "CastMode.CAST_RINT"
    assert str(pypto.CastMode.CAST_ROUND) == "CastMode.CAST_ROUND"
    assert str(pypto.CastMode.CAST_FLOOR) == "CastMode.CAST_FLOOR"
    assert str(pypto.CastMode.CAST_CEIL) == "CastMode.CAST_CEIL"
    assert str(pypto.CastMode.CAST_TRUNC) == "CastMode.CAST_TRUNC"
    assert str(pypto.CastMode.CAST_ODD) == "CastMode.CAST_ODD"


def test_op_type():
    assert str(pypto.OpType.EQ) == "OpType.EQ"
    assert str(pypto.OpType.NE) == "OpType.NE"
    assert str(pypto.OpType.LT) == "OpType.LT"
    assert str(pypto.OpType.LE) == "OpType.LE"
    assert str(pypto.OpType.GT) == "OpType.GT"
    assert str(pypto.OpType.GE) == "OpType.GE"


def test_out_type():
    assert str(pypto.OutType.BOOL) == "OutType.BOOL"
    assert str(pypto.OutType.BIT) == "OutType.BIT"


def test_topk_algo():
    assert str(pypto.TopKAlgo.MERGE_SORT) == "TopKAlgo.MERGE_SORT"
    assert str(pypto.TopKAlgo.RADIX_SELECT) == "TopKAlgo.RADIX_SELECT"


def test_atomic_rmw_mode():
    assert int(pypto.AtomicRMWMode.ADD) == 0
    assert int(pypto.AtomicRMWMode.MAX) == 1
    assert int(pypto.AtomicRMWMode.MIN) == 2
    assert pypto.AtomicRMWMode.ADD.name == "ADD"
    assert pypto.AtomicRMWMode.ADD.value == 0
    assert pypto.AtomicRMWMode.ADD == pypto.AtomicRMWMode.ADD


def test_precision_type():
    assert int(pypto.PrecisionType.INTRINSIC) == 0
    assert int(pypto.PrecisionType.HIGH_PRECISION) == 1
    assert pypto.PrecisionType.INTRINSIC.name == "INTRINSIC"
    assert pypto.PrecisionType.HIGH_PRECISION.value == 1


def test_external_error():
    assert int(pypto.pypto_impl.ExternalError.COMMON_EXTERNAL_ERROR) == 0x0FFFF
    assert int(pypto.pypto_impl.ExternalError.BAD_FD) == 0x00009
    assert int(pypto.pypto_impl.ExternalError.DYNAMIC_SHAPE_COMPUTE_UNSUPPORTED) == 0x0000A
    assert pypto.pypto_impl.ExternalError.UNKNOWN == pypto.pypto_impl.ExternalError.COMMON_EXTERNAL_ERROR
    assert pypto.pypto_impl.ExternalError.UNKNOWN.value == pypto.pypto_impl.ExternalError.COMMON_EXTERNAL_ERROR.value
    assert pypto.pypto_impl.ExternalError.INVALID_TYPE.value == 1
