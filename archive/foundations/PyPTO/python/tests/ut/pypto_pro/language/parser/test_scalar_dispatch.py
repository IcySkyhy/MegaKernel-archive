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
"""Tests for scalar operation dispatch in the DSL parser.

Verifies that pl.min, pl.max dispatch to scalar IR ops
when called with scalar arguments.
"""

import pypto_pro.language as pl

from pypto.pypto_impl import ir


def test_scalar_min():
    """Test pl.min(scalar, scalar) dispatches to ir.min_."""

    @pl.function(type=pl.FunctionType.Opaque)
    def test_min(
        config: pl.Tensor[[2], pl.DT_INT64],
        out: pl.Tensor[[2, 16, 128], pl.DT_FP32],
    ) -> pl.Tensor[[2, 16, 128], pl.DT_FP32]:
        a: pl.DT_UINT64 = pl.getval(config, 0)
        b: pl.DT_UINT64 = pl.getval(config, 1)
        c = pl.min(a, b)
        _ = c + 1
        return out

    assert isinstance(test_min, ir.Function)
    ir_text = ir.python_print(test_min)
    assert "min" in ir_text.lower()


def test_scalar_max():
    """Test pl.max(scalar, scalar) dispatches to ir.max_."""

    @pl.function(type=pl.FunctionType.Opaque)
    def test_max(
        config: pl.Tensor[[2], pl.DT_INT64],
        out: pl.Tensor[[2, 16, 128], pl.DT_FP32],
    ) -> pl.Tensor[[2, 16, 128], pl.DT_FP32]:
        a: pl.DT_UINT64 = pl.getval(config, 0)
        b: pl.DT_UINT64 = pl.getval(config, 1)
        c = pl.max(a, b)
        _ = c + 1
        return out

    assert isinstance(test_max, ir.Function)
    ir_text = ir.python_print(test_max)
    assert "max" in ir_text.lower()


def test_scalar_min_with_literal():
    """Test pl.min(scalar, int_literal) -the paged_attention use case."""

    @pl.function(type=pl.FunctionType.Opaque)
    def test_min_lit(
        config: pl.Tensor[[2], pl.DT_INT64],
        out: pl.Tensor[[2, 16, 128], pl.DT_FP32],
    ) -> pl.Tensor[[2, 16, 128], pl.DT_FP32]:
        a: pl.DT_UINT64 = pl.getval(config, 0)
        c = pl.min(a, 128)
        _ = c + 1
        return out

    assert isinstance(test_min_lit, ir.Function)
    ir_text = ir.python_print(test_min_lit)
    assert "min" in ir_text.lower()


def test_tile_min_still_works():
    """Ensure pl.min(tile, axis=...) still works as tile reduction."""

    @pl.function
    def test_tile_min(
        x: pl.Tensor[[32, 32], pl.DT_FP32],
    ) -> pl.Tensor[[32, 32], pl.DT_FP32]:
        tile_type = pl.TileType(shape=[32, 32], dtype=pl.DT_FP32)
        reduced_type = pl.TileType(shape=[32, 1], dtype=pl.DT_FP32)
        tile_a = pl.make_tile(tile_type, addr=0, size=4096)
        tmp = pl.make_tile(tile_type, addr=4096, size=4096)
        tile_c = pl.make_tile(reduced_type, addr=8192, size=128)
        pl.load(tile_a, x, [0, 0])
        pl.minimum(tile_a, tmp, tile_c, dim=0)
        out: pl.Tensor[[32, 32], pl.DT_FP32] = pl.store(x, tile_c, [0, 0])
        return out

    assert isinstance(test_tile_min, ir.Function)
