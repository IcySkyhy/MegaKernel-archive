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
"""Unit tests for compact mode propagation through Block tile creation."""

from pypto_pro import DataType
from pypto_pro.ir.op.block_ops import make_tile_expr
import pypto_pro.language as pl
import pytest

from pypto.pypto_impl import ir as _ir
from pypto.pypto_impl.ir import CompactMode, MemorySpace


@pytest.mark.parametrize("name", ["null", "normal", "row_plus_one"])
def test_compact_mode_values_are_exposed(name):
    assert hasattr(CompactMode, name)


def test_hardware_info_compact_is_readable_and_writable():
    hw = _ir.HardwareInfo()
    assert hw.compact == CompactMode.null

    hw.compact = CompactMode.normal
    assert hw.compact == CompactMode.normal

    hw.compact = CompactMode.row_plus_one
    assert hw.compact == CompactMode.row_plus_one


@pytest.mark.parametrize(
    "compact",
    [None, 1, 2],
)
def test_tile_type_descriptor_stores_compact(compact):
    kwargs = {"compact": compact} if compact is not None else {}
    tt = pl.TileType(shape=[128, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec, **kwargs)
    assert tt.compact == compact


@pytest.mark.parametrize(
    "compact,expected",
    [
        (None, CompactMode.null),
        (1, CompactMode.normal),
        (2, CompactMode.row_plus_one),
    ],
)
def test_compact_stored_in_make_tile_type(compact, expected):
    kwargs = {"compact": compact} if compact is not None else {}
    call = make_tile_expr(
        shape=[128, 128],
        dtype=DataType.FP16,
        target_memory=MemorySpace.Left,
        addr=0,
        size=32768,
        layout=pl.ZZ,
        **kwargs,
    )

    assert isinstance(call.type.hardware_info, _ir.HardwareInfo)
    assert call.type.hardware_info.compact == expected


@pytest.mark.parametrize(
    "compact,expected",
    [
        (None, CompactMode.null),
        (1, CompactMode.normal),
    ],
)
def test_compact_propagated_by_parser(compact, expected):
    if compact is None:
        @pl.function
        def func(a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16]):
            tile_type = pl.TileType(
                shape=[128, 128],
                dtype=pl.DT_FP16,
                target_memory=pl.MemorySpace.Left,
                layout=pl.NZ,
            )
            tile_a = pl.make_tile(tile_type, addr=0x00000, size=32768)  # noqa: F841
    else:
        @pl.function
        def func(a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16]):
            tile_type = pl.TileType(
                shape=[128, 128],
                dtype=pl.DT_FP16,
                target_memory=pl.MemorySpace.Left,
                layout=pl.NZ,
                compact=compact,
            )
            tile_a = pl.make_tile(tile_type, addr=0x00000, size=32768)  # noqa: F841

    stmt = func.body.stmts[0] if hasattr(func.body, "stmts") else func.body
    tile_type = stmt.var.type
    assert isinstance(tile_type.hardware_info, _ir.HardwareInfo)
    assert tile_type.hardware_info.compact == expected
