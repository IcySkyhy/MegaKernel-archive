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
"""Comprehensive tests for MemRef, MemorySpace, and TileView."""

from pypto_pro import DataType, ir
from pypto_pro.ir import IRBuilder


def test_builder_memref():
    """Test IRBuilder.memref() helper."""
    ib = IRBuilder()

    # Create memref with int address
    memref = ib.memref(ir.MemorySpace.DDR, 0x1000, 1024, 33)

    assert isinstance(memref, ir.MemRef)
    assert memref.memory_space_ == ir.MemorySpace.DDR
    assert memref.size_ == 1024


def test_builder_tile_view():
    """Test IRBuilder.tile_view() helper."""
    ib = IRBuilder()

    # Create tile view with integer dimensions
    tv = ib.tile_view([16, 16], [1, 16], 0)

    assert isinstance(tv, ir.TileView)
    assert len(tv.valid_shape) == 2
    assert len(tv.stride) == 2


def test_builder_tensor_type():
    """Test IRBuilder.tensor_type() helper."""
    ib = IRBuilder()

    # Simple tensor type
    tensor_t = ib.tensor_type([64, 128], DataType.FP32)

    assert isinstance(tensor_t, ir.TensorType)
    assert len(tensor_t.shape) == 2
    assert tensor_t.dtype == DataType.FP32
    assert tensor_t.memref is None


def test_builder_tensor_type_with_memref():
    """Test IRBuilder.tensor_type() with memref."""
    ib = IRBuilder()

    # Create memref
    memref = ib.memref(ir.MemorySpace.DDR, 0x1000, 1024, 34)

    # Tensor type with memref
    tensor_t = ib.tensor_type([64, 128], DataType.FP32, memref=memref)

    assert isinstance(tensor_t, ir.TensorType)
    assert tensor_t.memref is not None
    assert tensor_t.memref.memory_space_ == ir.MemorySpace.DDR


def test_builder_tile_type():
    """Test IRBuilder.tile_type() helper."""
    ib = IRBuilder()

    # Simple tile type
    tile_t = ib.tile_type([16, 16], DataType.FP16)

    assert isinstance(tile_t, ir.TileType)
    assert len(tile_t.shape) == 2
    assert tile_t.dtype == DataType.FP16


def test_builder_tile_type_with_memref_and_tileview():
    """Test IRBuilder.tile_type() with memref and tile_view."""
    ib = IRBuilder()

    # Create memref and tile view
    memref = ib.memref(ir.MemorySpace.Left, 0, 512, 35)
    tv = ib.tile_view([16, 16], [1, 16], 0)

    # Tile type with memref and tile_view
    tile_t = ib.tile_type([16, 16], DataType.FP16, memref=memref, tile_view=tv)

    assert isinstance(tile_t, ir.TileType)
    assert tile_t.memref is not None
    assert tile_t.tile_view is not None
    assert tile_t.memref.memory_space_ == ir.MemorySpace.Left


def test_builder_round_trip():
    """Test round-trip: create with builder, print to Python syntax."""
    ib = IRBuilder()

    # Create complex tile type with builder
    memref = ib.memref(ir.MemorySpace.Right, 0x200, 1024, 36)
    tv = ib.tile_view([32, 32], [1, 32], 0)
    tile_t = ib.tile_type([32, 32], DataType.FP32, memref=memref, tile_view=tv)

    # Print to Python syntax
    printed = ir.python_print(tile_t)

    # Verify output contains all expected elements
    assert "ir.TileType" in printed
    assert "ir.TileType([32, 32], ir.FP32," in printed
    assert "ir.FP32" in printed
    # MemRef prints as positional arg with full constructor syntax (no keyword)
    assert "memref=" not in printed
    assert "ir.MemRef" in printed
    assert "ir.MemorySpace.Right" in printed
    assert "tile_view=ir.TileView" in printed
