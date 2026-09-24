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


def test_print_tensor_with_memref():
    """Test Python printing of TensorType with MemRef."""
    span = ir.Span.unknown()
    shape = [ir.ConstInt(10, DataType.INT64, span)]

    memref = ir.MemRef(ir.MemorySpace.DDR, ir.ConstInt(0, DataType.INT64, span), 40, 25)

    tensor_type = ir.TensorType(shape, DataType.FP32, memref)
    tensor_var = ir.Var("tensor", tensor_type, span)
    stmt = ir.AssignStmt(tensor_var, ir.ConstInt(0, DataType.INT64, span), span)

    result = ir.python_print(stmt)
    # Just verify it doesn't crash and produces output
    assert result is not None
    assert len(result) > 0


def test_tile_type_with_memref_and_tileview_print():
    """Test printing TileType with MemRef variable name and TileView."""
    span = ir.Span.unknown()
    shape = [
        ir.ConstInt(16, DataType.INT64, span),
        ir.ConstInt(16, DataType.INT64, span),
    ]

    addr = ir.ConstInt(0x2000, DataType.INT64, span)
    memref = ir.MemRef(ir.MemorySpace.Left, addr, 512, 8)

    valid_shape = [
        ir.ConstInt(16, DataType.INT64, span),
        ir.ConstInt(16, DataType.INT64, span),
    ]
    stride = [
        ir.ConstInt(1, DataType.INT64, span),
        ir.ConstInt(16, DataType.INT64, span),
    ]
    start_offset = ir.ConstInt(0, DataType.INT64, span)
    tv = ir.TileView(valid_shape, stride, start_offset)

    tile_type = ir.TileType(shape, DataType.FP16, memref, tv)
    printed = ir.python_print(tile_type)

    assert "ir.TileType" in printed
    assert "ir.FP16" in printed
    # MemRef prints as positional arg with full constructor syntax (fixes #281)
    assert "memref=" not in printed
    assert "ir.MemRef" in printed
    assert "ir.MemorySpace.Left" in printed
    assert "8192" in printed  # 0x2000 in decimal
    assert "512" in printed  # size
    assert "tile_view=" in printed
    assert "ir.TileView" in printed
    assert "valid_shape=" in printed
    assert "stride=" in printed
    assert "start_offset=" in printed


def test_tensor_type_with_memref_and_tensorview_print():
    """Test printing TensorType with both MemRef and TensorView."""
    span = ir.Span.unknown()
    shape = [
        ir.ConstInt(64, DataType.INT64, span),
        ir.ConstInt(64, DataType.INT64, span),
    ]
    stride = [
        ir.ConstInt(1, DataType.INT64, span),
        ir.ConstInt(64, DataType.INT64, span),
    ]

    addr = ir.ConstInt(0x5000, DataType.INT64, span)
    memref = ir.MemRef(ir.MemorySpace.Left, addr, 4096, 42)
    tensor_view = ir.TensorView(stride, ir.TensorLayout.NZ)
    tensor_type = ir.TensorType(shape, DataType.FP16, memref=memref, tensor_view=tensor_view)

    printed = ir.python_print(tensor_type)

    assert "ir.Tensor" in printed
    assert "ir.FP16" in printed
    # MemRef prints as positional (no keyword), tensor_view as keyword
    assert "memref=" not in printed
    assert "ir.MemRef" in printed
    assert "tensor_view=" in printed
    assert "ir.TensorView" in printed
    assert "ir.TensorLayout.NZ" in printed
