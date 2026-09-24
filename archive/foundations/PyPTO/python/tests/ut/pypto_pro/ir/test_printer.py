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
"""Block-specific Python printer smoke tests.

The C++ IR tests own broad printer coverage. The block suite keeps only the
printer contracts that are directly consumed by pypto_pro Block parser/codegen
round trips.
"""

from pypto_pro import DataType, ir


def test_python_print_block_tensor_and_tile_types():
    span = ir.Span.unknown()
    tensor_type = ir.TensorType(
        [ir.ConstInt(64, DataType.INDEX, span), ir.ConstInt(128, DataType.INDEX, span)],
        DataType.FP16,
    )
    memref = ir.MemRef(ir.MemorySpace.Vec, ir.ConstInt(0, DataType.INDEX, span), 512, 0)
    tile_type = ir.TileType(
        [ir.ConstInt(16, DataType.INDEX, span), ir.ConstInt(16, DataType.INDEX, span)],
        DataType.FP16,
        memref,
    )

    assert ir.python_print_type(tensor_type) == "ir.Tensor[[64, 128], ir.FP16]"
    printed_tile = ir.python_print_type(tile_type)
    assert "ir.TileType([16, 16], ir.FP16" in printed_tile
    assert "ir.MemRef(ir.MemorySpace.Vec" in printed_tile


def test_python_print_program_preserves_block_function_type():
    span = ir.Span.unknown()
    x = ir.Var("x", ir.ScalarType(DataType.INT64), span)
    fn = ir.Function(
        "f",
        [x],
        [ir.ScalarType(DataType.INT64)],
        ir.YieldStmt([x], span),
        span,
        type=ir.FunctionType.InCore,
    )
    program = ir.Program([fn], "BlockProgram", span)

    result = ir.python_print(program)

    assert "# ir.program: BlockProgram" in result
    assert "@ir.function(type=ir.FunctionType.InCore)" in result
    assert "def f" in result


def test_python_print_block_load_store_ops():
    span = ir.Span.unknown()
    dim = ir.ConstInt(64, DataType.INDEX, span)
    tensor = ir.Var("x", ir.TensorType([dim], DataType.FP16), span)
    memref = ir.MemRef(ir.MemorySpace.Vec, ir.ConstInt(0, DataType.INDEX, span), 128, 0)
    tile = ir.Var("tile", ir.TileType([dim], DataType.FP16, memref), span)
    zero = ir.ConstInt(0, DataType.INDEX, span)

    load = ir.Call(ir.Op("block.load"), [tile, tensor, ir.MakeTuple([zero], span)], span)
    store = ir.Call(ir.Op("block.store"), [tensor, tile, ir.MakeTuple([zero], span)], span)
    body = ir.SeqStmts([ir.EvalStmt(load, span), ir.EvalStmt(store, span)], span)

    result = ir.python_print(body)

    assert "block.load" in result
    assert "block.store" in result
