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
"""Block-facing smoke tests for tuple values and subscript expressions.

The exhaustive TupleType/GetItemExpr behavior is covered in the generic IR
tests. Block keeps coverage only for tuple returns that are visible in the
pypto_pro DSL.
"""

from pypto_pro import DataType, ir


def _idx(value: int):
    return ir.ConstInt(value, DataType.INDEX, ir.Span.unknown())


def test_tuple_get_item_preserves_block_return_types():
    span = ir.Span.unknown()
    tuple_type = ir.TupleType([ir.ScalarType(DataType.INT64), ir.TensorType([_idx(16)], DataType.FP16)])
    tuple_var = ir.Var("ret", tuple_type, span)

    first = ir.GetItemExpr(tuple_var, _idx(0), span)
    second = ir.GetItemExpr(tuple_var, _idx(1), span)

    assert isinstance(first.type, ir.ScalarType)
    assert first.type.dtype == DataType.INT64
    assert isinstance(second.type, ir.TensorType)
    assert second.type.dtype == DataType.FP16


def test_tuple_get_item_preserves_block_tile_type():
    span = ir.Span.unknown()
    tile_type = ir.TileType([_idx(16), _idx(16)], DataType.FP16)
    tuple_type = ir.TupleType([tile_type])
    tuple_var = ir.Var("tiles", tuple_type, span)

    item = ir.GetItemExpr(tuple_var, _idx(0), span)

    assert isinstance(item.type, ir.TileType)
    assert item.type.dtype == tile_type.dtype


def test_make_tuple_supports_block_parser_roundtrip_shape():
    span = ir.Span.unknown()
    lhs = ir.MakeTuple([ir.ConstInt(1, DataType.INDEX, span), ir.ConstInt(2, DataType.INDEX, span)], span)
    rhs = ir.MakeTuple([ir.ConstInt(1, DataType.INDEX, span), ir.ConstInt(2, DataType.INDEX, span)], span)

    ir.assert_structural_equal(lhs, rhs)
