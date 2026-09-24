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
"""Unit tests for codegen with dynamic shape tensor parameters."""

from pypto_pro import DataType, ir
import pypto_pro.language as pl
from pypto_pro.language.parser.diagnostics import ParserSyntaxError, ParserTypeError, UnsupportedFeatureError
import pytest


@pl.program
class AddKernelDynamic:
    """Add kernel with dynamic shape tensor parameters."""

    @pl.function(type=pl.FunctionType.InCore)
    def add_kernel(
        self,
        a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
        b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
        output: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    ):
        """Adds two tensors element-wise with dynamic shapes: result = a + b"""
        tile_type = pl.TileType(shape=[128, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        a_tile = pl.make_tile(tile_type, addr=0x0000, size=65536)
        b_tile = pl.make_tile(tile_type, addr=0x10000, size=65536)
        result = pl.make_tile(tile_type, addr=0x20000, size=65536)
        pl.load(a_tile, a, [0, 0])
        pl.load(b_tile, b, [0, 0])
        pl.add(result, a_tile, b_tile)
        out = pl.store(output, result, [0, 0])
        return out


@pl.program
class AddKernelValidShape:
    """Add kernel with static tensors but dynamic valid_shapes passed at runtime."""

    @pl.function(type=pl.FunctionType.InCore)
    def add_kernel(
        self,
        a: pl.Tensor[[128, 128], pl.DT_FP32],
        b: pl.Tensor[[128, 128], pl.DT_FP32],
        output: pl.Tensor[[128, 128], pl.DT_FP32],
        m_var: pl.DT_INT64,
        n_var: pl.DT_INT64,
    ) -> pl.Tensor[[128, 128], pl.DT_FP32]:
        """Loads 128x128 tiles but marks only [M, N] as valid: result = a + b"""
        tile_type = pl.TileType(
            shape=[128, 128],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Vec,
            valid_shape=[-1, -1],
        )
        a_tile = pl.make_tile(tile_type, addr=0x0000, size=65536)
        b_tile = pl.make_tile(tile_type, addr=0x10000, size=65536)
        result = pl.make_tile(tile_type, addr=0x20000, size=65536)
        pl.set_validshape(a_tile, [m_var, n_var])
        pl.set_validshape(b_tile, [m_var, n_var])
        pl.set_validshape(result, [m_var, n_var])
        pl.load(a_tile, a, [0, 0])
        pl.load(b_tile, b, [0, 0])
        pl.add(result, a_tile, b_tile)
        pl.set_validshape(a_tile, [m_var, n_var])
        pl.set_validshape(b_tile, [m_var, n_var])
        pl.set_validshape(result, [m_var, n_var])
        out = pl.store(output, result, [0, 0])
        return out


@pl.program
class AddKernelShapeSubscript:
    """Add kernel using tensor.shape[i] for a dynamic loop bound."""

    @pl.function(type=pl.FunctionType.InCore)
    def add_kernel(
        self,
        a: pl.Tensor[[pl.DYNAMIC, 128], pl.DT_FP32],
        b: pl.Tensor[[pl.DYNAMIC, 128], pl.DT_FP32],
        output: pl.Tensor[[pl.DYNAMIC, 128], pl.DT_FP32],
    ):
        """Uses the canonical dynamic dimension stored in a TensorType."""
        m_var = a.shape[0]
        tile_type = pl.TileType(shape=[2, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        a_tile = pl.make_tile(tile_type, addr=0x0000, size=1024)
        b_tile = pl.make_tile(tile_type, addr=0x0400, size=1024)
        result = pl.make_tile(tile_type, addr=0x0800, size=1024)
        for i in pl.range(0, m_var, 2):
            offset_1 = i * 2
            pl.load(a_tile, a, [offset_1, 0])
            pl.load(b_tile, b, [offset_1, 0])
            pl.add(result, a_tile, b_tile)
            out = pl.store(output, result, [offset_1, 0])
        return out


@pl.program
class StaticShapeNegativeSubscript:
    """Exercise Python-style negative indexing for a fixed tensor shape."""

    @pl.function(type=pl.FunctionType.InCore)
    def shape_kernel(
        self,
        a: pl.Tensor[[2, 128], pl.DT_FP32],
        output: pl.Tensor[[2, 128], pl.DT_FP32],
    ) -> pl.Tensor[[2, 128], pl.DT_FP32]:
        n = a.shape[-1]
        for _ in pl.range(0, n, 1):
            out = output
        return out


@pl.program
class DynamicShapeTupleUnpack:
    """Exercise complete tensor.shape tuple access and repeated discard targets."""

    @pl.function(type=pl.FunctionType.InCore)
    def shape_kernel(
        self,
        a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, 128, 64], pl.DT_FP32],
    ):
        m, n, _, _ = a.shape
        for _ in pl.range(0, m, 1):
            out = a
        return out


@pl.program
class CompileTimeShapeAxis:
    """Exercise a compile-time integer expression as a shape index."""

    @pl.function(type=pl.FunctionType.InCore)
    def shape_kernel(
        self,
        a: pl.Tensor[[2, 128], pl.DT_FP32],
        output: pl.Tensor[[2, 128], pl.DT_FP32],
    ) -> pl.Tensor[[2, 128], pl.DT_FP32]:
        n = a.shape[0]
        for _ in pl.range(0, n, 1):
            out = output
        return out


def test_dynamic_shape_subscript_uses_parameter_shape_var():
    func = AddKernelShapeSubscript.functions["add_kernel"]
    dynamic_dim = func.params[0].type.shape[0]

    assert isinstance(dynamic_dim, ir.Var)
    assert dynamic_dim.name == "__pypto_dyn_a_0"
    assert dynamic_dim.type.dtype == DataType.INDEX
    assert dynamic_dim.name in ir.python_print(AddKernelShapeSubscript)


def test_negative_shape_subscript_uses_fixed_dimension():
    func = StaticShapeNegativeSubscript.functions["shape_kernel"]

    assert isinstance(func.params[0].type.shape[-1], ir.ConstInt)
    assert func.params[0].type.shape[-1].value == 128


def test_shape_tuple_unpack_uses_canonical_tensor_dimensions():
    func = DynamicShapeTupleUnpack.functions["shape_kernel"]

    assert isinstance(func.params[0].type.shape[0], ir.Var)
    assert func.params[0].type.shape[0].name == "__pypto_dyn_a_0"


def test_shape_tuple_unpack_rejects_arity_mismatch():
    code = """
@pl.function
def shape_unpack(x: pl.Tensor[[2, 128], pl.DT_FP32]) -> pl.Tensor[[2, 128], pl.DT_FP32]:
    m, n, k = x.shape
    return x
"""
    with pytest.raises(ParserTypeError, match="unpack"):
        pl.parse(code)


@pytest.mark.parametrize("index", ["True", "0.0", "2", "-3"])
def test_shape_subscript_rejects_invalid_constant_index(index):
    code = f"""
@pl.function
def shape_index(x: pl.Tensor[[2, 128], pl.DT_FP32]) -> pl.Tensor[[2, 128], pl.DT_FP32]:
    n = x.shape[{index}]
    return x
"""
    with pytest.raises((ParserSyntaxError, ParserTypeError)):
        pl.parse(code)


def test_shape_subscript_rejects_non_tensor_base():
    code = """
@pl.function
def shape_index(x: pl.DT_INT64) -> pl.DT_INT64:
    return x.shape[0]
"""
    with pytest.raises((ParserSyntaxError, ParserTypeError, UnsupportedFeatureError)):
        pl.parse(code)


def test_shape_subscript_rejects_runtime_axis():
    code = """
@pl.function
def shape_index(
    x: pl.Tensor[[2, 128], pl.DT_FP32],
    axis: pl.DT_INT64,
) -> pl.DT_INT64:
    return x.shape[axis]
"""
    with pytest.raises(ParserSyntaxError, match="tensor.shape index must be a compile-time integer"):
        pl.parse(code)
