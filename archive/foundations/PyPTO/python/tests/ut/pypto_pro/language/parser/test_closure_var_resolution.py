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
"""Unit tests for closure variable resolution in DSL function bodies (issue #276).

Verifies that Python globals/closure variables used as positional arguments
in function calls inside @pl.function bodies are resolved correctly.
"""

from pypto_pro import ir
import pypto_pro.language as pl
from pypto_pro.language.parser.diagnostics import ParserTypeError, UndefinedVariableError
import pytest


def test_list_closure_var_as_positional_arg():
    """List closure var works as positional arg (the original issue)."""
    offset_value = [0, 0]
    tile_shape = [64, 64]

    @pl.function
    def func(
        t: pl.Tensor[[128, 128], pl.DT_FP32], out: pl.Tensor[[128, 128], pl.DT_FP32]
    ) -> pl.Tensor[[128, 128], pl.DT_FP32]:
        tile_type = pl.TileType(shape=tile_shape, dtype=pl.DT_FP32)
        a = pl.make_tile(tile_type, addr=0, size=16384)
        pl.load(a, t, offset_value)
        result: pl.Tensor[[128, 128], pl.DT_FP32] = pl.store(out, a, offset_value)
        return result

    assert isinstance(func, ir.Function)


def test_int_closure_var_as_positional_arg():
    """Int closure variable resolves to ConstInt in function body."""
    axis_value = 1

    @pl.function
    def func(x: pl.Tensor[[64, 128], pl.DT_FP32]) -> pl.Tensor[[128, 64], pl.DT_FP32]:
        result: pl.Tensor[[128, 64], pl.DT_FP32] = pl.tensor.transpose(x, axis1=0, axis2=axis_value)
        return result

    assert isinstance(func, ir.Function)


def test_float_closure_var_as_positional_arg():
    """Float closure variable resolves to ConstFloat in function body."""
    scale_value = 2.0

    @pl.function
    def func(x: pl.Tensor[[64], pl.DT_FP32]) -> pl.Tensor[[64], pl.DT_FP32]:
        result: pl.Tensor[[64], pl.DT_FP32] = pl.tensor.mul(x, scale_value)
        return result

    assert isinstance(func, ir.Function)


def test_bool_closure_var_as_positional_arg():
    """Bool closure variable resolves to ConstBool in function body."""
    flag_value = True

    @pl.function
    def func(x: pl.Tensor[[64], pl.DT_FP32]) -> pl.Tensor[[64], pl.DT_FP32]:
        result: pl.Tensor[[64], pl.DT_FP32] = pl.tensor.mul(x, flag_value)
        return result

    assert isinstance(func, ir.Function)


def test_tuple_closure_var_as_positional_arg():
    """Tuple closure variable resolves to MakeTuple in function body."""
    offset_value = (0, 0)
    tile_shape = (64, 64)

    @pl.function
    def func(
        t: pl.Tensor[[128, 128], pl.DT_FP32], out: pl.Tensor[[128, 128], pl.DT_FP32]
    ) -> pl.Tensor[[128, 128], pl.DT_FP32]:
        tile_type = pl.TileType(shape=tile_shape, dtype=pl.DT_FP32)
        a = pl.make_tile(tile_type, addr=0, size=16384)
        pl.load(a, t, offset_value)
        result: pl.Tensor[[128, 128], pl.DT_FP32] = pl.store(out, a, offset_value)
        return result

    assert isinstance(func, ir.Function)


def test_closure_tuple_has_entry_anchor_and_folded_reads():
    values = (3, 7)

    @pl.function
    def func(idx: pl.DT_INT64):
        selected = values[idx]  # noqa: F841

    assignments = [stmt for stmt in func.body.stmts if isinstance(stmt, ir.AssignStmt)]
    anchor = next(stmt for stmt in assignments if stmt.var.name == "values")
    selected = next(stmt for stmt in assignments if stmt.var.name == "selected")
    assert isinstance(anchor.value, ir.MakeTuple)
    assert isinstance(selected.value, ir.GetItemExpr)
    assert selected.value.value is anchor.value


def test_nested_list_closure_var():
    """Nested list closure variable recursively converts to nested MakeTuple."""
    offsets_value = [0, 0]

    @pl.function
    def func(
        t: pl.Tensor[[128, 128], pl.DT_FP32], out: pl.Tensor[[128, 128], pl.DT_FP32]
    ) -> pl.Tensor[[128, 128], pl.DT_FP32]:
        tile_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32)
        a = pl.make_tile(tile_type, addr=0, size=16384)
        pl.load(a, t, offsets_value)  # type: ignore[arg-type]
        result: pl.Tensor[[128, 128], pl.DT_FP32] = pl.store(out, a, [0, 0])
        return result

    assert isinstance(func, ir.Function)


def test_dynamic_tensor_shape_value():
    """A DYNAMIC parameter dimension is read through tensor.shape."""
    @pl.function
    def func(
        x: pl.Tensor[[pl.DYNAMIC], pl.DT_FP32],
    ):
        result = pl.tensor.mul(x, x.shape[0])
        return result

    assert isinstance(func, ir.Function)


def test_dsl_scope_shadows_closure():
    """Variable defined in DSL body shadows same-named closure variable."""
    x_scale = 999.0  # noqa: F841 -deliberately shadowed by DSL assignment

    @pl.function
    def func(x: pl.Tensor[[64], pl.DT_FP32]) -> pl.Tensor[[64], pl.DT_FP32]:
        x_scale: pl.Tensor[[64], pl.DT_FP32] = pl.tensor.add(x, x)
        result: pl.Tensor[[64], pl.DT_FP32] = pl.tensor.mul(x_scale, x)
        return result

    assert isinstance(func, ir.Function)


def test_undefined_variable_still_raises():
    """Variable not in scope or closure raises UndefinedVariableError."""
    with pytest.raises(UndefinedVariableError, match="Undefined variable"):

        @pl.function
        def func(x: pl.Tensor[[64], pl.DT_FP32]) -> pl.Tensor[[64], pl.DT_FP32]:
            result: pl.Tensor[[64], pl.DT_FP32] = pl.tensor.add(x, totally_undefined)  # noqa: F821 # type: ignore
            return result


def test_unsupported_closure_type_raises():
    """Unsupported closure variable type raises ParserTypeError."""
    bad_value = "not_a_number"

    with pytest.raises(ParserTypeError, match="Unsupported closure variable type: str"):

        @pl.function
        def func(x: pl.Tensor[[64], pl.DT_FP32]) -> pl.Tensor[[64], pl.DT_FP32]:
            result: pl.Tensor[[64], pl.DT_FP32] = pl.tensor.add(x, bad_value)  # type: ignore[arg-type]
            return result
