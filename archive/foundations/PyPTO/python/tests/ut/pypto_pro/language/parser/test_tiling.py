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
"""Unit tests for tiling parameter support in the PyPTO language DSL."""

from __future__ import annotations

from dataclasses import dataclass
import logging

from pypto_pro import ir
import pypto_pro.language as pl
from pypto_pro.language.parser.diagnostics import (
    ParserSyntaxError,
    ParserTypeError,
    UnsupportedFeatureError,
)
import pytest

from pypto.pypto_impl.ir import DataType


def test_tiling_field_named_shape_uses_struct_field_access():
    """A tiling field named shape must not be routed as Tensor.shape."""

    @dataclass
    class Tiling:
        shape: int[4]

    @pl.function
    def kernel(tiling: Tiling) -> pl.DT_INT64:
        result: pl.DT_INT64 = tiling.shape[3]
        return result

    assert isinstance(kernel, ir.Function)
    assert isinstance(kernel.params[0].type, ir.TupleType)


def test_tiling_param_lowered_to_single_struct():
    @dataclass
    class Tiling:
        x: int
        y: float

    @pl.function
    def kernel(tiling: Tiling) -> pl.DT_INT64:
        result: pl.DT_INT64 = tiling.x
        return result

    assert isinstance(kernel, ir.Function)
    # The tiling class is lowered to a single struct (TupleType) parameter.
    assert len(kernel.params) == 1
    assert kernel.params[0].name == "tiling"
    assert isinstance(kernel.params[0].type, ir.TupleType)
    assert len(kernel.params[0].type.types) == 2


def test_tiling_scalar_dtypes_are_correct():
    @dataclass
    class Tiling:
        n: int
        scale: float
        flag: bool

    @pl.function
    def kernel(tiling: Tiling) -> pl.DT_INT64:
        result: pl.DT_INT64 = tiling.n
        return result

    assert isinstance(kernel, ir.Function)
    assert len(kernel.params) == 1
    tuple_type = kernel.params[0].type
    assert isinstance(tuple_type, ir.TupleType)
    elem_types = tuple_type.types
    # Scalar fields become scalar tuple elements in declaration order.
    assert elem_types[0].dtype == DataType.INDEX
    assert elem_types[1].dtype == DataType.FP32
    assert elem_types[2].dtype == DataType.BOOL


def test_tensors_plus_tiling_last():
    @dataclass
    class Tiling:
        n: int
        m: int
        arr: float[3]

    @pl.function
    def kernel(
        x: pl.Tensor[[64], pl.DT_FP32],
        y: pl.Tensor[[64], pl.DT_FP32],
        tiling: Tiling,
    ) -> pl.Tensor[[64], pl.DT_FP32]:
        n = tiling.n  # noqa: F841
        m = tiling.m  # noqa: F841
        tmp1 = tiling.arr[1]  # noqa: F841
        return x

    logging.info("%s", kernel)

    assert isinstance(kernel, ir.Function)
    # Tensors stay individual params; tiling collapses to one struct param last.
    assert len(kernel.params) == 3
    param_names = [p.name for p in kernel.params]
    assert param_names == ["x", "y", "tiling"]
    assert isinstance(kernel.params[2].type, ir.TupleType)


def test_tiling_name_registered_as_struct_in_scope():
    """The tiling name is registered in scope as a single struct param var."""

    @dataclass
    class Tiling:
        x: int
        y: int

    @pl.function
    def kernel(tiling: Tiling) -> pl.DT_INT64:
        a: pl.DT_INT64 = tiling.x
        b: pl.DT_INT64 = tiling.y  # noqa: F841
        return a

    assert isinstance(kernel, ir.Function)
    assert len(kernel.params) == 1
    assert kernel.params[0].name == "tiling"
    assert isinstance(kernel.params[0].type, ir.TupleType)


def test_tiling_field_access_lowers_to_getitem():
    """Accessing tiling.x lowers to a GetItemExpr on the struct param."""

    @dataclass
    class Tiling:
        x: int

    @pl.function
    def kernel(tiling: Tiling) -> pl.DT_INT64:
        result: pl.DT_INT64 = tiling.x
        return result

    assert isinstance(kernel, ir.Function)
    assert len(kernel.params) == 1
    assert kernel.params[0].name == "tiling"
    # tiling.x lowers to tiling[0] (field index 0 of the struct).
    assert "getitem(%tiling, 0)" in str(kernel)


def test_tiling_registry_reset_between_functions():
    """Tiling registry is reset for each new function, preventing leakage."""

    @dataclass
    class Tiling:
        n: int

    @pl.function
    def func1(tiling: Tiling):
        x: pl.DT_INT64 = tiling.n
        return x

    # Second function should not see tiling from first function
    @pl.function
    def func2(x: pl.Tensor[[64], pl.DT_FP32]) -> pl.Tensor[[64], pl.DT_FP32]:
        return x

    assert isinstance(func1, ir.Function)
    assert isinstance(func2, ir.Function)


def test_tiling_not_last_raises_parser_syntax_error():
    """Tiling parameter that is not the last param raises ParserSyntaxError."""

    @dataclass
    class Tiling:
        x: int

    with pytest.raises(ParserSyntaxError, match="must be the last parameter"):

        @pl.function
        def kernel(
            tiling: Tiling,  # Not last!
            x: pl.Tensor[[64], pl.DT_FP32],
        ):
            pass


def test_multiple_tiling_params_raises_parser_syntax_error():
    """More than one tiling parameter raises ParserSyntaxError."""

    @dataclass
    class TilingA:
        x: int

    @dataclass
    class TilingB:
        y: float

    with pytest.raises(ParserSyntaxError, match="at most 1"):

        @pl.function
        def kernel(
            ta: TilingA,
            tb: TilingB,
        ):
            pass


def test_nonexistent_tiling_field_raises_error():
    """Accessing a field that doesn't exist on the tiling struct raises an error.

    A missing field is not in the struct's named-tuple field table, so the attribute
    access fails to lower and surfaces as UnsupportedFeatureError.
    """

    @dataclass
    class Tiling:
        x: int

    with pytest.raises(UnsupportedFeatureError, match="Standalone attribute access not supported"):

        @pl.function
        def kernel(tiling: Tiling) -> pl.DT_INT64:
            result: pl.DT_INT64 = tiling.nonexistent  # type: ignore[attr-defined]
            return result


def test_array_field_lowers_to_nested_tuple():
    @dataclass
    class Tiling:
        offsets: int[3]

    @pl.function
    def kernel(tiling: Tiling) -> pl.DT_INT64:
        result: pl.DT_INT64 = tiling.offsets[0]
        return result

    assert isinstance(kernel, ir.Function)
    assert len(kernel.params) == 1
    tuple_type = kernel.params[0].type
    assert isinstance(tuple_type, ir.TupleType)
    # An int[3] field becomes a nested TupleType of 3 scalars.
    assert len(tuple_type.types) == 1
    inner = tuple_type.types[0]
    assert isinstance(inner, ir.TupleType)
    assert len(inner.types) == 3


def test_array_field_dtypes():
    @dataclass
    class Tiling:
        ints: int[2]
        floats: float[2]
        bools: bool[2]

    @pl.function
    def kernel(tiling: Tiling) -> pl.DT_INT64:
        result: pl.DT_INT64 = tiling.ints[0]
        return result

    assert isinstance(kernel, ir.Function)
    assert len(kernel.params) == 1
    elem_types = kernel.params[0].type.types
    assert all(t.dtype == DataType.INDEX for t in elem_types[0].types)
    assert all(t.dtype == DataType.FP32 for t in elem_types[1].types)
    assert all(t.dtype == DataType.BOOL for t in elem_types[2].types)


def test_mixed_scalar_and_array_fields():
    @dataclass
    class Tiling:
        n: int
        offsets: int[2]
        scale: float

    @pl.function
    def kernel(tiling: Tiling) -> pl.DT_INT64:
        result: pl.DT_INT64 = tiling.n
        return result

    assert isinstance(kernel, ir.Function)
    assert len(kernel.params) == 1
    tuple_type = kernel.params[0].type
    assert len(tuple_type.types) == 3
    # Scalar field -> scalar element; array field -> nested tuple element.
    assert isinstance(tuple_type.types[0], ir.ScalarType)
    assert isinstance(tuple_type.types[1], ir.TupleType)
    assert len(tuple_type.types[1].types) == 2
    assert isinstance(tuple_type.types[2], ir.ScalarType)


def test_array_subscript_access():
    @dataclass
    class Tiling:
        offsets: int[3]

    @pl.function
    def kernel(tiling: Tiling) -> pl.DT_INT64:
        result: pl.DT_INT64 = tiling.offsets[1]
        return result

    assert isinstance(kernel, ir.Function)
    assert len(kernel.params) == 1
    assignments = [stmt for stmt in kernel.body.stmts if isinstance(stmt, ir.AssignStmt)]
    result = next(stmt for stmt in assignments if stmt.var.name == "result")
    assert isinstance(result.value, ir.GetItemExpr)
    assert isinstance(result.value.value, ir.Var)
    assert isinstance(result.value.slice, ir.ConstInt)
    assert result.value.slice.value == 1

    field = next(stmt for stmt in assignments if stmt.var.name == result.value.value.name)
    assert isinstance(field.value, ir.GetItemExpr)
    assert isinstance(field.value.value, ir.Var)
    assert field.value.value.name == "tiling"
    assert isinstance(field.value.slice, ir.ConstInt)
    assert field.value.slice.value == 0


def test_array_all_indices_accessible():
    @dataclass
    class Tiling:
        vals: int[3]

    @pl.function
    def kernel0(tiling: Tiling) -> pl.DT_INT64:
        result: pl.DT_INT64 = tiling.vals[0]
        return result

    @pl.function
    def kernel1(tiling: Tiling) -> pl.DT_INT64:
        result: pl.DT_INT64 = tiling.vals[1]
        return result

    @pl.function
    def kernel2(tiling: Tiling) -> pl.DT_INT64:
        result: pl.DT_INT64 = tiling.vals[2]
        return result

    assert isinstance(kernel0, ir.Function)
    assert isinstance(kernel1, ir.Function)
    assert isinstance(kernel2, ir.Function)


def test_array_bare_name_resolves_to_nested_tuple():
    """Bare array field access now resolves to the nested tuple value (no error)."""

    @dataclass
    class Tiling:
        offsets: int[3]

    @pl.function
    def kernel(tiling: Tiling) -> pl.DT_INT64:
        vals = tiling.offsets
        result: pl.DT_INT64 = vals[0]
        return result

    assert isinstance(kernel, ir.Function)
    # tiling.offsets lowers to tiling[0], the nested tuple holding the array elements.
    assert "getitem(%tiling, 0)" in str(kernel)


def test_array_out_of_bounds_raises_error():
    @dataclass
    class Tiling:
        offsets: int[3]

    with pytest.raises(ParserSyntaxError, match="out of bounds"):

        @pl.function
        def kernel(tiling: Tiling) -> pl.DT_INT64:
            result: pl.DT_INT64 = tiling.offsets[99]
            return result


def test_array_non_literal_index_allowed():
    """A non-literal index into a uniform array field is now supported."""

    @dataclass
    class Tiling:
        offsets: int[3]

    @pl.function
    def kernel(tiling: Tiling) -> pl.DT_INT64:
        # Use a previously-computed value as subscript (non-literal).
        first: pl.DT_INT64 = tiling.offsets[0]
        result: pl.DT_INT64 = tiling.offsets[first]  # type: ignore[index]
        return result

    assert isinstance(kernel, ir.Function)


def test_scalar_subscript_raises_type_error():
    @dataclass
    class Tiling:
        n: int

    with pytest.raises(ParserTypeError, match="Subscript requires tuple, tile, or tensor type"):

        @pl.function
        def kernel(tiling: Tiling) -> pl.DT_INT64:
            result: pl.DT_INT64 = tiling.n[0]  # type: ignore[index]
            return result
