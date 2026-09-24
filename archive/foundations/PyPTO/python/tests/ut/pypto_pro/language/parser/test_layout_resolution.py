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
"""Unit tests for TypeResolver."""
from __future__ import annotations

# DSL function bodies are parsed as AST, not executed -suppress pyright errors
# from type-checking the annotations and kwargs inside @pl.function bodies.
import ast
from typing import TYPE_CHECKING, Any

from pypto_pro import DataType, ir
import pypto_pro.language as pl
from pypto_pro.language.parser._expr_evaluator import ExprEvaluator
from pypto_pro.language.parser._type_resolver import TypeResolver
from pypto_pro.language.parser.diagnostics import ParserTypeError
import pytest

if TYPE_CHECKING:
    from collections.abc import Callable


def _make_resolver(
    closure_vars: dict | None = None, scope_lookup: "Callable[[str], Any | None] | None" = None
) -> TypeResolver:
    """Create a TypeResolver with ExprEvaluator from closure_vars."""
    ev = ExprEvaluator(closure_vars=closure_vars or {})
    return TypeResolver(expr_evaluator=ev, scope_lookup=scope_lookup)


@pytest.mark.parametrize(
    "layout_str, expected_layout",
    [
        ("pl.NZ", ir.TensorLayout.NZ),
        ("pl.DN", ir.TensorLayout.DN),
    ],
)
def test_resolve_tensor_with_layout(layout_str, expected_layout):
    """Tensor with various layouts creates TensorType with TensorView."""
    resolver = _make_resolver()
    node = ast.parse(f"pl.Tensor[[64, 128], pl.DT_FP16, {layout_str}]", mode="eval").body
    result = resolver.resolve_type(node)

    assert isinstance(result, ir.TensorType)
    assert len(result.shape) == 2
    assert result.dtype == DataType.FP16
    assert result.tensor_view is not None
    assert result.tensor_view.layout == expected_layout


def test_resolve_tensor_without_layout_has_no_view():
    """Tensor without an explicit layout has no TensorView."""
    resolver = _make_resolver()
    node = ast.parse("pl.Tensor[[64, 128], pl.DT_FP16]", mode="eval").body
    result = resolver.resolve_type(node)

    assert isinstance(result, ir.TensorType)
    assert result.tensor_view is None


def test_resolve_tensor_layout_invalid():
    """Invalid layout raises ParserTypeError."""
    resolver = _make_resolver()
    node = ast.parse("pl.Tensor[[64, 128], pl.DT_FP16, pl.INVALID]", mode="eval").body

    with pytest.raises(ParserTypeError, match="Unknown layout"):
        resolver.resolve_type(node)


def test_resolve_layout_bare_name():
    """Layout specified as bare name (NZ) instead of pl.NZ."""
    resolver = _make_resolver()
    node = ast.parse("pl.Tensor[[64, 128], pl.DT_FP16, NZ]", mode="eval").body
    result = resolver.resolve_type(node)

    assert isinstance(result, ir.TensorType)
    assert result.tensor_view is not None
    assert result.tensor_view.layout == ir.TensorLayout.NZ


def test_resolve_layout_from_closure():
    """Layout from closure variable."""
    resolver = _make_resolver(closure_vars={"my_layout": ir.TensorLayout.NZ})
    node = ast.parse("pl.Tensor[[64, 128], pl.DT_FP16, my_layout]", mode="eval").body
    result = resolver.resolve_type(node)

    assert isinstance(result, ir.TensorType)
    assert result.tensor_view is not None
    assert result.tensor_view.layout == ir.TensorLayout.NZ


def test_resolve_layout_closure_invalid_type():
    """Non-TensorLayout closure variable raises error."""
    resolver = _make_resolver(closure_vars={"my_layout": "NZ"})
    node = ast.parse("pl.Tensor[[64, 128], pl.DT_FP16, my_layout]", mode="eval").body

    with pytest.raises(ParserTypeError, match="must be a TensorLayout"):
        resolver.resolve_type(node)


def test_resolve_tensor_layout_with_dynamic_shape():
    """Layout works with dynamic shapes."""
    resolver = _make_resolver(closure_vars={"pl": pl})
    node = ast.parse("pl.Tensor[[pl.DYNAMIC, 128], pl.DT_FP16, pl.NZ]", mode="eval").body
    result = resolver.resolve_param_type(node, parameter_name="x")

    assert isinstance(result, ir.TensorType)
    assert isinstance(result.shape[0], ir.Var)
    assert result.shape[0].name == "__pypto_dyn_x_0"
    assert result.tensor_view is not None
    assert result.tensor_view.layout == ir.TensorLayout.NZ


def test_resolve_tensor_layout_with_shape_variable():
    """Layout works with shape variable from closure."""
    resolver = _make_resolver(closure_vars={"shape": [64, 128]})
    node = ast.parse("pl.Tensor[shape, pl.DT_FP16, pl.DN]", mode="eval").body
    result = resolver.resolve_type(node)

    assert isinstance(result, ir.TensorType)
    assert len(result.shape) == 2
    assert result.tensor_view is not None
    assert result.tensor_view.layout == ir.TensorLayout.DN


def test_function_with_tensor_layout():
    """@pl.function with layout in parameter and return type."""

    @pl.function
    def func(
        x: pl.Tensor[[64, 128], pl.DT_FP16, pl.NZ],
    ) -> pl.Tensor[[64, 128], pl.DT_FP16, pl.NZ]:
        return x

    assert isinstance(func, ir.Function)
    param_type = func.params[0].type
    assert isinstance(param_type, ir.TensorType)
    assert param_type.dtype == DataType.FP16
    assert param_type.tensor_view is not None
    assert param_type.tensor_view.layout == ir.TensorLayout.NZ

    ret_type = func.return_types[0]
    assert isinstance(ret_type, ir.TensorType)
    assert ret_type.tensor_view is not None
    assert ret_type.tensor_view.layout == ir.TensorLayout.NZ


def test_function_mixed_layout_and_no_layout():
    """@pl.function with some params having layout and some not."""

    @pl.function
    def func(
        a: pl.Tensor[[64, 128], pl.DT_FP16, pl.NZ],
        b: pl.Tensor[[64, 128], pl.DT_FP16],
    ) -> pl.Tensor[[64, 128], pl.DT_FP16]:
        return a

    a_type = func.params[0].type
    b_type = func.params[1].type
    assert isinstance(a_type, ir.TensorType)
    assert isinstance(b_type, ir.TensorType)
    assert a_type.tensor_view is not None
    assert a_type.tensor_view.layout == ir.TensorLayout.NZ
    assert b_type.tensor_view is not None


def test_program_with_tensor_layout():
    """@pl.program with layout annotations."""

    @pl.program
    class MyProgram:
        @pl.function
        def compute(
            self,
            x: pl.Tensor[[64, 128], pl.DT_FP16, pl.NZ],
        ) -> pl.Tensor[[64, 128], pl.DT_FP16, pl.NZ]:
            return x

    assert isinstance(MyProgram, ir.Program)
    func = list(MyProgram.functions.values())[0]
    param_type = func.params[0].type
    assert isinstance(param_type, ir.TensorType)
    assert param_type.tensor_view is not None
    assert param_type.tensor_view.layout == ir.TensorLayout.NZ


def test_function_layout_from_closure_variable():
    """@pl.function with layout from closure variable."""
    layout = pl.NZ

    @pl.function
    def func(
        x: pl.Tensor[[64, 128], pl.DT_FP16, layout],
    ) -> pl.Tensor[[64, 128], pl.DT_FP16, layout]:
        return x

    param_type = func.params[0].type
    assert isinstance(param_type, ir.TensorType)
    assert param_type.tensor_view is not None
    assert param_type.tensor_view.layout == ir.TensorLayout.NZ


@pytest.mark.parametrize(
    "layout, expected",
    [
        (pl.NZ, ir.TensorLayout.NZ),
        (pl.DN, ir.TensorLayout.DN),
    ],
)
def test_parametrized_layout(layout, expected):
    """pytest.mark.parametrize with layout."""

    @pl.function
    def func(
        x: pl.Tensor[[64, 128], pl.DT_FP16, layout],
    ) -> pl.Tensor[[64, 128], pl.DT_FP16, layout]:
        return x

    param_type = func.params[0].type
    assert isinstance(param_type, ir.TensorType)
    assert param_type.tensor_view is not None
    assert param_type.tensor_view.layout == expected


def test_function_with_dn_layout():
    """@pl.function with DN layout for column-major tensors."""

    @pl.function
    def func(
        x: pl.Tensor[[16, 1], pl.DT_FP16, pl.DN],
    ) -> pl.Tensor[[16, 1], pl.DT_FP16, pl.DN]:
        return x

    param_type = func.params[0].type
    assert isinstance(param_type, ir.TensorType)
    assert param_type.tensor_view is not None
    assert param_type.tensor_view.layout == ir.TensorLayout.DN


def test_backward_compat_bare_layout():
    """pl.Tensor[[shape], dtype, pl.NZ] layout syntax works."""
    resolver = _make_resolver()
    node = ast.parse("pl.Tensor[[64, 128], pl.DT_FP32, pl.NZ]", mode="eval").body
    result = resolver.resolve_type(node)

    assert isinstance(result, ir.TensorType)
    assert result.tensor_view is not None
    assert result.tensor_view.layout == ir.TensorLayout.NZ


def test_backward_compat_memref():
    """pl.Tensor[[shape], dtype, pl.MemRef(...)] syntax works."""
    resolver = _make_resolver()
    node = ast.parse(
        "pl.Tensor[[64], pl.DT_FP32, pl.MemRef(pl.MemorySpace.DDR, 0, 1024, 0)]", mode="eval"
    ).body
    result = resolver.resolve_type(node)

    assert isinstance(result, ir.TensorType)
    assert result.memref is not None
    assert result.tensor_view is None
