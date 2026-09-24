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

if TYPE_CHECKING:
    from collections.abc import Callable


def _make_resolver(
    closure_vars: dict | None = None, scope_lookup: "Callable[[str], Any | None] | None" = None
) -> TypeResolver:
    """Create a TypeResolver with ExprEvaluator from closure_vars."""
    ev = ExprEvaluator(closure_vars=closure_vars or {})
    return TypeResolver(expr_evaluator=ev, scope_lookup=scope_lookup)


def test_ptr_basic_annotation():
    """pl.Ptr[pl.DT_FP32] resolves to PtrType with FP32 dtype."""
    resolver = _make_resolver()
    node = ast.parse("pl.Ptr[pl.DT_FP32]", mode="eval").body
    result = resolver.resolve_type(node)

    assert isinstance(result, ir.PtrType)
    assert result.dtype == DataType.FP32


def test_ptr_int_dtype():
    """pl.Ptr[pl.DT_INT8] resolves to PtrType with INT8 dtype."""
    resolver = _make_resolver()
    node = ast.parse("pl.Ptr[pl.DT_INT8]", mode="eval").body
    result = resolver.resolve_type(node)

    assert isinstance(result, ir.PtrType)
    assert result.dtype == DataType.INT8


def test_ptr_via_pl_function():
    """@pl.function with pl.Ptr[pl.DT_FP32] parameter resolves to PtrType."""

    @pl.function
    def func(ptr: pl.Ptr[pl.DT_FP32]):
        pass

    assert isinstance(func.params[0].type, ir.PtrType)
    assert func.params[0].type.dtype == DataType.FP32
