#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See the License in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Error handling and boundary tests for array fields in pl.struct / pl.struct_array."""

from __future__ import annotations

from pypto_pro import ir
import pypto_pro.language as pl
from pypto_pro.language.parser.diagnostics import (
    ParserSyntaxError,
    ParserTypeError,
    UnsupportedFeatureError,
)
import pytest

# =============================================================================
# 8.1 Array field index out of bounds
# =============================================================================

def test_err_arr_field_index_out_of_bounds():
    """s.arr[4] on a 4-element array field should fail at compile time."""

    with pytest.raises(ParserSyntaxError):
        @pl.function
        def kernel():
            s = pl.struct("S", arr=[0, 0, 0, 0])
            val: pl.DT_INT64 = s.arr[4]
            return val


def test_err_arr_field_negative_index():
    """s.arr[-1] should fail — negative indices are not supported."""

    with pytest.raises(ParserSyntaxError):
        @pl.function
        def kernel():
            s = pl.struct("S", arr=[0, 0, 0, 0])
            val: pl.DT_INT64 = s.arr[-1]
            return val


def test_err_struct_array_arr_field_oob():
    """arr[5].data[0] on a size=3 struct_array should fail at compile time."""

    with pytest.raises(ParserSyntaxError):
        @pl.function
        def kernel():
            arr = pl.struct_array(3, "S", x=0, data=[0, 0, 0])
            val: pl.DT_INT64 = arr[5].data[0]
            return val


# =============================================================================
# 8.2 Subscript on a scalar field
# =============================================================================

def test_err_subscript_scalar_field_read():
    """s.x[0] where x is a scalar field should raise ParserTypeError."""

    with pytest.raises(ParserTypeError, match="Subscript requires tuple"):
        @pl.function
        def kernel():
            s = pl.struct("S", x=0)
            val: pl.DT_INT64 = s.x[0]
            return val


def test_err_subscript_scalar_field_write():
    """s.x[0] = 10 where x is a scalar field should raise ParserTypeError."""

    with pytest.raises(ParserTypeError, match="field is not an array"):
        @pl.function
        def kernel():
            s = pl.struct("S", x=0)
            s.x[0] = 10
            return s.x


def test_err_subscript_nested_struct_field_write():
    """s.inner[0] = 10 where inner is a nested struct must NOT be treated as an array."""

    with pytest.raises(ParserTypeError, match="field is not an array"):
        @pl.function
        def kernel():
            inner = pl.struct("Inner", a=0, b=0)
            s = pl.struct("Outer", x=0, inner=inner)
            s.inner[0] = 10
            return s.x


def test_err_subscript_array_field_oob_write():
    """s.arr[4] = 10 on a 4-element array field should fail at compile time."""

    with pytest.raises(ParserSyntaxError, match="out of bounds"):
        @pl.function
        def kernel():
            s = pl.struct("S", arr=[0, 0, 0, 0])
            s.arr[4] = 10
            return s.arr[0]


def test_err_subscript_array_field_multi_index_write():
    """s.arr[0, 1] = 10 (multi-dimensional) should fail at compile time."""

    with pytest.raises(ParserSyntaxError, match="Multi-dimensional subscript"):
        @pl.function
        def kernel():
            s = pl.struct("S", arr=[0, 0, 0, 0])
            s.arr[0, 1] = 10
            return s.arr[0]


# =============================================================================
# 8.3 Nonexistent array field
# =============================================================================

def test_err_nonexistent_field_read():
    """s.nonexistent[0] should raise an error for a missing field."""

    with pytest.raises((UnsupportedFeatureError, ParserTypeError)):
        @pl.function
        def kernel():
            s = pl.struct("S", x=0)
            val: pl.DT_INT64 = s.nonexistent[0]
            return val


def test_err_nonexistent_field_write():
    """s.nonexistent[0] = 1 should raise ParserTypeError for a missing field."""

    with pytest.raises(ParserTypeError, match="Struct has no field"):
        @pl.function
        def kernel():
            s = pl.struct("S", x=0)
            s.nonexistent[0] = 1
            return s.x


# =============================================================================
# 8.4 Whole array field assignment — now supported via element-wise expansion
# =============================================================================

def test_arr_field_whole_assign_expands():
    """s.arr = [1, 2, 3, 4] is expanded to 4 element-wise struct.set calls."""

    @pl.function
    def kernel():
        s = pl.struct("S", arr=[0, 0, 0, 0])
        s.arr = [1, 2, 3, 4]
        val: pl.DT_INT64 = s.arr[0]
        return val

    assert isinstance(kernel, ir.Function)
    eval_stmts = [stmt for stmt in kernel.body.stmts if isinstance(stmt, ir.EvalStmt)]
    struct_set_stmts = [stmt for stmt in eval_stmts if "struct.set" in str(stmt)]
    assert len(struct_set_stmts) == 4


def test_arr_field_whole_assign_length_mismatch():
    """s.arr = [1, 2] on a 4-element array field should raise ParserSyntaxError."""

    with pytest.raises(ParserSyntaxError, match="expects 4 elements, got 2"):
        @pl.function
        def kernel():
            s = pl.struct("S", arr=[0, 0, 0, 0])
            s.arr = [1, 2]
            return s.arr[0]
