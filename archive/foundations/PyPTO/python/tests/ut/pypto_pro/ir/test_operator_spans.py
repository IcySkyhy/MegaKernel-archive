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
"""Smoke tests for automatic span capture in scalar operator overloads."""

from pypto_pro import DataType, ir
import pytest


def _scalar_var(name: str, dtype=DataType.INT32):
    return ir.Var(name, ir.ScalarType(dtype), ir.Span.unknown())


def _assert_captured_here(expr):
    assert expr.span.filename.endswith("test_operator_spans.py")
    assert expr.span.is_valid()
    assert expr.span.begin_line > 0


def test_binary_and_comparison_operators_capture_span():
    x = _scalar_var("x")
    y = _scalar_var("y")

    _assert_captured_here(x + y)
    _assert_captured_here(x < y)


def test_reverse_and_unary_operators_capture_span():
    x = _scalar_var("x")

    _assert_captured_here(5 + x)
    _assert_captured_here(-x)


def test_nested_operations_capture_distinct_lines():
    x = _scalar_var("x", DataType.FP32)
    y = _scalar_var("y", DataType.FP32)
    z = _scalar_var("z", DataType.FP32)

    lhs = x + y
    rhs = lhs * z

    _assert_captured_here(lhs)
    _assert_captured_here(rhs)
    assert lhs.span.begin_line != rhs.span.begin_line


def test_tensor_var_operator_still_rejects_scalar_arithmetic():
    tensor_var = ir.Var("t", ir.TensorType([128, 256], DataType.FP32), ir.Span.unknown())
    scalar_var = _scalar_var("x")

    with pytest.raises(ValueError, match="ScalarType"):
        _ = tensor_var + scalar_var
