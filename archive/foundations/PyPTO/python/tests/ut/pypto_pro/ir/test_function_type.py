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
"""Tests for function type attribute feature."""

from pypto_pro import DataType, ir
from pypto_pro.ir import IRBuilder
import pypto_pro.language as pl


def test_ir_builder_with_function_type():
    """Test IR Builder with function type parameter."""
    ib = IRBuilder()
    span = ir.Span.unknown()
    dtype = DataType.INT64

    # Build function with InCore type
    with ib.function("aicore_kernel", span=span, func_type=ir.FunctionType.InCore) as f:
        y = f.param("y", ir.ScalarType(dtype), span=span)
        f.return_type(ir.ScalarType(dtype))
        ib.return_stmt(y, span=span)

    func2 = f.get_result()
    assert func2.name == "aicore_kernel"
    assert func2.func_type == ir.FunctionType.InCore


def test_function_type_python_print():
    """Test that function type is correctly printed in Python syntax."""
    ib = IRBuilder()
    span = ir.Span.unknown()
    dtype = DataType.INT64

    # Opaque function should not print type parameter
    with ib.function("default_func", span=span) as f:
        x = f.param("x", ir.ScalarType(dtype), span=span)
        f.return_type(ir.ScalarType(dtype))
        ib.return_stmt(x, span=span)

    func_opaque = f.get_result()
    printed = ir.python_print(func_opaque, "pl")
    assert "@pl.function\n" in printed
    assert "type=" not in printed  # Opaque should not print type parameter

    # InCore function should print type parameter
    with ib.function("kernel", span=span, func_type=ir.FunctionType.InCore) as f:
        x = f.param("x", ir.ScalarType(dtype), span=span)
        f.return_type(ir.ScalarType(dtype))
        ib.return_stmt(x, span=span)

    func_incore = f.get_result()
    printed_incore = ir.python_print(func_incore, "pl")
    assert "@pl.function(type=pl.FunctionType.InCore)" in printed_incore


def test_function_type_decorator_parsing():
    """Test parsing functions with type parameter in decorator."""

    # Test Opaque (default)
    @pl.function
    def default_func(x: pl.Tensor[[4], pl.DT_INT64]) -> pl.Tensor[[4], pl.DT_INT64]:
        return x

    assert default_func.name == "default_func"
    assert default_func.func_type == ir.FunctionType.Opaque

    # Test InCore
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(x: pl.Tensor[[4], pl.DT_INT64]) -> pl.Tensor[[4], pl.DT_INT64]:
        return x

    assert kernel.name == "kernel"
    assert kernel.func_type == ir.FunctionType.InCore


def test_function_type_language_export():
    """Test that FunctionType is exported from language module."""

    assert hasattr(pl, "FunctionType")
    assert pl.FunctionType.Opaque
    assert pl.FunctionType.InCore
