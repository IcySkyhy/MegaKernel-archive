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
"""Unit tests for IR Builder."""

from pypto_pro import DataType, ir
from pypto_pro.ir import IRBuilder


def _single_stmt(stmt: ir.Stmt) -> ir.Stmt:
    assert isinstance(stmt, ir.SeqStmts)
    assert len(stmt.stmts) == 1
    return stmt.stmts[0]


def test_empty_program():
    """Test building an empty program."""
    ib = IRBuilder()

    with ib.program("empty_program") as p:
        pass

    program = p.get_result()

    assert program is not None
    assert program.name == "empty_program"
    assert len(program.functions) == 0


def test_program_with_single_function():
    """Test building a program with a single function."""
    ib = IRBuilder()

    with ib.program("simple_program") as p:
        # Build function
        with ib.function("add") as f:
            x = f.param("x", ir.ScalarType(DataType.INT64))
            y = f.param("y", ir.ScalarType(DataType.INT64))
            f.return_type(ir.ScalarType(DataType.INT64))

            result = ib.let("result", x + y)
            ib.return_stmt(result)

        add_func = f.get_result()
        p.add_function(add_func)

    program = p.get_result()

    assert program is not None
    assert program.name == "simple_program"
    assert len(program.functions) == 1

    # Verify function is accessible
    retrieved_func = program.get_function("add")
    assert retrieved_func is not None
    assert retrieved_func.name == "add"
    assert len(retrieved_func.params) == 2


def test_program_with_multiple_functions():
    """Test building a program with multiple functions."""
    ib = IRBuilder()

    with ib.program("math_lib") as p:
        # Build square function
        with ib.function("square") as f:
            x = f.param("x", ir.ScalarType(DataType.INT64))
            f.return_type(ir.ScalarType(DataType.INT64))
            result = ib.let("result", x * x)
            ib.return_stmt(result)

        p.add_function(f.get_result())

        # Build double function
        with ib.function("double") as f:
            x = f.param("x", ir.ScalarType(DataType.INT64))
            f.return_type(ir.ScalarType(DataType.INT64))
            two = ib.let("two", ir.ConstInt(2, DataType.INT64, ir.Span.unknown()))
            result = ib.let("result", x * two)
            ib.return_stmt(result)

        p.add_function(f.get_result())

    program = p.get_result()

    assert program is not None
    assert len(program.functions) == 2

    # Verify both functions are accessible
    square_func = program.get_function("square")
    double_func = program.get_function("double")
    assert square_func is not None
    assert double_func is not None


def test_program_with_cross_function_calls():
    """Test building a program with cross-function calls using Op."""
    ib = IRBuilder()

    with ib.program("call_test") as p:
        # Build square function
        with ib.function("square") as f:
            x = f.param("x", ir.ScalarType(DataType.INT64))
            f.return_type(ir.ScalarType(DataType.INT64))
            result = ib.let("result", x * x)
            ib.return_stmt(result)

        p.add_function(f.get_result())

        # Build sum_of_squares function that calls square
        with ib.function("sum_of_squares", func_type=ir.FunctionType.Opaque) as f:
            a = f.param("a", ir.ScalarType(DataType.INT64))
            b = f.param("b", ir.ScalarType(DataType.INT64))
            f.return_type(ir.ScalarType(DataType.INT64))

            # Call square function using Op - return type auto-inferred
            square_op = ir.Op("square")
            a_sq = ib.let("a_sq", ir.Call(square_op, [a], ir.Span.unknown()))
            b_sq = ib.let("b_sq", ir.Call(square_op, [b], ir.Span.unknown()))
            result = ib.let("result", a_sq + b_sq)
            ib.return_stmt(result)

        p.add_function(f.get_result())

    program = p.get_result()

    assert program is not None
    assert len(program.functions) == 2

    # Verify cross-function call exists in IR
    sum_func = program.get_function("sum_of_squares")
    assert sum_func is not None


def test_simple_section_vector():
    """Test building a simple Vector section."""
    ib = IRBuilder()

    with ib.function("section_vector_test") as f:
        x = f.param("x", ir.ScalarType(DataType.INT64))
        f.return_type(ir.ScalarType(DataType.INT64))

        with ib.section(ir.SectionKind.Vector):
            result = ib.var("result", ir.ScalarType(DataType.INT64))
            add_expr = ir.Add(x, x, DataType.INT64, ir.Span.unknown())
            ib.assign(result, add_expr)

    func = f.get_result()

    assert func is not None
    body = _single_stmt(func.body)
    assert isinstance(body, ir.SectionStmt)
    assert body.section_kind == ir.SectionKind.Vector


def test_section_vector_with_multiple_statements():
    """Test Vector section with multiple statements in body."""
    ib = IRBuilder()

    with ib.function("multi_section_test") as f:
        x = f.param("x", ir.ScalarType(DataType.INT64))
        f.return_type(ir.ScalarType(DataType.INT64))

        with ib.section(ir.SectionKind.Vector):
            a = ib.var("a", ir.ScalarType(DataType.INT64))
            b = ib.var("b", ir.ScalarType(DataType.INT64))

            add_expr = ir.Add(x, x, DataType.INT64, ir.Span.unknown())
            ib.assign(a, add_expr)

            mul_expr = ir.Mul(a, a, DataType.INT64, ir.Span.unknown())
            ib.assign(b, mul_expr)

    func = f.get_result()

    assert func is not None
    body = _single_stmt(func.body)
    assert isinstance(body, ir.SectionStmt)
    assert body.section_kind == ir.SectionKind.Vector
    assert isinstance(body.body, ir.SeqStmts)


def test_simple_section_cube():
    """Test building a simple Cube section."""
    ib = IRBuilder()

    with ib.function("section_cube_test") as f:
        x = f.param("x", ir.ScalarType(DataType.INT64))
        f.return_type(ir.ScalarType(DataType.INT64))

        with ib.section(ir.SectionKind.Cube):
            result = ib.var("result", ir.ScalarType(DataType.INT64))
            mul_expr = ir.Mul(x, x, DataType.INT64, ir.Span.unknown())
            ib.assign(result, mul_expr)

    func = f.get_result()

    assert func is not None
    body = _single_stmt(func.body)
    assert isinstance(body, ir.SectionStmt)
    assert body.section_kind == ir.SectionKind.Cube


def test_section_cube_with_multiple_statements():
    """Test Cube section with multiple statements in body."""
    ib = IRBuilder()

    with ib.function("multi_section_test") as f:
        x = f.param("x", ir.ScalarType(DataType.INT64))
        f.return_type(ir.ScalarType(DataType.INT64))

        with ib.section(ir.SectionKind.Cube):
            a = ib.var("a", ir.ScalarType(DataType.INT64))
            b = ib.var("b", ir.ScalarType(DataType.INT64))

            add_expr = ir.Add(x, x, DataType.INT64, ir.Span.unknown())
            ib.assign(a, add_expr)

            mul_expr = ir.Mul(a, a, DataType.INT64, ir.Span.unknown())
            ib.assign(b, mul_expr)

    func = f.get_result()

    assert func is not None
    body = _single_stmt(func.body)
    assert isinstance(body, ir.SectionStmt)
    assert body.section_kind == ir.SectionKind.Cube
    assert isinstance(body.body, ir.SeqStmts)
