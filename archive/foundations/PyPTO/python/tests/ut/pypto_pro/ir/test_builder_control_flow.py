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
import pytest


def _single_stmt(stmt: ir.Stmt) -> ir.Stmt:
    assert isinstance(stmt, ir.SeqStmts)
    assert len(stmt.stmts) == 1
    return stmt.stmts[0]


def test_simple_for_loop():
    """Test building a simple for loop."""
    ib = IRBuilder()

    with ib.function("loop_func") as f:
        f.return_type(ir.ScalarType(DataType.INT64))

        i = ib.var("i", ir.ScalarType(DataType.INT64))

        with ib.for_loop(i, 0, 10, 1):
            # Empty loop body
            pass

    func = f.get_result()

    assert func is not None
    # Function body should be a for loop
    body = _single_stmt(func.body)
    assert isinstance(body, ir.ForStmt)
    assert body.loop_var.name == "i"


def test_for_loop_with_iter_args():
    """Test for loop with iteration arguments."""
    ib = IRBuilder()

    with ib.function("sum_func") as f:
        n = f.param("n", ir.ScalarType(DataType.INT64))
        f.return_type(ir.ScalarType(DataType.INT64))

        i = ib.var("i", ir.ScalarType(DataType.INT64))

        with ib.for_loop(i, 0, n, 1) as loop:
            sum_iter = loop.iter_arg("sum", 0).iterVar
            sum_final = loop.return_var("sum_final")

            add_expr = ir.Add(sum_iter, i, DataType.INT64, ir.Span.unknown())
            yield_stmt = ir.YieldStmt([add_expr], ir.Span.unknown())  # type: ignore[arg-type]
            ib.emit(yield_stmt)
        ib.return_stmt(sum_final)
    func = f.get_result()

    assert func is not None


def test_for_loop_iter_args_mismatch_error():
    """Test that mismatched iter_args and return_vars raises error."""
    ib = IRBuilder()

    # The error will be raised when exiting the for_loop context
    # Note: Error handling with context managers can be complex, so we just
    # check that RuntimeError is raised
    with pytest.raises(RuntimeError):
        with ib.function("mismatch_func") as f:
            f.return_type(ir.ScalarType(DataType.INT64))

            i = ib.var("i", ir.ScalarType(DataType.INT64))

            with ib.for_loop(i, 0, 10, 1) as loop:
                # Add iter_arg but no return_var - should fail
                loop.iter_arg("sum", 0)
                    # Missing loop.return_var() - will fail when exiting context


def test_simple_while_loop():
    """Test building a simple while loop."""
    ib = IRBuilder()

    with ib.function("while_func") as f:
        f.return_type(ir.ScalarType(DataType.INT64))

        x = ib.var("x", ir.ScalarType(DataType.INT64))
        ten = ir.ConstInt(10, DataType.INT64, ir.Span.unknown())
        condition = ir.Lt(x, ten, DataType.INT64, ir.Span.unknown())

        with ib.while_loop(condition):
            # Empty loop body
            pass

    func = f.get_result()

    assert func is not None
    # Function body should be a while loop
    body = _single_stmt(func.body)
    assert isinstance(body, ir.WhileStmt)
    assert isinstance(body.condition, ir.Lt)


def test_while_loop_with_iter_args():
    """Test while loop with iteration arguments."""
    ib = IRBuilder()

    with ib.function("while_sum_func") as f:
        n = f.param("n", ir.ScalarType(DataType.INT64))
        f.return_type(ir.ScalarType(DataType.INT64))

        # Initialize x
        init_x = ir.ConstInt(0, DataType.INT64, ir.Span.unknown())

        with ib.while_loop(ir.Lt(init_x, n, DataType.INT64, ir.Span.unknown())) as loop:
            x_iter = loop.iter_arg("x", init_x).iterVar
            x_final = loop.return_var("x_final")

            one = ir.ConstInt(1, DataType.INT64, ir.Span.unknown())
            add_expr = ir.Add(x_iter, one, DataType.INT64, ir.Span.unknown())
            yield_stmt = ir.YieldStmt([add_expr], ir.Span.unknown())  # type: ignore[arg-type]
            ib.emit(yield_stmt)

        ib.return_stmt(x_final)

    func = f.get_result()

    assert func is not None
    assert isinstance(func.body, ir.SeqStmts)
    # First statement should be the while loop
    while_stmt = func.body.stmts[0]
    assert isinstance(while_stmt, ir.WhileStmt)
    assert len(while_stmt.iter_args) == 1
    assert len(while_stmt.return_vars) == 1


def test_while_loop_iter_args_mismatch_error():
    """Test that mismatched iter_args and return_vars raises error."""
    ib = IRBuilder()

    with pytest.raises(RuntimeError):
        with ib.function("mismatch_func") as f:
            f.return_type(ir.ScalarType(DataType.INT64))

            x = ib.var("x", ir.ScalarType(DataType.INT64))
            ten = ir.ConstInt(10, DataType.INT64, ir.Span.unknown())
            condition = ir.Lt(x, ten, DataType.INT64, ir.Span.unknown())

            with ib.while_loop(condition) as loop:
                # Add iter_arg but no return_var - should fail
                loop.iter_arg("x", 0)


def test_while_loop_output():
    """Test while loop output() method."""
    ib = IRBuilder()

    with ib.function("while_output_func") as f:
        f.return_type(ir.ScalarType(DataType.INT64))

        init_x = ir.ConstInt(0, DataType.INT64, ir.Span.unknown())
        ten = ir.ConstInt(10, DataType.INT64, ir.Span.unknown())

        with ib.while_loop(ir.Lt(init_x, ten, DataType.INT64, ir.Span.unknown())) as loop:
            x_iter = loop.iter_arg("x", init_x).iterVar
            _x_final = loop.return_var("x_final")

            one = ir.ConstInt(1, DataType.INT64, ir.Span.unknown())
            add_expr = ir.Add(x_iter, one, DataType.INT64, ir.Span.unknown())
            yield_stmt = ir.YieldStmt([add_expr], ir.Span.unknown())  # type: ignore[arg-type]
            ib.emit(yield_stmt)

        # Access output after loop
        x_result = loop.output()
        ib.return_stmt(x_result)

    func = f.get_result()

    assert func is not None
    assert isinstance(func.body, ir.SeqStmts)


def test_simple_if_stmt():
    """Test building a simple if statement."""
    ib = IRBuilder()

    with ib.function("if_func") as f:
        x = f.param("x", ir.ScalarType(DataType.INT64))
        f.return_type(ir.ScalarType(DataType.INT64))

        zero = ir.ConstInt(0, DataType.INT64, ir.Span.unknown())
        condition = ir.Gt(x, zero, DataType.INT64, ir.Span.unknown())

        with ib.if_stmt(condition):
            result = ib.var("result", ir.ScalarType(DataType.INT64))
            ib.assign(result, x)

    func = f.get_result()

    assert func is not None
    body = _single_stmt(func.body)
    assert isinstance(body, ir.IfStmt)
    assert body.condition is not None
    assert body.else_body is None


def test_if_else_stmt():
    """Test building an if-else statement."""
    ib = IRBuilder()

    with ib.function("if_else_func") as f:
        x = f.param("x", ir.ScalarType(DataType.INT64))
        f.return_type(ir.ScalarType(DataType.INT64))

        zero = ir.ConstInt(0, DataType.INT64, ir.Span.unknown())
        one = ir.ConstInt(1, DataType.INT64, ir.Span.unknown())
        condition = ir.Gt(x, zero, DataType.INT64, ir.Span.unknown())
        result = ib.var("result", ir.ScalarType(DataType.INT64))

        with ib.if_stmt(condition) as if_builder:
            # Then branch
            ib.assign(result, one)

            # Else branch
            if_builder.else_()
            ib.assign(result, zero)

    func = f.get_result()

    assert func is not None
    if_stmt = _single_stmt(func.body)
    assert isinstance(if_stmt, ir.IfStmt)

    assert if_stmt.else_body is not None


def test_simple_return_with_value():
    """Test building a return statement with a value."""
    ib = IRBuilder()

    with ib.function("return_func") as f:
        x = f.param("x", ir.ScalarType(DataType.INT64))
        f.return_type(ir.ScalarType(DataType.INT64))

        ib.return_stmt(x)

    func = f.get_result()

    assert func is not None
    body = _single_stmt(func.body)
    assert isinstance(body, ir.ReturnStmt)
    assert len(body.value) == 1


def test_return_with_multiple_values():
    """Test return statement with multiple values."""
    ib = IRBuilder()

    with ib.function("multi_return_func") as f:
        x = f.param("x", ir.ScalarType(DataType.INT64))
        y = f.param("y", ir.ScalarType(DataType.INT64))
        f.return_type(ir.ScalarType(DataType.INT64))
        f.return_type(ir.ScalarType(DataType.INT64))

        ib.return_stmt([x, y])

    func = f.get_result()

    assert func is not None
    body = _single_stmt(func.body)
    assert isinstance(body, ir.ReturnStmt)
    assert len(body.value) == 2


def test_return_without_value():
    """Test return statement without values."""
    ib = IRBuilder()

    with ib.function("void_return_func") as f:
        f.return_type(ir.ScalarType(DataType.INT64))

        ib.return_stmt()

    func = f.get_result()

    assert func is not None
    body = _single_stmt(func.body)
    assert isinstance(body, ir.ReturnStmt)
    assert len(body.value) == 0


def test_return_with_expression():
    """Test return statement with expression."""
    ib = IRBuilder()

    with ib.function("expr_return_func") as f:
        x = f.param("x", ir.ScalarType(DataType.INT64))
        y = f.param("y", ir.ScalarType(DataType.INT64))
        f.return_type(ir.ScalarType(DataType.INT64))

        add_expr = ir.Add(x, y, DataType.INT64, ir.Span.unknown())
        ib.return_stmt(add_expr)

    func = f.get_result()

    assert func is not None
    body = _single_stmt(func.body)
    assert isinstance(body, ir.ReturnStmt)
    assert len(body.value) == 1
    assert isinstance(body.value[0], ir.Add)


def test_return_in_if_statement():
    """Test return statement inside if statement."""
    ib = IRBuilder()

    with ib.function("conditional_return") as f:
        x = f.param("x", ir.ScalarType(DataType.INT64))
        f.return_type(ir.ScalarType(DataType.INT64))

        zero = ir.ConstInt(0, DataType.INT64, ir.Span.unknown())
        one = ir.ConstInt(1, DataType.INT64, ir.Span.unknown())
        condition = ir.Gt(x, zero, DataType.INT64, ir.Span.unknown())

        with ib.if_stmt(condition) as if_builder:
            # Then branch: return 1
            ib.return_stmt(one)

            # Else branch: return 0
            if_builder.else_()
            ib.return_stmt(zero)

    func = f.get_result()

    assert func is not None
    body = _single_stmt(func.body)
    assert isinstance(body, ir.IfStmt)


def test_return_with_explicit_span():
    """Test return statement with explicit span."""
    ib = IRBuilder()
    my_span = ir.Span("test.py", 42, 1)

    with ib.function("span_return") as f:
        x = f.param("x", ir.ScalarType(DataType.INT64))
        f.return_type(ir.ScalarType(DataType.INT64))

        ib.return_stmt(x, span=my_span)

    func = f.get_result()

    assert func is not None
    body = _single_stmt(func.body)
    assert isinstance(body, ir.ReturnStmt)
    assert body.span.filename == "test.py"
    assert body.span.begin_line == 42


def test_if_return_var_with_explicit_type():
    """Test if return_var requires explicit type."""
    ib = IRBuilder()

    with ib.function("if_return_test") as f:
        x = f.param("x", ir.ScalarType(DataType.INT64))
        f.return_type(ir.ScalarType(DataType.INT64))

        zero = ir.ConstInt(0, DataType.INT64, ir.Span.unknown())
        one = ir.ConstInt(1, DataType.INT64, ir.Span.unknown())
        condition = ir.Gt(x, zero, DataType.INT64, ir.Span.unknown())

        with ib.if_stmt(condition) as if_builder:
            # Type must be provided explicitly
            if_builder.return_var("result", ir.ScalarType(DataType.INT64))

            # Then branch: yield 1
            ib.emit(ir.YieldStmt([one], ir.Span.unknown()))

            # Else branch: yield 0
            if_builder.else_()
            ib.emit(ir.YieldStmt([zero], ir.Span.unknown()))

    func = f.get_result()
    assert func is not None
    # Verify the if statement has return_vars
    body = _single_stmt(func.body)
    assert isinstance(body, ir.IfStmt)
    assert len(body.return_vars) == 1


def test_if_return_var_with_multiple_returns():
    """Test if return_var with multiple return variables."""
    ib = IRBuilder()

    with ib.function("multi_if_return_test") as f:
        x = f.param("x", ir.ScalarType(DataType.INT64))
        f.return_type(ir.ScalarType(DataType.INT64))
        f.return_type(ir.ScalarType(DataType.INT64))

        zero = ir.ConstInt(0, DataType.INT64, ir.Span.unknown())
        one = ir.ConstInt(1, DataType.INT64, ir.Span.unknown())
        two = ir.ConstInt(2, DataType.INT64, ir.Span.unknown())
        condition = ir.Gt(x, zero, DataType.INT64, ir.Span.unknown())

        with ib.if_stmt(condition) as if_builder:
            # Both return vars need explicit types
            if_builder.return_var("result1", ir.ScalarType(DataType.INT64))
            if_builder.return_var("result2", ir.ScalarType(DataType.INT64))

            # Then branch: yield two values
            ib.emit(ir.YieldStmt([one, two], ir.Span.unknown()))

            # Else branch: yield two values
            if_builder.else_()
            ib.emit(ir.YieldStmt([zero, zero], ir.Span.unknown()))

    func = f.get_result()
    assert func is not None
    # Verify the if statement has 2 return_vars
    body = _single_stmt(func.body)
    assert isinstance(body, ir.IfStmt)
    assert len(body.return_vars) == 2


def test_loop_output_single_return_var():
    """Test output() with single return variable."""
    ib = IRBuilder()

    with ib.function("loop_output_test") as f:
        n = f.param("n", ir.ScalarType(DataType.INT64))
        f.return_type(ir.ScalarType(DataType.INT64))

        i = ib.var("i", ir.ScalarType(DataType.INT64))

        with ib.for_loop(i, 0, n, 1) as loop:
            sum_iter = loop.iter_arg("sum", 0).iterVar
            loop.return_var("sum_final")

            # Loop body
            add_expr = ir.Add(sum_iter, i, DataType.INT64, ir.Span.unknown())
            ib.emit(ir.YieldStmt([add_expr], ir.Span.unknown()))

        # Get the output return variable
        result = loop.output()

        assert result.name == "sum_final"
        assert isinstance(result.type, ir.ScalarType)
        assert result.type.dtype == DataType.INDEX

        ib.return_stmt(result)

    func = f.get_result()
    assert func is not None


def test_loop_outputs_method():
    """Test outputs() method to get all return variables at once."""
    ib = IRBuilder()

    with ib.function("loop_outputs_test") as f:
        n = f.param("n", ir.ScalarType(DataType.INT64))
        f.return_type(ir.ScalarType(DataType.INT64))
        f.return_type(ir.ScalarType(DataType.INT64))

        i = ib.var("i", ir.ScalarType(DataType.INT64))

        with ib.for_loop(i, 0, n, 1) as loop:
            sum_iter = loop.iter_arg("sum", 0).iterVar
            prod_iter = loop.iter_arg("prod", 1).iterVar

            loop.return_var("sum_final")
            loop.return_var("prod_final")

            # Loop body
            add_expr = ir.Add(sum_iter, i, DataType.INT64, ir.Span.unknown())
            mul_expr = ir.Mul(prod_iter, i, DataType.INT64, ir.Span.unknown())
            ib.emit(ir.YieldStmt([add_expr, mul_expr], ir.Span.unknown()))

        # Get all outputs at once
        results = loop.outputs()

        assert len(results) == 2
        assert results[0].name == "sum_final"
        assert results[1].name == "prod_final"

        # Test unpacking
        sum_out, prod_out = loop.outputs()
        assert sum_out.name == "sum_final"
        assert prod_out.name == "prod_final"

        ib.return_stmt(results)

    func = f.get_result()
    assert func is not None


def test_loop_output_default_index():
    """Test output() with default index (0)."""
    ib = IRBuilder()

    with ib.function("default_index_test") as f:
        n = f.param("n", ir.ScalarType(DataType.INT64))
        f.return_type(ir.ScalarType(DataType.INT64))

        i = ib.var("i", ir.ScalarType(DataType.INT64))

        with ib.for_loop(i, 0, n, 1) as loop:
            sum_iter = loop.iter_arg("sum", 0).iterVar
            prod_iter = loop.iter_arg("prod", 1).iterVar

            loop.return_var("sum_final")
            loop.return_var("prod_final")

            # Loop body
            add_expr = ir.Add(sum_iter, i, DataType.INT64, ir.Span.unknown())
            mul_expr = ir.Mul(prod_iter, i, DataType.INT64, ir.Span.unknown())
            ib.emit(ir.YieldStmt([add_expr, mul_expr], ir.Span.unknown()))

        # Default index should be 0
        default_output = loop.output()
        explicit_output = loop.output(0)

        assert default_output.name == explicit_output.name
        assert default_output.name == "sum_final"

    func = f.get_result()
    assert func is not None


def test_loop_output_index_out_of_range():
    """Test that output() raises IndexError for out of range index."""
    ib = IRBuilder()

    with ib.function("out_of_range_test") as f:
        n = f.param("n", ir.ScalarType(DataType.INT64))
        f.return_type(ir.ScalarType(DataType.INT64))

        i = ib.var("i", ir.ScalarType(DataType.INT64))

        with ib.for_loop(i, 0, n, 1) as loop:
            sum_iter = loop.iter_arg("sum", 0).iterVar
            loop.return_var("sum_final")

            # Loop body
            add_expr = ir.Add(sum_iter, i, DataType.INT64, ir.Span.unknown())
            ib.emit(ir.YieldStmt([add_expr], ir.Span.unknown()))

        # Try to access index out of range
        with pytest.raises(IndexError, match="Return variable index 1 out of range"):
            loop.output(1)

    func = f.get_result()
    assert func is not None


def test_if_output_single_return_var():
    """Test output() with single return variable."""
    ib = IRBuilder()

    with ib.function("if_output_test") as f:
        x = f.param("x", ir.ScalarType(DataType.INT64))
        f.return_type(ir.ScalarType(DataType.INT64))

        zero = ir.ConstInt(0, DataType.INT64, ir.Span.unknown())
        one = ir.ConstInt(1, DataType.INT64, ir.Span.unknown())
        condition = ir.Gt(x, zero, DataType.INT64, ir.Span.unknown())

        with ib.if_stmt(condition) as if_builder:
            if_builder.return_var("result", ir.ScalarType(DataType.INT64))

            ib.emit(ir.YieldStmt([one], ir.Span.unknown()))
            if_builder.else_()
            ib.emit(ir.YieldStmt([zero], ir.Span.unknown()))

        # Get the output return variable
        result = if_builder.output()

        assert result.name == "result"
        assert isinstance(result.type, ir.ScalarType)
        assert result.type.dtype == DataType.INT64

    func = f.get_result()
    assert func is not None


def test_if_outputs_method():
    """Test outputs() method to get all return variables at once."""
    ib = IRBuilder()

    with ib.function("outputs_test") as f:
        x = f.param("x", ir.ScalarType(DataType.INT64))
        f.return_type(ir.ScalarType(DataType.INT64))
        f.return_type(ir.ScalarType(DataType.INT64))

        zero = ir.ConstInt(0, DataType.INT64, ir.Span.unknown())
        one = ir.ConstInt(1, DataType.INT64, ir.Span.unknown())
        two = ir.ConstInt(2, DataType.INT64, ir.Span.unknown())
        condition = ir.Gt(x, zero, DataType.INT64, ir.Span.unknown())

        with ib.if_stmt(condition) as if_builder:
            if_builder.return_var("result1", ir.ScalarType(DataType.INT64))
            if_builder.return_var("result2", ir.ScalarType(DataType.INT64))

            ib.emit(ir.YieldStmt([one, two], ir.Span.unknown()))
            if_builder.else_()
            ib.emit(ir.YieldStmt([zero, zero], ir.Span.unknown()))

        # Get all outputs at once
        results = if_builder.outputs()

        assert len(results) == 2
        assert results[0].name == "result1"
        assert results[1].name == "result2"

    func = f.get_result()
    assert func is not None
