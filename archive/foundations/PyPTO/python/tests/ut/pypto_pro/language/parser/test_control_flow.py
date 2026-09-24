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
"""Unit tests for control flow parsing (for loops, if statements)."""

from pypto_pro import ir
import pypto_pro.language as pl


def test_loop_without_iter_args():
    """Test loop without iter_args."""

    @pl.function
    def loop_without_iter_args(x: pl.Tensor[[64], pl.DT_FP32]) -> pl.Tensor[[64], pl.DT_FP32]:
        result: pl.Tensor[[64], pl.DT_FP32] = x
        for i in pl.range(3):
            if i > 0:
                temp = pl.tensor.mul(result, 2.0)
                result = temp
            else:
                temp = pl.tensor.add(result, 1.0)
                result = temp
        return result

    assert isinstance(loop_without_iter_args, ir.Function)


def test_loop_carried_constant_is_not_folded_in_body():
    @pl.function
    def func(n: pl.DT_INT64):
        x = 0
        for _ in pl.range(n):
            observed = x  # noqa: F841
            x += 1

    for_stmt = _find_for_stmt(func)
    body = for_stmt.body.stmts
    observed_assignment = next(
        stmt for stmt in body if isinstance(stmt, ir.AssignStmt) and stmt.var.name == "observed"
    )
    assert isinstance(observed_assignment.value, ir.Var)


def test_static_if_only_emits_selected_branch():
    @pl.function
    def func():
        if True:
            selected = 7
        else:
            selected = (1, 2)[3]  # noqa: F841

    assert all(not isinstance(stmt, ir.IfStmt) for stmt in func.body.stmts)


def _find_for_stmt(func: ir.Function) -> ir.ForStmt:
    """Helper to extract the first ForStmt from a function body."""
    body = func.body
    if isinstance(body, ir.ForStmt):
        return body
    if isinstance(body, ir.SeqStmts):
        for stmt in body.stmts:
            if isinstance(stmt, ir.ForStmt):
                return stmt
    raise AssertionError("No ForStmt found in function body")


def _find_while_stmt(func: ir.Function) -> ir.WhileStmt:
    """Helper to find WhileStmt in function body."""
    stmt = func.body
    # Handle SeqStmts wrapper
    if isinstance(stmt, ir.SeqStmts):
        for s in stmt.stmts:
            if isinstance(s, ir.WhileStmt):
                return s
            # Check nested statements
            if isinstance(s, ir.ForStmt) and isinstance(s.body, ir.SeqStmts):
                for nested in s.body.stmts:
                    if isinstance(nested, ir.WhileStmt):
                        return nested
    # Direct while statement
    if isinstance(stmt, ir.WhileStmt):
        return stmt
    raise ValueError("No WhileStmt found in function body")


def _assert_materialized_stop(func: ir.Function, expr_type) -> None:
    for_stmt = _find_for_stmt(func)
    assert isinstance(for_stmt.stop, ir.Var)
    assignment = next(
        stmt
        for stmt in func.body.stmts
        if isinstance(stmt, ir.AssignStmt) and stmt.var.name == for_stmt.stop.name
    )
    assert isinstance(assignment.value, expr_type)


def _assert_lowered_while_guard(while_stmt: ir.WhileStmt, expr_type) -> None:
    assert isinstance(while_stmt.condition, ir.ConstBool)
    assert while_stmt.condition.value
    guard_index = next(
        index for index, stmt in enumerate(while_stmt.body.stmts) if isinstance(stmt, ir.IfStmt)
    )
    assert any(
        isinstance(stmt, ir.AssignStmt) and isinstance(stmt.value, expr_type)
        for stmt in while_stmt.body.stmts[:guard_index]
    )
    guard = while_stmt.body.stmts[guard_index]
    assert any(isinstance(stmt, ir.BreakStmt) for stmt in guard.then_body.stmts)


def test_scalar_param_as_stop():
    """Test pl.range(n) where n is a DT_INT64 scalar parameter."""

    @pl.function
    def scalar_stop(n: pl.DT_INT64, x: pl.Tensor[[64], pl.DT_FP32]) -> pl.Tensor[[64], pl.DT_FP32]:
        for _ in pl.range(n):
            y: pl.Tensor[[64], pl.DT_FP32] = pl.tensor.add(x, 1.0)
        return y

    assert isinstance(scalar_stop, ir.Function)
    for_stmt = _find_for_stmt(scalar_stop)
    # stop should be a Var reference to the Scalar parameter 'n'
    assert isinstance(for_stmt.stop, ir.Var)
    assert for_stmt.stop.name == "n"
    assert isinstance(for_stmt.stop.type, ir.ScalarType)


def test_scalar_param_as_start_stop():
    """Test pl.range(0, n) where n is a DT_INT64 scalar parameter."""

    @pl.function
    def scalar_start_stop(
        n: pl.DT_INT64, x: pl.Tensor[[64], pl.DT_FP32]
    ) -> pl.Tensor[[64], pl.DT_FP32]:
        for _ in pl.range(0, n):
            y: pl.Tensor[[64], pl.DT_FP32] = pl.tensor.add(x, 1.0)
        return y

    assert isinstance(scalar_start_stop, ir.Function)
    for_stmt = _find_for_stmt(scalar_start_stop)
    assert isinstance(for_stmt.start, ir.ConstInt)
    assert isinstance(for_stmt.stop, ir.Var)
    assert for_stmt.stop.name == "n"


def test_scalar_param_as_start_stop_step():
    """Test pl.range(0, n, s) where n and s are DT_INT64 scalar parameters."""

    @pl.function
    def scalar_full_range(
        n: pl.DT_INT64,
        s: pl.DT_INT64,
        x: pl.Tensor[[64], pl.DT_FP32],
    ) -> pl.Tensor[[64], pl.DT_FP32]:
        for _ in pl.range(0, n, s):
            y: pl.Tensor[[64], pl.DT_FP32] = pl.tensor.add(x, 1.0)
        return y

    assert isinstance(scalar_full_range, ir.Function)
    for_stmt = _find_for_stmt(scalar_full_range)
    assert isinstance(for_stmt.start, ir.ConstInt)
    assert isinstance(for_stmt.stop, ir.Var)
    assert for_stmt.stop.name == "n"
    assert isinstance(for_stmt.step, ir.Var)
    assert for_stmt.step.name == "s"


def test_scalar_expression_as_stop():
    """Test pl.range(n * 2) where n is a DT_INT64 scalar parameter."""

    @pl.function
    def scalar_expr_stop(n: pl.DT_INT64, x: pl.Tensor[[64], pl.DT_FP32]) -> pl.Tensor[[64], pl.DT_FP32]:
        for _ in pl.range(n * 2):  # type: ignore[operator]
            y: pl.Tensor[[64], pl.DT_FP32] = pl.tensor.add(x, 1.0)
        return y

    assert isinstance(scalar_expr_stop, ir.Function)
    _assert_materialized_stop(scalar_expr_stop, ir.Mul)


def test_scalar_complex_expression_as_stop():
    """Test pl.range(n * 2 + 1) where n is a DT_INT64 scalar parameter."""

    @pl.function
    def scalar_complex_expr(
        n: pl.DT_INT64, x: pl.Tensor[[64], pl.DT_FP32]
    ) -> pl.Tensor[[64], pl.DT_FP32]:
        for _ in pl.range(n * 2 + 1):  # type: ignore[operator]
            y: pl.Tensor[[64], pl.DT_FP32] = pl.tensor.add(x, 1.0)
        return y

    assert isinstance(scalar_complex_expr, ir.Function)
    _assert_materialized_stop(scalar_complex_expr, ir.Add)


def test_scalar_floordiv_expression_as_stop():
    """Test pl.range(n // 4) where n is a DT_INT64 scalar parameter."""

    @pl.function
    def scalar_floordiv_expr(
        n: pl.DT_INT64, x: pl.Tensor[[64], pl.DT_FP32]
    ) -> pl.Tensor[[64], pl.DT_FP32]:
        for _ in pl.range(n // 4):  # type: ignore[operator]
            y: pl.Tensor[[64], pl.DT_FP32] = pl.tensor.add(x, 1.0)
        return y

    assert isinstance(scalar_floordiv_expr, ir.Function)
    _assert_materialized_stop(scalar_floordiv_expr, ir.FloorDiv)


def test_natural_while_loop():
    """Test natural while loop syntax (non-SSA form)."""

    @pl.function
    def natural_while(n: pl.DT_INT64) -> pl.DT_INT64:
        x: pl.DT_INT64 = 0
        while x < n:
            x = x + 1
        return x

    assert isinstance(natural_while, ir.Function)
    assert natural_while.name == "natural_while"

    # Find the while statement
    while_stmt = _find_while_stmt(natural_while)
    assert isinstance(while_stmt, ir.WhileStmt)

    # Natural syntax has no iter_args initially (ConvertToSSA adds them later)
    assert len(while_stmt.iter_args) == 0
    assert len(while_stmt.return_vars) == 0

    _assert_lowered_while_guard(while_stmt, ir.Lt)

    # Body should be present
    assert while_stmt.body is not None


def test_natural_while_loop_with_initialization():
    """Test natural while loop with explicit initialization."""

    @pl.function
    def natural_while_init(limit: pl.DT_INT64) -> pl.DT_INT64:
        counter: pl.DT_INT64 = 0
        sum_val: pl.DT_INT64 = 0
        while counter < limit:
            sum_val = sum_val + counter
            counter = counter + 1
        return sum_val

    assert isinstance(natural_while_init, ir.Function)

    # Find the while statement
    while_stmt = _find_while_stmt(natural_while_init)
    assert isinstance(while_stmt, ir.WhileStmt)

    _assert_lowered_while_guard(while_stmt, ir.Lt)

    # Natural form has no SSA iter_args
    assert len(while_stmt.iter_args) == 0


def test_while_loop_with_tensors():
    """Test while loop with tensor operations."""

    @pl.function
    def while_tensors(n: pl.DT_INT64, x: pl.Tensor[[64], pl.DT_FP32]) -> pl.Tensor[[64], pl.DT_FP32]:
        i: pl.DT_INT64 = 0
        acc: pl.Tensor[[64], pl.DT_FP32] = pl.tensor.create_tensor([64], dtype=pl.DT_FP32)
        while i < n:
            i = i + 1
            acc = pl.tensor.add(acc, x)
        return acc

    assert isinstance(while_tensors, ir.Function)

    # Find the while statement
    while_stmt = _find_while_stmt(while_tensors)
    assert isinstance(while_stmt, ir.WhileStmt)

    # Body should contain assignments
    assert while_stmt.body is not None
    if isinstance(while_stmt.body, ir.SeqStmts):
        assert len(while_stmt.body.stmts) >= 1


def test_nested_while_loops():
    """Test nested while loops."""

    @pl.function
    def nested_while(n: pl.DT_INT64) -> pl.DT_INT64:
        x: pl.DT_INT64 = 0
        while x < n:
            y: pl.DT_INT64 = 0
            while y < 3:
                y = y + 1
            x = x + 1
        return x

    assert isinstance(nested_while, ir.Function)

    # Find the outer while statement
    outer_while = _find_while_stmt(nested_while)
    assert isinstance(outer_while, ir.WhileStmt)

    # Find inner while statement in the outer while's body
    inner_while = None
    if isinstance(outer_while.body, ir.SeqStmts):
        for stmt in outer_while.body.stmts:
            if isinstance(stmt, ir.WhileStmt):
                inner_while = stmt
                break

    assert inner_while is not None, "Expected nested WhileStmt in outer while body"
    assert isinstance(inner_while, ir.WhileStmt)

    _assert_lowered_while_guard(outer_while, ir.Lt)
    _assert_lowered_while_guard(inner_while, ir.Lt)


def test_while_tensor_getval_condition_is_recomputed_in_body():
    """Tensor getval and its comparison must execute before the guard each iteration."""

    @pl.function
    def while_tensor_condition(values: pl.Tensor[[8], pl.DT_INT32], limit: pl.DT_INT64) -> pl.DT_INT64:
        i: pl.DT_INT64 = 0
        while values[i] > 0:
            i = i + 1
            if i >= limit:
                break
        return i

    while_stmt = _find_while_stmt(while_tensor_condition)
    _assert_lowered_while_guard(while_stmt, ir.Gt)


def test_while_not_condition_uses_expression_parser():
    """The source-level not condition is materialized inside the lowered loop guard."""

    @pl.function
    def while_not_condition(n: pl.DT_INT64) -> pl.DT_INT64:
        i: pl.DT_INT64 = 0
        while not i >= n:
            i = i + 1
        return i

    while_stmt = _find_while_stmt(while_not_condition)
    _assert_lowered_while_guard(while_stmt, ir.Not)


def test_while_with_multiple_updates():
    """Test while loop with multiple variable updates."""

    @pl.function
    def while_multi_update(n: pl.DT_INT64) -> pl.DT_INT64:
        x: pl.DT_INT64 = 0
        y: pl.DT_INT64 = 1

        while x < n:
            x = x + 1
            y = y * 2

        return y

    assert isinstance(while_multi_update, ir.Function)

    # Find the while statement
    while_stmt = _find_while_stmt(while_multi_update)
    assert isinstance(while_stmt, ir.WhileStmt)

    # Body should contain multiple assignments
    assert while_stmt.body is not None
    if isinstance(while_stmt.body, ir.SeqStmts):
        # Should have at least 2 statements (x = x + 1, y = y * 2)
        assert len(while_stmt.body.stmts) >= 2


closure_allowed = [5, 6]


def test_in_operator_with_kernel_local_list():
    """``x in <list>`` accepts a list bound inside the function, not just a literal."""

    @pl.function
    def func(n: pl.DT_INT64) -> pl.DT_INT64:
        allowed = [0, 1]
        total: pl.DT_INT64 = 0
        for i in pl.range(n):
            if i in allowed:
                total = total + 1
        return total

    assert isinstance(func, ir.Function)


def test_in_operator_kernel_local_list_shadows_closure():
    """A function-local list wins over a same-named closure list, as in Python."""

    @pl.function
    def func(n: pl.DT_INT64) -> pl.DT_INT64:
        closure_allowed = [0, 1]  # noqa: F841 - shadows the module-level list
        total: pl.DT_INT64 = 0
        for i in pl.range(n):
            if i in closure_allowed:
                total = total + 1
        return total

    text = str(func)
    # The local values (0, 1) drive the eq-chain; the closure values (5, 6) do not.
    assert "== 5" not in text and "== 6" not in text, text


def test_in_operator_with_closure_list_still_works():
    @pl.function
    def func(n: pl.DT_INT64) -> pl.DT_INT64:
        total: pl.DT_INT64 = 0
        for i in pl.range(n):
            if i in closure_allowed:
                total = total + 1
        return total

    assert isinstance(func, ir.Function)


_FP32_INT32 = (pl.DT_FP32, pl.DT_INT32)


def _addr_for(dtype):
    """Plain Python helper the parser inlines when it appears in a kwarg."""
    return 0x8000 if dtype in _FP32_INT32 else 0x4000


def _addr_for_not_in(dtype):
    return 0x1000 if dtype not in _FP32_INT32 else 0x2000


def _tile_group_addr_ir(dtype, helper) -> str:
    @pl.kernel(auto_mutex=True)
    def k(a: pl.Tensor[[128, 128], pl.DT_FP16]):
        tt = pl.TileType(shape=[128, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
        dt = dtype
        g = pl.make_tile_group(type=tt, addrs=helper(dt), mutex_ids=[0, 1])
        pl.load(g.next(), a, [0, 0])

    return str(k.parse_target_program(ir.SectionKind.Vector)[0])


def test_enum_in_tuple_folds_to_selected_branch():
    """``dtype in (pl.DT_FP32, ...)`` has no runtime form; it folds at parse time.

    An enum operand cannot be lowered to an IR constant, so the membership test
    must be decided while parsing and the enclosing ternary collapsed.
    """
    assert "target_memory=Mat" in _tile_group_addr_ir(pl.DT_FP32, _addr_for)
    assert "memref_addr=32768" in _tile_group_addr_ir(pl.DT_FP32, _addr_for)
    assert "memref_addr=16384" in _tile_group_addr_ir(pl.DT_FP16, _addr_for)


def test_enum_not_in_tuple_folds_to_selected_branch():
    assert "memref_addr=4096" in _tile_group_addr_ir(pl.DT_FP16, _addr_for_not_in)
    assert "memref_addr=8192" in _tile_group_addr_ir(pl.DT_FP32, _addr_for_not_in)
