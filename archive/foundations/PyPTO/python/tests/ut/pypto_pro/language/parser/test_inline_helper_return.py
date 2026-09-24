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

from pypto_pro import ir
import pypto_pro.language as pl
from pypto_pro.language.parser.diagnostics import ParserSyntaxError, ParserTypeError
import pytest


def _walk_statements(stmt):
    yield stmt
    if isinstance(stmt, ir.SeqStmts):
        for child in stmt.stmts:
            yield from _walk_statements(child)
    elif isinstance(stmt, (ir.ForStmt, ir.WhileStmt)):
        yield from _walk_statements(stmt.body)
    elif isinstance(stmt, ir.IfStmt):
        yield from _walk_statements(stmt.then_body)
        if stmt.else_body is not None:
            yield from _walk_statements(stmt.else_body)


def test_inline_helper_multiple_value_returns_use_one_wrapper():
    def choose(value):
        if value > 0:
            return value + 1
        return value - 1

    @pl.function
    def caller(value: pl.DT_INT64) -> pl.DT_INT64:
        return choose(value)

    statements = list(_walk_statements(caller.body))
    helper_loops = [stmt for stmt in statements if isinstance(stmt, ir.WhileStmt)]
    helper_returns = [stmt for stmt in statements if isinstance(stmt, ir.ReturnStmt)]
    return_assigns = [
        stmt
        for stmt in statements
        if isinstance(stmt, ir.AssignStmt) and stmt.var.name.startswith("__inline_0_return_val")
    ]

    assert len(helper_loops) == 1
    assert len(helper_returns) == 1
    assert len(return_assigns) == 3  # Unknown init plus the two concrete return assignments.
    assert len({stmt.var.name for stmt in return_assigns}) == 1


def test_inline_helper_return_inside_loop_adds_propagation_guard():
    def choose(value):
        index = 0
        while index < value:
            if index >= 2:
                return index
            index = index + 1
        return value

    @pl.function
    def caller(value: pl.DT_INT64) -> pl.DT_INT64:
        return choose(value)

    wrapper = next(stmt for stmt in caller.body.stmts if isinstance(stmt, ir.WhileStmt))
    nested_loop_index = next(i for i, stmt in enumerate(wrapper.body.stmts) if isinstance(stmt, ir.WhileStmt))
    guard = wrapper.body.stmts[nested_loop_index + 1]

    assert isinstance(guard, ir.IfStmt)
    assert isinstance(guard.condition, ir.Var)
    assert guard.condition.name == "__inline_0_returned"
    assert any(isinstance(stmt, ir.BreakStmt) for stmt in _walk_statements(guard.then_body))


def test_inline_helper_loop_guard_folds_constant_false():
    def inner(value):
        if value > 0:
            return value
        return value + 1

    def outer(value):
        result = value
        for _ in pl.range(1):
            result = inner(value)
        # The second return is what puts `outer` on the lowering path at all. It has to come
        # after the loop -- an early return ahead of the loop would leave `returned` non-constant
        # there and the guard would survive -- and one statement clear of it, so that an `if`
        # sitting right behind the loop can only be the guard itself.
        doubled = result + result
        if doubled > 0:
            return doubled
        return result

    @pl.function
    def caller(value: pl.DT_INT64) -> pl.DT_INT64:
        return outer(value)

    outer_wrapper = next(stmt for stmt in caller.body.stmts if isinstance(stmt, ir.WhileStmt))
    outer_for_index = next(i for i, stmt in enumerate(outer_wrapper.body.stmts) if isinstance(stmt, ir.ForStmt))
    statement_after_for = outer_wrapper.body.stmts[outer_for_index + 1]

    assert not isinstance(statement_after_for, ir.IfStmt)


def test_inline_helper_bare_returns_share_return_state():
    def stop_early(value):
        if value > 0:
            return
        return None

    @pl.function
    def caller(value: pl.DT_INT64) -> pl.DT_INT64:
        stop_early(value)
        return value

    statements = list(_walk_statements(caller.body))
    assert sum(isinstance(stmt, ir.WhileStmt) for stmt in statements) == 1
    assert sum(
        isinstance(stmt, ir.AssignStmt) and stmt.var.name == "__inline_0_return_val" for stmt in statements
    ) == 1


def test_inline_helper_mixed_bare_and_value_returns_share_return_val():
    def choose(value):
        if value > 0:
            return value
        return

    @pl.function
    def caller(value: pl.DT_INT64):
        return choose(value)

    statements = list(_walk_statements(caller.body))
    assert sum(
        isinstance(stmt, ir.AssignStmt) and stmt.var.name == "__inline_0_return_val" for stmt in statements
    ) == 2


def test_inline_helper_without_return_uses_default_return_val():
    def update(value):
        result = value
        for _ in pl.range(1):
            result = result + 1

    @pl.function
    def caller(value: pl.DT_INT64):
        return update(value)

    statements = list(_walk_statements(caller.body))
    wrapper = next(stmt for stmt in caller.body.stmts if isinstance(stmt, ir.WhileStmt))
    loop_index = next(i for i, stmt in enumerate(wrapper.body.stmts) if isinstance(stmt, ir.ForStmt))
    assert not isinstance(wrapper.body.stmts[loop_index + 1], ir.IfStmt)
    assert sum(
        isinstance(stmt, ir.AssignStmt) and stmt.var.name == "__inline_0_return_val" for stmt in statements
    ) == 1


def test_inline_helper_value_return_can_fall_through():
    def choose(value):
        if value > 0:
            return value

    @pl.function
    def caller(value: pl.DT_INT64) -> pl.DT_INT64:
        return choose(value)

    assert any(isinstance(stmt, ir.WhileStmt) for stmt in caller.body.stmts)


def test_kernel_uses_runtime_tile_valid_shape_ir():
    @pl.function(type=pl.FunctionType.InCore)
    def caller():
        tile_type = pl.TileType(shape=[16, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        tile = pl.make_tile(tile_type, addr=0, size=1024)
        _kernel_rows = tile.valid_shape[0]
        _kernel_cols = tile.valid_shape[1]

    function_ir = str(caller)
    assert function_ir.count("block.tile_valid_shape") == 2


@pytest.mark.parametrize("index", ["True", "0.0", "2", "-3"])
def test_tile_valid_shape_rejects_invalid_index(index):
    code = f"""
@pl.function(type=pl.FunctionType.InCore)
def invalid_valid_shape():
    tile_type = pl.TileType(shape=[16, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile = pl.make_tile(tile_type, addr=0, size=1024)
    value = tile.valid_shape[{index}]
"""
    with pytest.raises((ParserSyntaxError, ParserTypeError)):
        pl.parse(code)


def test_vector_function_rejects_explicit_return():
    @pl.vector_function
    def invalid_vf():
        return

    with pytest.raises(ParserSyntaxError, match="cannot contain return"):

        @pl.function
        def caller(value: pl.DT_INT64) -> pl.DT_INT64:
            invalid_vf()
            return value


def test_vector_function_rejects_non_vector_helper_call():
    def helper(value):
        return value

    @pl.vector_function
    def invalid_vf(value):
        helper(value)

    with pytest.raises(ParserSyntaxError, match="cannot call non-vector inline function 'helper'"):

        @pl.function
        def caller(value: pl.DT_INT64) -> pl.DT_INT64:
            invalid_vf(value)
            return value
