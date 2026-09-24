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
"""Tests for system operation DSL parsing and unified printing."""

import pypto_pro
import pypto_pro.language as pl
from pypto_pro.language.parser.diagnostics import ParserSyntaxError, ParserTypeError
import pytest


def _const_int(value: int):
    return pypto_pro.ir.ConstInt(value, pypto_pro.ir.DataType.INDEX, pypto_pro.ir.Span.unknown())


def test_sync_src_print_style():
    """Test unified printing for pl.system.sync_src."""

    @pl.program
    class Before:
        @pl.function
        def main(self, x: pl.Tensor[[64], pl.DT_FP32]) -> pl.Tensor[[64], pl.DT_FP32]:
            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
            return x

    printed = pypto_pro.ir.python_print(Before)
    assert "ir.call @system.sync_src_dyn(" in printed
    assert "set_pipe" in printed
    assert "MTE2" in printed
    assert "wait_pipe" in printed
    assert "PipeType.V" in printed
    assert "0" in printed


def test_bar_all_print_style():
    """Test unified printing for pl.system.bar_all."""

    @pl.program
    class Before:
        @pl.function
        def main(self, x: pl.Tensor[[64], pl.DT_FP32]) -> pl.Tensor[[64], pl.DT_FP32]:
            pl.system.bar_all()
            return x

    printed = pypto_pro.ir.python_print(Before)
    assert "ir.call @system.bar_all()" in printed


def test_multiple_system_ops_print_style():
    """Test unified printing with multiple system ops in a single function."""

    @pl.program
    class Before:
        @pl.function
        def main(self, x: pl.Tensor[[64], pl.DT_FP32]) -> pl.Tensor[[64], pl.DT_FP32]:
            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
            pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
            pl.system.bar_all()
            return x

    printed = pypto_pro.ir.python_print(Before)
    assert "ir.call @system.sync_src_dyn(" in printed
    assert "ir.call @system.sync_dst_dyn(" in printed
    assert "ir.call @system.bar_all()" in printed


def test_sync_with_different_pipe_types():
    """Test sync ops with various PipeType enum values."""

    @pl.program
    class Before:
        @pl.function
        def main(self, x: pl.Tensor[[64], pl.DT_FP32]) -> pl.Tensor[[64], pl.DT_FP32]:
            pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=1)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.S, event_id=2)
            return x

    printed = pypto_pro.ir.python_print(Before)
    assert "ir.PipeType.MTE1" in printed
    assert "ir.PipeType.M" in printed
    assert "ir.PipeType.MTE3" in printed
    assert "ir.PipeType.S" in printed


def test_sync_src_pipe_kwarg_rejects_plain_integer():
    with pytest.raises(ParserTypeError, match="'set_pipe' expects an enum value"):

        @pl.function
        def func():
            pl.system.sync_src(set_pipe=1, wait_pipe=pl.PipeType.V, event_id=0)


def test_sync_dst_pipe_kwarg_rejects_plain_integer():
    with pytest.raises(ParserTypeError, match="'wait_pipe' expects an enum value"):

        @pl.function
        def func():
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=5, event_id=0)


def test_mutex_lock_pipe_kwarg_rejects_plain_integer():
    with pytest.raises(ParserTypeError, match="'pipe' expects an enum value"):

        @pl.function
        def func():
            pl.system.mutex_lock(pipe=5, mutex_id=0)


@pytest.mark.parametrize("builder", [pl.system.sync_src, pl.system.sync_dst])
@pytest.mark.parametrize("set_pipe", [1, pl.SyncCoreType.AIV_ONLY])
def test_sync_ir_builder_requires_pipe_type(builder, set_pipe):
    with pytest.raises(TypeError, match="set_pipe must be a PipeType"):
        builder(set_pipe=set_pipe, wait_pipe=pl.PipeType.V, event_id=0)


@pytest.mark.parametrize("builder", [pl.system.mutex_lock, pl.system.mutex_unlock])
@pytest.mark.parametrize("pipe", [1, pl.SyncCoreType.AIV_ONLY])
def test_mutex_ir_builder_requires_pipe_type(builder, pipe):
    with pytest.raises(TypeError, match="pipe must be a PipeType"):
        builder(pipe=pipe, mutex_id=0)


def test_sync_all_core_type_rejects_plain_integer():
    with pytest.raises(ParserTypeError, match="'core_type' expects an enum value"):

        @pl.function
        def func():
            pl.system.sync_all(core_type=2)


def test_sync_all_mode_rejects_plain_integer():
    with pytest.raises(ParserTypeError, match="'mode' expects an enum value"):

        @pl.function
        def func():
            pl.system.sync_all(mode=0)


@pytest.mark.parametrize("builder", [pl.system.sync_src, pl.system.sync_dst])
@pytest.mark.parametrize("event_id", [True, False])
def test_sync_event_id_rejects_bool(builder, event_id):
    with pytest.raises(TypeError, match="event_id must be a Python int or an integer scalar expression"):
        builder(
            set_pipe=pl.PipeType.MTE2,
            wait_pipe=pl.PipeType.V,
            event_id=event_id,
        )


@pytest.mark.parametrize("builder", [pl.system.sync_src, pl.system.sync_dst])
def test_sync_ir_builder_normalizes_python_int_id(builder):
    call = builder(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=3)

    assert isinstance(call.args[0], pypto_pro.ir.ConstInt)
    assert call.args[0].value == 3


@pytest.mark.parametrize("builder", [pl.system.mutex_lock, pl.system.mutex_unlock])
@pytest.mark.parametrize("mutex_id", [True, False])
def test_mutex_id_rejects_bool(builder, mutex_id):
    with pytest.raises(TypeError, match="mutex_id must be a Python int or an integer scalar expression"):
        builder(pipe=pl.PipeType.MTE2, mutex_id=mutex_id)


@pytest.mark.parametrize("builder", [pl.system.mutex_lock, pl.system.mutex_unlock])
def test_mutex_ir_builder_normalizes_python_int_id(builder):
    call = builder(pipe=pl.PipeType.MTE2, mutex_id=0)

    assert isinstance(call.args[0], pypto_pro.ir.ConstInt)
    assert call.args[0].value == 0


@pytest.mark.parametrize("builder", [pl.system.sync_src, pl.system.sync_dst])
@pytest.mark.parametrize("event_id", [-1, 8])
def test_sync_static_event_id_range_is_validated_by_frontend(builder, event_id):
    with pytest.raises(ValueError, match=r"event_id must be in \[0, 8\)"):
        builder(
            set_pipe=pl.PipeType.MTE2,
            wait_pipe=pl.PipeType.V,
            event_id=event_id,
        )


@pytest.mark.parametrize("builder", [pl.system.sync_src, pl.system.sync_dst])
def test_sync_rejects_equal_pipes_in_frontend(builder):
    with pytest.raises(ValueError, match="set_pipe and wait_pipe must differ"):
        builder(
            set_pipe=pl.PipeType.MTE2,
            wait_pipe=pl.PipeType.MTE2,
            event_id=0,
        )


@pytest.mark.parametrize("builder", [pl.system.sync_src, pl.system.sync_dst])
@pytest.mark.parametrize(
    ("set_pipe", "wait_pipe"),
    [(pl.PipeType.ALL, pl.PipeType.V), (pl.PipeType.MTE2, pl.PipeType.ALL)],
)
def test_sync_rejects_all_pipe_in_frontend(builder, set_pipe, wait_pipe):
    with pytest.raises(ValueError, match="must identify one concrete pipe"):
        builder(set_pipe=set_pipe, wait_pipe=wait_pipe, event_id=0)


@pytest.mark.parametrize("builder", [pl.system.mutex_lock, pl.system.mutex_unlock])
@pytest.mark.parametrize("mutex_id", [-1, 32])
def test_mutex_static_id_range_is_validated_by_frontend(builder, mutex_id):
    with pytest.raises(ValueError, match=r"mutex_id must be in \[0, 32\)"):
        builder(pipe=pl.PipeType.MTE2, mutex_id=_const_int(mutex_id))


@pytest.mark.parametrize("builder", [pl.system.mutex_lock, pl.system.mutex_unlock])
def test_mutex_rejects_all_pipe_in_frontend(builder):
    with pytest.raises(ValueError, match="pipe must identify one concrete pipe"):
        builder(pipe=pl.PipeType.ALL, mutex_id=_const_int(0))


@pytest.mark.parametrize("builder", [pl.system.mutex_lock, pl.system.mutex_unlock])
@pytest.mark.parametrize("max_mutex_id", [0, 33])
@pytest.mark.parametrize("dynamic_id", [False, True], ids=["static-id", "dynamic-id"])
def test_mutex_max_id_range_is_validated_by_frontend(builder, max_mutex_id, dynamic_id):
    span = pypto_pro.ir.Span.unknown()
    mutex_id = (
        pypto_pro.ir.Var("mutex_id", pypto_pro.ir.ScalarType(pypto_pro.ir.DataType.INT64), span)
        if dynamic_id
        else _const_int(0)
    )

    with pytest.raises(ValueError, match=r"max_mutex_id must be in \[1, 32\]"):
        builder(pipe=pl.PipeType.MTE2, mutex_id=mutex_id, max_mutex_id=max_mutex_id)


@pytest.mark.parametrize("builder", [pl.system.mutex_lock, pl.system.mutex_unlock])
@pytest.mark.parametrize("mutex_ids", [[], (), [-1], [32]])
@pytest.mark.parametrize("dynamic_id", [False, True], ids=["static-id", "dynamic-id"])
def test_mutex_candidate_ids_are_validated_by_frontend(builder, mutex_ids, dynamic_id):
    span = pypto_pro.ir.Span.unknown()
    mutex_id = (
        pypto_pro.ir.Var("mutex_id", pypto_pro.ir.ScalarType(pypto_pro.ir.DataType.INT64), span)
        if dynamic_id
        else _const_int(0)
    )

    with pytest.raises(ValueError, match="mutex_ids"):
        builder(pipe=pl.PipeType.MTE2, mutex_id=mutex_id, mutex_ids=mutex_ids)


def test_empty_mutex_ids_rejected_after_kwarg_resolution():
    with pytest.raises(ParserSyntaxError, match="mutex_ids must not be empty"):

        @pl.function
        def func():
            pl.system.mutex_lock(pipe=pl.PipeType.MTE2, mutex_id=0, mutex_ids=[])


@pytest.mark.parametrize("builder", [pl.system.mutex_lock, pl.system.mutex_unlock])
def test_static_mutex_preserves_max_mutex_id_attribute(builder):
    call = builder(pipe=pl.PipeType.MTE2, mutex_id=_const_int(0), max_mutex_id=4)

    assert call.kwargs["max_mutex_id"] == 4


@pytest.mark.parametrize("builder", [pl.system.mutex_lock, pl.system.mutex_unlock])
def test_static_mutex_preserves_candidate_ids_attribute(builder):
    call = builder(pipe=pl.PipeType.MTE2, mutex_id=_const_int(8), mutex_ids=(8, 9))

    assert 'mutex_ids="mutex_ids": [8, 9]' in str(call)


def test_sync_all_hard_mode_rejects_workspaces_in_frontend():
    with pytest.raises(ValueError, match="Hard mode sync_all does not accept workspace arguments"):
        pl.system.sync_all([0], mode=pl.SyncAllMode.HARD)


def test_sync_all_soft_mode_requires_workspaces_in_frontend():
    with pytest.raises(ValueError, match="Soft mode sync_all requires workspaces list"):
        pl.system.sync_all(mode=pl.SyncAllMode.SOFT)


def test_complex_integer_event_id_expression_is_accepted():
    @pl.function
    def func(base: pl.DT_INT64):
        event_id = (base * 3 + 1) % 8
        pl.system.sync_src(
            set_pipe=pl.PipeType.MTE2,
            wait_pipe=pl.PipeType.V,
            event_id=event_id,
        )

    assert "system.sync_src_dyn" in str(func)


def test_constant_integer_event_id_expression_is_folded_to_operand():
    @pl.function
    def func():
        pl.system.sync_src(
            set_pipe=pl.PipeType.MTE2,
            wait_pipe=pl.PipeType.V,
            event_id=(1 + 2) % 8,
        )
        pl.system.sync_dst(
            set_pipe=pl.PipeType.MTE2,
            wait_pipe=pl.PipeType.V,
            event_id=(1 + 2) % 8,
        )

    calls = [stmt.expr for stmt in func.body.stmts if isinstance(stmt, pypto_pro.ir.EvalStmt)]
    assert [call.name for call in calls] == ["system.sync_src_dyn", "system.sync_dst_dyn"]
    for call in calls:
        assert isinstance(call.args[0], pypto_pro.ir.ConstInt)
        assert call.args[0].value == 3


def test_constant_integer_mutex_id_expression_is_folded_to_operand():
    @pl.function
    def func():
        pl.system.mutex_lock(
            pipe=pl.PipeType.MTE2,
            mutex_id=(5 * 2 + 1) % 32,
            max_mutex_id=12,
        )
        pl.system.mutex_unlock(
            pipe=pl.PipeType.MTE2,
            mutex_id=(5 * 2 + 1) % 32,
            max_mutex_id=12,
        )

    calls = [stmt.expr for stmt in func.body.stmts if isinstance(stmt, pypto_pro.ir.EvalStmt)]
    assert [call.name for call in calls] == ["system.mutex_lock_dyn", "system.mutex_unlock_dyn"]
    for call in calls:
        assert isinstance(call.args[0], pypto_pro.ir.ConstInt)
        assert call.args[0].value == 11


def test_bool_event_id_expression_is_rejected_by_parser():
    with pytest.raises(ParserSyntaxError, match="event_id must be an integer scalar expression"):

        @pl.function
        def func(event_id: pl.DT_BOOL):
            pl.system.sync_src(
                set_pipe=pl.PipeType.MTE2,
                wait_pipe=pl.PipeType.V,
                event_id=event_id,
            )


def test_bool_mutex_id_expression_is_rejected_by_parser():
    with pytest.raises(ParserSyntaxError, match="mutex_id must be an integer scalar expression"):

        @pl.function
        def func(mutex_id: pl.DT_BOOL):
            pl.system.mutex_lock(
                pipe=pl.PipeType.MTE2,
                mutex_id=mutex_id,
            )


def test_complex_float_event_id_expression_is_rejected_by_parser():
    with pytest.raises(ParserSyntaxError, match="event_id must be an integer scalar expression"):

        @pl.function
        def func(base: pl.DT_INT64):
            event_id = (base * 3 + 1) / 2
            pl.system.sync_src(
                set_pipe=pl.PipeType.MTE2,
                wait_pipe=pl.PipeType.V,
                event_id=event_id,
            )


def test_complex_float_mutex_id_expression_is_rejected_by_parser():
    with pytest.raises(ParserSyntaxError, match="mutex_id must be an integer scalar expression"):

        @pl.function
        def func(base: pl.DT_INT64):
            mutex_id = (base * 3 + 1) / 2
            pl.system.mutex_lock(
                pipe=pl.PipeType.MTE2,
                mutex_id=mutex_id,
                max_mutex_id=8,
            )


def test_control_flow_integer_event_id_expression_is_accepted():
    @pl.function
    def func(base: pl.DT_INT64):
        event_id = base % 8
        if base > 4:
            event_id = (base * 3 + 1) % 8
        else:
            event_id = (base * 5 + 1) % 8
        pl.system.sync_src(
            set_pipe=pl.PipeType.MTE2,
            wait_pipe=pl.PipeType.V,
            event_id=event_id,
        )

    assert "system.sync_src_dyn" in str(func)


def test_control_flow_float_event_id_expression_is_rejected_by_parser():
    with pytest.raises(ParserSyntaxError, match="event_id must be an integer scalar expression"):

        @pl.function
        def func(base: pl.DT_INT64):
            event_id = (base + 1) / 2
            if base > 4:
                event_id = (base * 3 + 1) / 2
            else:
                event_id = (base * 5 + 1) / 2
            pl.system.sync_src(
                set_pipe=pl.PipeType.MTE2,
                wait_pipe=pl.PipeType.V,
                event_id=event_id,
            )


@pytest.mark.parametrize("builder", [pl.system.mutex_lock, pl.system.mutex_unlock])
def test_mutex_rejects_removed_mode_kwarg(builder):
    with pytest.raises(TypeError, match="unexpected keyword argument 'mode'"):
        builder(pipe=pl.PipeType.MTE2, mutex_id=_const_int(0), mode=0)


@pytest.mark.parametrize("builder", [pl.system.mutex_lock, pl.system.mutex_unlock])
@pytest.mark.parametrize("mutex_ids", [[True], [0, False], (True, 1)])
def test_mutex_candidate_ids_reject_bool(builder, mutex_ids):
    span = pypto_pro.ir.Span.unknown()
    dynamic_id = pypto_pro.ir.Var(
        "mutex_id", pypto_pro.ir.ScalarType(pypto_pro.ir.DataType.INT64), span
    )

    with pytest.raises(TypeError, match=r"mutex_ids\[\d+\] must be a Python int"):
        builder(pipe=pl.PipeType.MTE2, mutex_id=dynamic_id, mutex_ids=mutex_ids)


@pytest.mark.parametrize("builder", [pl.system.mutex_lock, pl.system.mutex_unlock])
@pytest.mark.parametrize(
    "mutex_ids",
    [1, {0, 1}, (mutex_id for mutex_id in (0, 1))],
    ids=["integer", "set", "generator"],
)
def test_mutex_candidate_ids_reject_wrong_container_type(builder, mutex_ids):
    with pytest.raises(TypeError, match="mutex_ids must be a list, tuple, or None"):
        builder(pipe=pl.PipeType.MTE2, mutex_id=_const_int(0), mutex_ids=mutex_ids)


def test_dcci_gm_tensor_with_offset_print_style():
    """Test unified printing for pl.system.dcci with GM tensor and offset."""

    @pl.program
    class Before:
        @pl.function
        def main(self, x: pl.Tensor[[16, 16], pl.DT_FP32]) -> pl.Tensor[[16, 16], pl.DT_FP32]:
            pl.system.dcci(x, [0, 0], cache_line=pl.CacheLine.SINGLE_CACHE_LINE, dst=pl.DcciDst.CACHELINE_OUT)
            return x

    printed = pypto_pro.ir.python_print(Before)
    assert "ir.call @system.dcci(" in printed
    assert "cache_line=" in printed
    assert "dst=" in printed


def test_dcci_gm_tensor_rejects_float_scalar_offset():
    """Test pl.system.dcci rejects float scalar offset for GM tensor."""

    with pytest.raises(RuntimeError, match="scalar integer element offset"):

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[16, 16], pl.DT_FP32]) -> pl.Tensor[[16, 16], pl.DT_FP32]:
                pl.system.dcci(x, 1.5)
                return x


def test_dcci_gm_tensor_rejects_float_tuple_offset():
    """Test pl.system.dcci rejects float tuple element offset for GM tensor."""

    with pytest.raises(RuntimeError, match="per-dimension list/tuple"):

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[16, 16], pl.DT_FP32]) -> pl.Tensor[[16, 16], pl.DT_FP32]:
                pl.system.dcci(x, [1.5, 0])
                return x


def test_dcci_with_tuple_offset_print_style():
    """Test unified printing for pl.system.dcci with tuple offset."""

    @pl.program
    class Before:
        @pl.function
        def main(self, x: pl.Tensor[[16, 16], pl.DT_FP32]) -> pl.Tensor[[16, 16], pl.DT_FP32]:
            pl.system.dcci(x, (1, 2))
            return x

    printed = pypto_pro.ir.python_print(Before)
    assert "ir.call @system.dcci(" in printed
