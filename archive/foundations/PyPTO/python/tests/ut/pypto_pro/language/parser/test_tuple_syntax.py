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
"""Tests for tuple literal and subscript syntax in parser."""

from pypto_pro import ir
import pypto_pro.language as pl
from pypto_pro.language.parser.diagnostics import ParserTypeError
import pytest


def test_parse_empty_tuple():
    """Test parsing empty tuple literal."""

    @pl.function
    def func():
        _ = ()

    # Verify function was created
    assert func is not None
    assert isinstance(func, ir.Function)


def test_parse_tuple_with_two_elements():
    """Test parsing tuple with two elements."""

    @pl.function
    def func(x: pl.Tensor[[10], pl.DT_FP32], y: pl.DT_INT64):
        _ = (x, y)

    assert func is not None
    assert isinstance(func, ir.Function)


def test_parse_tuple_with_constants():
    """Test parsing tuple with constant values."""

    @pl.function
    def func():
        _ = (1, 2, 3)

    assert func is not None
    assert isinstance(func, ir.Function)


def test_parse_nested_tuple():
    """Test parsing nested tuples."""

    @pl.function
    def func(x: pl.DT_INT64):
        inner = (x, x)
        _ = (inner, x)

    assert func is not None
    assert isinstance(func, ir.Function)


def test_parse_singleton_tuple():
    """Test parsing single element tuple."""

    @pl.function
    def func(x: pl.DT_INT64):
        _ = (x,)

    assert func is not None
    assert isinstance(func, ir.Function)


def test_parse_nested_subscript():
    """Test parsing nested tuple subscript."""

    @pl.function
    def func(x: pl.DT_INT64, y: pl.DT_FP32):
        inner = (x, x)
        nested = (inner, y)
        _ = nested[0]
        _ = nested[0][1]

    assert func is not None
    assert isinstance(func, ir.Function)


def test_static_subscript_of_let_bound_tuple_is_folded():
    """Immutable tuples retain static-element folding after let binding."""

    @pl.function
    def func(x: pl.DT_INT64):
        values = (x, 7)
        selected = values[1]  # noqa: F841

    selected = next(
        stmt for stmt in func.body.stmts
        if isinstance(stmt, ir.AssignStmt) and stmt.var.name == "selected"
    )
    assert isinstance(selected.value, ir.ConstInt)


def test_struct_array_static_subscript_remains_getitem():
    """Struct arrays must share CCE's backing array for static and dynamic access."""

    @pl.function
    def func():
        slots = pl.struct_array(2, "Slot", value=0)
        slot = slots[0]  # noqa: F841

    slot = next(
        stmt for stmt in func.body.stmts
        if isinstance(stmt, ir.AssignStmt) and stmt.var.name == "slot"
    )
    assert isinstance(slot.value, ir.GetItemExpr)
    assert isinstance(slot.value.value, ir.Var)
    assert slot.value.value.name == "slots"


def test_named_tuple_static_field_is_folded_with_runtime_fields():
    """A static named-tuple field folds even when another field is runtime."""

    @pl.function
    def func(x: pl.DT_INT64):
        info = pl.make_tuple(flag=False, size=x)
        if info.flag:
            selected = (1, 2)[3]
        else:
            selected = info.size  # noqa: F841

    assert all(not isinstance(stmt, ir.IfStmt) for stmt in func.body.stmts)


def test_variable_index_homogeneous_tuple():
    """Variable index on a homogeneous tuple generates valid IR."""

    @pl.function
    def func(x: pl.DT_INT64, y: pl.DT_INT64, idx: pl.DT_INT64):
        my_tuple = (x, y)
        _ = my_tuple[idx]

    assert func is not None
    assert isinstance(func, ir.Function)


def test_variable_index_heterogeneous_tuple_raises():
    """Variable index on heterogeneous tuple raises an error."""
    with pytest.raises(Exception):

        @pl.function
        def func(x: pl.DT_INT64, y: pl.DT_FP32, idx: pl.DT_INT64):
            my_tuple = (x, y)
            _ = my_tuple[idx]


def test_variable_index_generates_get_item_expr():
    """Variable index generates a GetItemExpr with a non-constant slice in the IR.

    The CCE codegen handles this via array pre-declaration.
    """

    @pl.function
    def func(x: pl.DT_INT64, y: pl.DT_INT64, idx: pl.DT_INT64):
        my_tuple = (x, y)
        _ = my_tuple[idx]

    assert func is not None
    body_stmts = func.body.stmts
    # Find AssignStmts whose RHS is a GetItemExpr with a dynamic (non-ConstInt) slice
    dyn_gi = []
    for s in body_stmts:
        if (isinstance(s, ir.AssignStmt)
                and isinstance(s.value, ir.GetItemExpr)
                and not isinstance(s.value.slice, ir.ConstInt)):
            dyn_gi.append(s.value)
    assert len(dyn_gi) >= 1, "Expected at least one GetItemExpr with dynamic slice"


def test_constant_tuple_dynamic_index_uses_folded_make_tuple_base():
    @pl.function
    def func(idx: pl.DT_INT64):
        event_ids = (3, 7)
        _ = event_ids[idx]

    body_stmts = func.body.stmts
    tuple_assignments = [
        stmt for stmt in body_stmts
        if isinstance(stmt, ir.AssignStmt) and isinstance(stmt.value, ir.MakeTuple)
    ]
    assert len(tuple_assignments) == 1
    dynamic_reads = []
    for stmt in body_stmts:
        if (isinstance(stmt, ir.AssignStmt)
                and isinstance(stmt.value, ir.GetItemExpr)
                and not isinstance(stmt.value.slice, ir.ConstInt)):
            dynamic_reads.append(stmt.value)
    assert len(dynamic_reads) == 1
    assert isinstance(dynamic_reads[0].value, ir.MakeTuple)
    assert dynamic_reads[0].value is tuple_assignments[0].value


def test_dynamic_index_keeps_folded_tuple_through_base_expression():
    @pl.function
    def func(x: pl.DT_INT64, y: pl.DT_INT64, idx: pl.DT_INT64):
        primary = (x, y)
        fallback = (x, y)
        pair = (primary, fallback)
        selector = 0
        selected = pair[selector][idx]  # noqa: F841

    dynamic_read = None
    for stmt in func.body.stmts:
        if (isinstance(stmt, ir.AssignStmt)
                and stmt.var.name == "selected"
                and isinstance(stmt.value, ir.GetItemExpr)):
            dynamic_read = stmt.value
            break
    assert isinstance(dynamic_read.value, ir.MakeTuple)


def test_nested_tuple_has_anchor_but_remains_make_tuple_expression():
    @pl.function
    def func(x: pl.DT_INT64, y: pl.DT_INT64):
        outer = (x, (x, y))  # noqa: F841

    assignments = [stmt for stmt in func.body.stmts if isinstance(stmt, ir.AssignStmt)]
    outer = next(stmt for stmt in assignments if stmt.var.name == "outer")
    assert isinstance(outer.value, ir.MakeTuple)
    nested = outer.value.elements[1]
    assert isinstance(nested, ir.MakeTuple)
    anchors = [stmt for stmt in assignments if stmt.var.name.startswith("_tuple_anchor_")]
    assert len(anchors) == 1
    assert anchors[0].value is nested


def test_constant_tuple_index_is_direct_element():
    @pl.function
    def func():
        event_ids = (3, 7)
        selected = event_ids[1]  # noqa: F841

    assignments = [stmt for stmt in func.body.stmts if isinstance(stmt, ir.AssignStmt)]
    assert isinstance(assignments[-1].value, ir.ConstInt)
    assert assignments[-1].value.value == 7


def test_tile_factories_accept_propagated_constant_kwargs():
    """Tile factory kwargs accept parser-propagated dtype and shape constants."""

    @pl.function
    def func():
        dtype_selector = 0
        if dtype_selector == 0:
            if dtype_selector + 1 == 1:
                dtype = pl.DT_FP16
            else:
                dtype = pl.DT_BF16
        else:
            dtype = pl.DT_BF16

        shape_selector = 1
        if shape_selector == 1:
            if shape_selector * 64 == 64:
                shape = [64, 64]
            else:
                shape = [128, 128]
        else:
            shape = [128, 128]

        valid_shape_selector = 2
        if valid_shape_selector == 2:
            if valid_shape_selector * 16 == 32:
                valid_shape = [32, 64]
            else:
                valid_shape = [64, 64]
        else:
            valid_shape = [64, 64]
        tile_type = pl.TileType(shape=shape, valid_shape=valid_shape, dtype=dtype)
        tile = pl.make_tile(tile_type, addr=0, size=8192)  # noqa: F841
        group = pl.make_tile_group(type=tile_type, addrs=0x2000, mutex_ids=[0, 1])  # noqa: F841

    tile_assignments = [
        stmt
        for stmt in func.body.stmts
        if isinstance(stmt, ir.AssignStmt) and isinstance(stmt.var.type, ir.TileType)
    ]
    # One explicit tile plus the two slots created for the tile group.
    assert len(tile_assignments) == 3
    for tile_assignment in tile_assignments:
        tile_type = tile_assignment.var.type
        assert tile_type.dtype == pl.DT_FP16
        assert all(isinstance(element, ir.ConstInt) for element in tile_type.shape)
        assert [element.value for element in tile_type.shape] == [64, 64]
        assert isinstance(tile_assignment.value, ir.Call)
        valid_shape = tile_assignment.value.args[1]
        assert isinstance(valid_shape, ir.MakeTuple)
        assert [element.value for element in valid_shape.elements] == [32, 64]


# ---------------------------------------------------------------------------
# Compile-time-only kwargs reject runtime values
# ---------------------------------------------------------------------------
# TileType shape/valid_shape and the load/store `order` axes are consumed while
# parsing, so a runtime value in them cannot be lowered at all: a scalar var
# would leak its name into the generated C++ tile declaration, which is hoisted
# above the point where that var is defined. They are resolved through the
# strict compile-time accessor rather than the permissive kwarg resolver, which
# passes runtime IR through by design.
#
# The tests below pin the *diagnostic*, not just the rejection. Routing these
# positions through the permissive resolver still rejects all four cases, but
# the error degrades to a raw TypeError / pybind cast failure from inside the
# builder, with no span and no hint.


def test_tile_type_shape_rejects_runtime_value():
    """A runtime tensor dimension is not a usable TileType shape."""
    with pytest.raises(ParserTypeError, match="must be a compile-time integer"):

        @pl.function
        def func(x: pl.Tensor[[pl.DYNAMIC, 128], pl.DT_FP16]):
            m = x.shape[0]
            tile_type = pl.TileType(  # noqa: F841
                shape=[m, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat
            )


def test_tile_type_valid_shape_rejects_runtime_value():
    """Same for valid_shape — pl.set_validshape() is the runtime form."""
    with pytest.raises(ParserTypeError, match="must be a compile-time integer"):

        @pl.function
        def func(x: pl.Tensor[[pl.DYNAMIC, 128], pl.DT_FP16]):
            m = x.shape[0]
            tile_type = pl.TileType(  # noqa: F841
                shape=[128, 128],
                valid_shape=[m, 128],
                dtype=pl.DT_FP16,
                target_memory=pl.MemorySpace.Mat,
            )


def _parse_order_kernel(kernel_def):
    return kernel_def.parse_target_program(ir.SectionKind.Vector)[0]


def test_order_kwarg_selects_the_tensor_axes():
    """A constant `order` reaches the builder and becomes the load's tile_dims."""

    @pl.kernel(auto_mutex=True)
    def k(a: pl.Tensor[[2, 128, 128], pl.DT_FP16]):
        tile_type = pl.TileType(shape=[128, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
        group = pl.make_tile_group(type=tile_type, addrs=0x10000, mutex_ids=[0, 1])
        pl.load(group.next(), a, [0, 0, 0], order=[0, 2])

    assert "tile_dims=[0, 2]" in str(_parse_order_kernel(k))


def test_order_kwarg_accepts_a_kernel_local_axis_list():
    """The axes only have to be known while parsing, not spelled as a literal."""

    @pl.kernel(auto_mutex=True)
    def k(a: pl.Tensor[[2, 128, 128], pl.DT_FP16]):
        axes = [0, 2]
        tile_type = pl.TileType(shape=[128, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
        group = pl.make_tile_group(type=tile_type, addrs=0x10000, mutex_ids=[0, 1])
        pl.load(group.next(), a, [0, 0, 0], order=axes)

    assert "tile_dims=[0, 2]" in str(_parse_order_kernel(k))


def test_load_without_order_is_left_alone():
    """The order hook is a no-op when the call does not pass the kwarg."""

    @pl.kernel(auto_mutex=True)
    def k(a: pl.Tensor[[2, 128, 128], pl.DT_FP16]):
        tile_type = pl.TileType(shape=[128, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
        group = pl.make_tile_group(type=tile_type, addrs=0x10000, mutex_ids=[0, 1])
        pl.load(group.next(), a, [0, 0, 0])

    ir_str = str(_parse_order_kernel(k))
    assert "block.load" in ir_str
    assert "tile_dims" not in ir_str


def test_order_kwarg_rejects_bool_axes():
    """bool subclasses int, but ``True`` is not an axis."""

    @pl.kernel(auto_mutex=True)
    def k(a: pl.Tensor[[2, 128, 128], pl.DT_FP16]):
        tile_type = pl.TileType(shape=[128, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
        group = pl.make_tile_group(type=tile_type, addrs=0x10000, mutex_ids=[0, 1])
        pl.load(group.next(), a, [0, 0, 0], order=[True, 2])

    with pytest.raises(ParserTypeError, match="'order' must be a compile-time integer list"):
        _parse_order_kernel(k)


def test_order_kwarg_rejects_runtime_axis():
    """`order` selects tensor axes at parse time, so an axis may not be runtime."""

    @pl.kernel(auto_mutex=True)
    def k(a: pl.Tensor[[2, 128, 128], pl.DT_FP16], axis: pl.DT_INT32):
        tile_type = pl.TileType(shape=[128, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
        group = pl.make_tile_group(type=tile_type, addrs=0x10000, mutex_ids=[0, 1])
        pl.load(group.next(), a, [0, 0, 0], order=[axis, 2])

    with pytest.raises(ParserTypeError, match="'order' must be a compile-time integer list"):
        _parse_order_kernel(k)


def test_order_kwarg_rejects_float_axes():
    """A constant `order` still has to be integral."""

    @pl.kernel(auto_mutex=True)
    def k(a: pl.Tensor[[2, 128, 128], pl.DT_FP16]):
        tile_type = pl.TileType(shape=[128, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
        group = pl.make_tile_group(type=tile_type, addrs=0x10000, mutex_ids=[0, 1])
        pl.load(group.next(), a, [0, 0, 0], order=[1.5, 2])

    with pytest.raises(ParserTypeError, match="'order' must be a compile-time integer list"):
        _parse_order_kernel(k)
