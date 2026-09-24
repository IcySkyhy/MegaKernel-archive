#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See the License in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Unit tests for array field support in pl.struct and pl.struct_array."""

from __future__ import annotations

from pypto_pro import ir
import pypto_pro.language as pl

from pypto.pypto_impl.ir import DataType


def test_struct_array_field_lowers_to_nested_tuple():
    """A list-valued field in pl.struct becomes a nested TupleType."""

    @pl.function
    def kernel():
        s = pl.struct("S", x=0, arr=[0, 0, 0, 0])
        val: pl.DT_INT64 = s.arr[0]
        return val

    assert isinstance(kernel, ir.Function)
    assignments = [stmt for stmt in kernel.body.stmts if isinstance(stmt, ir.AssignStmt)]
    struct_assign = next(stmt for stmt in assignments if stmt.var.name == "s")
    struct_type = struct_assign.var.type
    assert isinstance(struct_type, ir.TupleType)
    assert len(struct_type.types) == 2
    assert isinstance(struct_type.types[0], ir.ScalarType)
    assert isinstance(struct_type.types[1], ir.TupleType)
    assert len(struct_type.types[1].types) == 4


def test_struct_array_field_read_produces_nested_getitem():
    """s.arr[idx] lowers to GetItemExpr(GetItemExpr(s, field_idx), elem_idx)."""

    @pl.function
    def kernel():
        s = pl.struct("S", x=0, arr=[0, 0, 0, 0])
        val: pl.DT_INT64 = s.arr[2]
        return val

    assert isinstance(kernel, ir.Function)
    assignments = [stmt for stmt in kernel.body.stmts if isinstance(stmt, ir.AssignStmt)]
    result = next(stmt for stmt in assignments if stmt.var.name == "val")
    assert isinstance(result.value, ir.GetItemExpr)


def test_struct_mixed_scalar_and_array_fields():
    """pl.struct with both scalar and array fields."""

    @pl.function
    def kernel():
        s = pl.struct("S", n=0, offsets=[0, 0, 0], scale=0.0)
        val: pl.DT_INT64 = s.offsets[1]
        return val

    assert isinstance(kernel, ir.Function)
    assignments = [stmt for stmt in kernel.body.stmts if isinstance(stmt, ir.AssignStmt)]
    struct_assign = next(stmt for stmt in assignments if stmt.var.name == "s")
    struct_type = struct_assign.var.type
    assert len(struct_type.types) == 3
    assert isinstance(struct_type.types[0], ir.ScalarType)
    assert isinstance(struct_type.types[1], ir.TupleType)
    assert len(struct_type.types[1].types) == 3
    assert isinstance(struct_type.types[2], ir.ScalarType)


def test_struct_multiple_array_fields():
    """pl.struct with multiple array fields."""

    @pl.function
    def kernel():
        s = pl.struct("S", a=[0, 0], b=[0, 0, 0, 0], c=0)
        val: pl.DT_INT64 = s.b[3]
        return val

    assert isinstance(kernel, ir.Function)
    assignments = [stmt for stmt in kernel.body.stmts if isinstance(stmt, ir.AssignStmt)]
    struct_assign = next(stmt for stmt in assignments if stmt.var.name == "s")
    struct_type = struct_assign.var.type
    assert len(struct_type.types) == 3
    assert isinstance(struct_type.types[0], ir.TupleType)
    assert len(struct_type.types[0].types) == 2
    assert isinstance(struct_type.types[1], ir.TupleType)
    assert len(struct_type.types[1].types) == 4


def test_struct_array_with_array_field():
    """pl.struct_array with array fields creates N struct.create slots."""

    @pl.function
    def kernel():
        arr = pl.struct_array(2, "Slot", x=0, vals=[0, 0, 0, 0])
        val: pl.DT_INT64 = arr[0].vals[1]
        return val

    assert isinstance(kernel, ir.Function)
    assignments = [stmt for stmt in kernel.body.stmts if isinstance(stmt, ir.AssignStmt)]
    slot0 = next(stmt for stmt in assignments if stmt.var.name == "arr_0")
    assert isinstance(slot0.value, ir.Call)
    assert slot0.value.name == "struct.create"


def test_struct_array_array_field_read():
    """arr[0].arr_field[idx] read access on struct_array."""

    @pl.function
    def kernel():
        arr = pl.struct_array(3, "Slot", x=0, data=[0, 0, 0])
        val: pl.DT_INT64 = arr[1].data[2]
        return val

    assert isinstance(kernel, ir.Function)
    assignments = [stmt for stmt in kernel.body.stmts if isinstance(stmt, ir.AssignStmt)]
    result = next(stmt for stmt in assignments if stmt.var.name == "val")
    assert isinstance(result.value, ir.GetItemExpr)


def test_struct_array_attribute_on_subscript_read():
    """arr[0].field read access on struct_array (scalar field)."""

    @pl.function
    def kernel():
        arr = pl.struct_array(2, "Slot", x=0, y=0)
        val: pl.DT_INT64 = arr[0].x
        return val

    assert isinstance(kernel, ir.Function)
    assignments = [stmt for stmt in kernel.body.stmts if isinstance(stmt, ir.AssignStmt)]
    result = next(stmt for stmt in assignments if stmt.var.name == "val")
    assert isinstance(result.value, ir.GetItemExpr)


def test_struct_array_field_dynamic_index_read():
    """s.arr[i] with a loop variable as index."""

    @pl.function
    def kernel():
        s = pl.struct("S", arr=[0, 0, 0, 0])
        total: pl.DT_INT64 = 0
        for i in pl.range(0, 4):
            total = total + s.arr[i]
        return total

    assert isinstance(kernel, ir.Function)


def test_struct_array_field_all_indices_accessible():
    """All indices of an array field are accessible."""

    @pl.function
    def kernel():
        s = pl.struct("S", arr=[0, 0, 0])
        a: pl.DT_INT64 = s.arr[0]
        b: pl.DT_INT64 = s.arr[1]
        c: pl.DT_INT64 = s.arr[2]
        return a + b + c

    assert isinstance(kernel, ir.Function)


def test_struct_float_array_field():
    """pl.struct with a float array field."""

    @pl.function
    def kernel():
        s = pl.struct("S", x=0, farr=[0.0, 0.0, 0.0])
        val: pl.DT_FP32 = s.farr[1]
        return val

    assert isinstance(kernel, ir.Function)
    assignments = [stmt for stmt in kernel.body.stmts if isinstance(stmt, ir.AssignStmt)]
    struct_assign = next(stmt for stmt in assignments if stmt.var.name == "s")
    arr_type = struct_assign.var.type.types[1]
    assert isinstance(arr_type, ir.TupleType)
    assert all(t.dtype == DataType.FP32 for t in arr_type.types)


def test_struct_bool_array_field():
    """pl.struct with a bool array field."""

    @pl.function
    def kernel():
        s = pl.struct("S", x=0, barr=[True, False, True])
        val: pl.DT_INT64 = s.barr[0]
        return val

    assert isinstance(kernel, ir.Function)
    assignments = [stmt for stmt in kernel.body.stmts if isinstance(stmt, ir.AssignStmt)]
    struct_assign = next(stmt for stmt in assignments if stmt.var.name == "s")
    arr_type = struct_assign.var.type.types[1]
    assert isinstance(arr_type, ir.TupleType)
    assert all(t.dtype == DataType.BOOL for t in arr_type.types)


def test_struct_mixed_dtype_array_fields():
    """pl.struct with int, float, and bool array fields simultaneously."""

    @pl.function
    def kernel():
        s = pl.struct("S", ints=[0, 0], floats=[0.0, 0.0], bools=[True, False])
        iv: pl.DT_INT64 = s.ints[0]
        fv: pl.DT_FP32 = s.floats[1]
        bv: pl.DT_INT64 = s.bools[0]
        return iv + fv + bv

    assert isinstance(kernel, ir.Function)
    assignments = [stmt for stmt in kernel.body.stmts if isinstance(stmt, ir.AssignStmt)]
    struct_assign = next(stmt for stmt in assignments if stmt.var.name == "s")
    types = struct_assign.var.type.types
    assert all(t.dtype == DataType.INDEX for t in types[0].types)
    assert all(t.dtype == DataType.FP32 for t in types[1].types)
    assert all(t.dtype == DataType.BOOL for t in types[2].types)


def test_struct_array_field_length_1():
    """Single-element array field produces a TupleType of length 1."""

    @pl.function
    def kernel():
        s = pl.struct("S", x=0, arr=[0])
        val: pl.DT_INT64 = s.arr[0]
        return val

    assert isinstance(kernel, ir.Function)
    assignments = [stmt for stmt in kernel.body.stmts if isinstance(stmt, ir.AssignStmt)]
    struct_assign = next(stmt for stmt in assignments if stmt.var.name == "s")
    arr_type = struct_assign.var.type.types[1]
    assert isinstance(arr_type, ir.TupleType)
    assert len(arr_type.types) == 1


def test_struct_array_field_length_8():
    """Eight-element array field produces a TupleType of length 8."""

    @pl.function
    def kernel():
        s = pl.struct("S", arr=[0, 0, 0, 0, 0, 0, 0, 0])
        val: pl.DT_INT64 = s.arr[7]
        return val

    assert isinstance(kernel, ir.Function)
    assignments = [stmt for stmt in kernel.body.stmts if isinstance(stmt, ir.AssignStmt)]
    struct_assign = next(stmt for stmt in assignments if stmt.var.name == "s")
    arr_type = struct_assign.var.type.types[0]
    assert isinstance(arr_type, ir.TupleType)
    assert len(arr_type.types) == 8


def test_struct_array_field_read_last_index():
    """The last index of an array field is accessible."""

    @pl.function
    def kernel():
        s = pl.struct("S", arr=[10, 20, 30, 40])
        val: pl.DT_INT64 = s.arr[3]
        return val

    assert isinstance(kernel, ir.Function)
    assignments = [stmt for stmt in kernel.body.stmts if isinstance(stmt, ir.AssignStmt)]
    result = next(stmt for stmt in assignments if stmt.var.name == "val")
    assert isinstance(result.value, ir.GetItemExpr)
