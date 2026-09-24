# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See the License in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""UT for make_tuple chain attribute access: t.a.b and t.field.method()."""

from pypto_pro import ir
import pypto_pro.language as pl
from pypto_pro.language.parser.diagnostics import ParserSyntaxError, UnsupportedFeatureError
import pytest


def _find_assign(func, name):
    return next(stmt for stmt in func.body.stmts if isinstance(stmt, ir.AssignStmt) and stmt.var.name == name)


# =============================================================================
# Chain attribute access: t.a.b (nested make_tuple)
# =============================================================================

def test_make_tuple_chain_attr_const_fold():
    """t.a.b where inner is const MakeTuple — folds to element."""

    @pl.function
    def kernel():
        inner = pl.make_tuple(x=10, y=20)
        outer = pl.make_tuple(a=inner)
        val: pl.DT_INT64 = outer.a.x
        return val

    assert isinstance(kernel, ir.Function)


def test_make_tuple_chain_attr_getitem():
    """t.a.b where inner is a runtime Var — lowers to nested GetItemExpr."""

    @pl.function
    def kernel(x: pl.DT_INT32):
        s = pl.struct("S", f=0)
        t = pl.make_tuple(s=s)
        val: pl.DT_INT32 = t.s.f
        return val

    assert isinstance(kernel, ir.Function)


def test_make_tuple_chain_three_level():
    """t.a.b.c — three-level chain."""

    @pl.function
    def kernel():
        leaf = pl.make_tuple(v=42)
        mid = pl.make_tuple(b=leaf)
        root = pl.make_tuple(a=mid)
        val: pl.DT_INT64 = root.a.b.v
        return val

    assert isinstance(kernel, ir.Function)


# =============================================================================
# 2.1 Constant-fold lowering path — IR structure
# =============================================================================

def test_chain_const_fold_lowers_to_static_element():
    """outer.a.x folds both const hops down to the static element.

    lower_attr_access folds a const MakeTuple base to its element, so the
    whole chain collapses to the innermost constant instead of keeping a
    GetItemExpr over a MakeTuple.
    """

    @pl.function
    def kernel():
        inner = pl.make_tuple(x=10, y=20)
        outer = pl.make_tuple(a=inner)
        val: pl.DT_INT64 = outer.a.x
        return val

    stmt = _find_assign(kernel, "val")
    assert isinstance(stmt.value, ir.ConstInt)
    assert stmt.value.value == 10


def test_chain_const_fold_arithmetic():
    """Chain reads fold to constants and participate in const arithmetic."""

    @pl.function
    def kernel():
        inner = pl.make_tuple(x=10, y=20)
        outer = pl.make_tuple(a=inner)
        val: pl.DT_INT64 = outer.a.x + outer.a.y
        return val

    stmt = _find_assign(kernel, "val")
    assert isinstance(stmt.value, ir.ConstInt)
    assert stmt.value.value == 30


def test_chain_subscript_after_attr():
    """Subscript on a folded chain element resolves to the constant."""

    @pl.function
    def kernel():
        inner = pl.make_tuple(x=10, y=20)
        outer = pl.make_tuple(a=inner)
        val: pl.DT_INT64 = outer.a[0]
        return val

    stmt = _find_assign(kernel, "val")
    assert isinstance(stmt.value, ir.ConstInt)
    assert stmt.value.value == 10


# =============================================================================
# 2.2 Runtime lowering path — IR structure
# =============================================================================

def test_chain_runtime_lowers_to_nested_getitem():
    """t.s.f folds the const t.s hop to the runtime struct Var, then GetItemExpr."""

    @pl.function
    def kernel(x: pl.DT_INT32):
        s = pl.struct("S", f=0)
        t = pl.make_tuple(s=s)
        val: pl.DT_INT32 = t.s.f
        return val

    stmt = _find_assign(kernel, "val")
    assert isinstance(stmt.value, ir.GetItemExpr)
    assert isinstance(stmt.value.value, ir.Var)
    assert stmt.value.value.name == "s"
    assert stmt.value.slice.value == 0


def test_make_tuple_chain_with_kernel_arg():
    """Chain access still lowers when the kernel carries runtime args."""

    @pl.function
    def kernel(x: pl.DT_INT64):
        inner = pl.make_tuple(x=10)
        outer = pl.make_tuple(a=inner)
        val: pl.DT_INT64 = outer.a.x
        return val

    assert isinstance(kernel, ir.Function)


# =============================================================================
# 2.3 Multi-level chains
# =============================================================================

def test_make_tuple_chain_four_level():
    """t.a.b.c.d — four-level const chain folds to the static element."""

    @pl.function
    def kernel():
        l1 = pl.make_tuple(w=1)
        l2 = pl.make_tuple(c=l1)
        l3 = pl.make_tuple(b=l2)
        root = pl.make_tuple(a=l3)
        val: pl.DT_INT64 = root.a.b.c.w
        return val

    stmt = _find_assign(kernel, "val")
    assert isinstance(stmt.value, ir.ConstInt)
    assert stmt.value.value == 1


def test_chain_parallel_branches():
    """Two independent const chains fold completely, no intermediate temps.

    root.a.c.w folds to l1.w and root.b.w folds to l1.w; both are static.
    """

    @pl.function
    def kernel():
        l1 = pl.make_tuple(w=1)
        l2 = pl.make_tuple(c=l1)
        root = pl.make_tuple(a=l2, b=l1)
        val: pl.DT_INT64 = root.a.c.w + root.b.w
        return val

    _expr_tmps = [
        stmt for stmt in kernel.body.stmts
        if isinstance(stmt, ir.AssignStmt) and stmt.var.name.startswith("_expr_tmp")
    ]
    assert len(_expr_tmps) == 0
    assert isinstance(_find_assign(kernel, "val").value, ir.ConstInt)
    assert _find_assign(kernel, "val").value.value == 2


# =============================================================================
# 2.4 Element-type interactions
# =============================================================================

def test_chain_make_tuple_of_struct_array_elem():
    """make_tuple wrapping a struct_array element: t.s.v lowers via slots[0]."""

    @pl.function
    def kernel():
        slots = pl.struct_array(2, "Slot", v=0)
        t = pl.make_tuple(s=slots[0])
        val: pl.DT_INT32 = t.s.v
        return val

    stmt = _find_assign(kernel, "val")
    assert isinstance(stmt.value, ir.GetItemExpr)
    assert isinstance(stmt.value.value, ir.Var)
    assert stmt.value.value.name.startswith("_expr_tmp")


def test_chain_same_level_multi_field_folds():
    """Same-level fields on a const MakeTuple fold to a constant."""

    @pl.function
    def kernel():
        inner = pl.make_tuple(x=10, y=20)
        val: pl.DT_INT64 = inner.x + inner.y
        return val

    stmt = _find_assign(kernel, "val")
    assert isinstance(stmt.value, ir.ConstInt)
    assert stmt.value.value == 30


# =============================================================================
# 3. tile_group chain method access
# =============================================================================

def test_chain_tile_group_next():
    """info.buf.next() lowers to a dynamic tile select + mutex id."""

    @pl.function
    def kernel():
        tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        db = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[20, 21])
        info = pl.make_tuple(buf=db)
        t = info.buf.next()
        return t

    idx = next(
        stmt for stmt in kernel.body.stmts
        if isinstance(stmt, ir.AssignStmt) and stmt.var.name.startswith("_bufidx_")
    )
    assert isinstance(idx.value, ir.FloorMod)
    mutex = next(
        stmt
        for stmt in kernel.body.stmts
        if isinstance(stmt, ir.AssignStmt) and stmt.var.name.endswith("__mutexid")
    )
    assert isinstance(mutex.value, ir.GetItemExpr)


def test_chain_tile_group_current():
    """info.buf.current() lowers without advancing the cursor."""

    @pl.function
    def kernel():
        tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        db = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[20, 21])
        info = pl.make_tuple(buf=db)
        t = info.buf.current()
        return t

    mutex = next(
        stmt
        for stmt in kernel.body.stmts
        if isinstance(stmt, ir.AssignStmt) and stmt.var.name.endswith("__mutexid")
    )
    assert isinstance(mutex.value, ir.GetItemExpr)


def test_chain_tile_group_previous():
    """info.buf.previous() lowers to a dynamic tile select."""

    @pl.function
    def kernel():
        tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        db = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[20, 21])
        info = pl.make_tuple(buf=db)
        t = info.buf.previous()
        return t

    mutex = next(
        stmt
        for stmt in kernel.body.stmts
        if isinstance(stmt, ir.AssignStmt) and stmt.var.name.endswith("__mutexid")
    )
    assert isinstance(mutex.value, ir.GetItemExpr)


def test_chain_tile_group_method_sequence():
    """next/next/previous sequence keeps compiling and maintains the cursor."""

    @pl.function
    def kernel():
        tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        db = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[20, 21])
        info = pl.make_tuple(buf=db)
        t1 = info.buf.next()
        info.buf.next()
        info.buf.previous()
        return t1

    assert isinstance(kernel, ir.Function)


def test_chain_tile_group_single_slot():
    """Single-slot tile group: no cursor, static const mutex id."""

    @pl.function
    def kernel():
        tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        db = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[20])
        info = pl.make_tuple(buf=db)
        t = info.buf.next()
        return t

    mutex = next(
        stmt
        for stmt in kernel.body.stmts
        if isinstance(stmt, ir.AssignStmt) and stmt.var.name.endswith("__mutexid")
    )
    assert isinstance(mutex.value, ir.ConstInt)
    assert mutex.value.value == 20


def test_chain_tile_group_acc_multi_slot():
    """Chained next() on an Acc-memory multi-slot tile_group with valid_shape/compact."""

    @pl.function
    def kernel():
        tt = pl.TileType(
            shape=[64, 64],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Acc,
            valid_shape=[-1, -1],
            compact=1,
        )
        acc_db = pl.make_tile_group(type=tt, addrs=0x20000, mutex_ids=[2, 3])
        tensor_info = pl.make_tuple(acc_db=acc_db)
        acc = tensor_info.acc_db.next()
        return acc

    idx = next(
        stmt for stmt in kernel.body.stmts
        if isinstance(stmt, ir.AssignStmt) and stmt.var.name.startswith("_bufidx_")
    )
    assert isinstance(idx.value, ir.FloorMod)
    mutex = next(
        stmt
        for stmt in kernel.body.stmts
        if isinstance(stmt, ir.AssignStmt) and stmt.var.name.endswith("__mutexid")
    )
    assert isinstance(mutex.value, ir.GetItemExpr)


def test_chain_tile_group_unknown_method():
    """Unknown method on a chained tile_group raises a ParserSyntaxError."""
    with pytest.raises(ParserSyntaxError):

        @pl.function
        def kernel():
            tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
            db = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[20])
            info = pl.make_tuple(buf=db)
            return info.buf.foo()


# =============================================================================
# 5. Error handling and boundaries
# =============================================================================

def test_err_chain_nonexistent_field():
    """Chained read of a nonexistent field raises UnsupportedFeatureError."""
    with pytest.raises(UnsupportedFeatureError):

        @pl.function
        def kernel():
            inner = pl.make_tuple(x=10)
            outer = pl.make_tuple(a=inner)
            val: pl.DT_INT64 = outer.a.non_exist
            return val


def test_err_chain_on_scalar_element():
    """Chaining onto a scalar element raises UnsupportedFeatureError."""
    with pytest.raises(UnsupportedFeatureError):

        @pl.function
        def kernel():
            inner = pl.make_tuple(x=10, y=20)
            outer = pl.make_tuple(a=inner)
            val: pl.DT_INT64 = outer.a.x.foo
            return val


def test_err_chain_write():
    """make_tuple is immutable; chained write raises ParserSyntaxError."""
    with pytest.raises(ParserSyntaxError, match="immutable named tuple field"):

        @pl.function
        def kernel():
            inner = pl.make_tuple(x=10)
            outer = pl.make_tuple(a=inner)
            outer.a.x = 99
            val: pl.DT_INT64 = outer.a.x
            return val


def test_err_named_tuple_array_field_element_write():
    """make_tuple is immutable; array field element write raises ParserSyntaxError."""
    with pytest.raises(ParserSyntaxError, match="immutable named tuple field"):

        @pl.function
        def kernel():
            t = pl.make_tuple(a=[1, 2, 3])
            t.a[0] = 99
            val: pl.DT_INT64 = t.a[0]
            return val


def test_err_chain_string_subscript():
    """String subscript on a make_tuple raises ParserSyntaxError."""
    with pytest.raises(ParserSyntaxError, match="integer"):

        @pl.function
        def kernel():
            inner = pl.make_tuple(x=10)
            outer = pl.make_tuple(a=inner)
            val: pl.DT_INT64 = outer["a"]
            return val


def test_err_struct_field_holding_make_tuple():
    """Chained read through a struct field holding a make_tuple is unsupported."""
    with pytest.raises(UnsupportedFeatureError):

        @pl.function
        def kernel():
            inner = pl.make_tuple(x=10)
            s = pl.struct("S", t=0)
            s.t = inner
            val: pl.DT_INT32 = s.t.x
            return val
