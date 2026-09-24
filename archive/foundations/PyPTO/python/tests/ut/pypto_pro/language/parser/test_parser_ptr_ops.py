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
"""Parser and IR tests for pointer reinterpretation and tensor re-view APIs."""

import pypto_pro.language as pl
import pytest

from pypto.pypto_impl import ir


def _ir_str(kernel_def) -> str:
    return str(_parse_kernel(kernel_def))


def _parse_kernel(kernel_def) -> ir.Program:
    return kernel_def.parse_target_program(ir.SectionKind.Vector)[0]


@pl.kernel
def _make_ptr_change_dtype_kernel(p: pl.Ptr[pl.DT_UINT8]):
    # Reinterpret a raw uint8 GM pointer as fp16, then view it as a 2D tensor.
    fp16_ptr = pl.make_ptr(p, dtype=pl.DT_FP16)
    view = pl.make_tensor(fp16_ptr, [64, 128], [128, 1])  # noqa: F841


@pl.kernel
def _make_ptr_identity_kernel(p: pl.Ptr[pl.DT_FP16]):
    # No dtype kwarg -> identity reinterpret (keeps the source dtype).
    same = pl.make_ptr(p)
    view = pl.make_tensor(same, [32, 32], [32, 1])  # noqa: F841


@pl.kernel
def _make_ptr_then_addptr_kernel(p: pl.Ptr[pl.DT_UINT8]):
    # make_ptr first, then advance with element (fp16) semantics.
    fp16_ptr = pl.make_ptr(p, dtype=pl.DT_FP16)
    bumped = pl.addptr(fp16_ptr, 64 * 128)
    view = pl.make_tensor(bumped, [64, 128], [128, 1])  # noqa: F841


def test_make_ptr_emits_ptr_make_ptr_call_with_new_dtype():
    s = _ir_str(_make_ptr_change_dtype_kernel)
    assert "ptr.make_ptr" in s
    assert "dtype=half" in s
    # The tensor view built on the reinterpreted pointer carries the new dtype.
    assert "tensor<64 x 128, half, tensor_view<, 128 x 1, ND, %fp16_ptr>>" in s


def test_make_ptr_result_type_is_ptr_with_changed_dtype():
    prog = _parse_kernel(_make_ptr_change_dtype_kernel)
    body = prog.get_function(prog_first_name(prog)).body
    # Walk the function body to find the make_ptr AssignStmt and inspect its type.
    found = _find_assign_by_call(body, "ptr.make_ptr")
    assert found is not None, "expected a ptr.make_ptr assignment"
    ptr_type = found.var.type
    assert isinstance(ptr_type, ir.PtrType)
    assert ptr_type.dtype == pl.DT_FP16


def test_make_ptr_identity_keeps_source_dtype():
    prog = _parse_kernel(_make_ptr_identity_kernel)
    found = _find_assign_by_call(prog.get_function(prog_first_name(prog)).body, "ptr.make_ptr")
    assert found is not None
    assert isinstance(found.var.type, ir.PtrType)
    assert found.var.type.dtype == pl.DT_FP16


def test_make_ptr_then_addptr_ir():
    s = _ir_str(_make_ptr_then_addptr_kernel)
    assert "ptr.make_ptr" in s
    assert "ptr.addptr" in s


@pl.kernel
def _ptr_plus_offset_kernel(p: pl.Ptr[pl.DT_UINT8]):
    # `ptr + offset` is sugar for pl.addptr(ptr, offset).
    fp16_ptr = pl.make_ptr(p, dtype=pl.DT_FP16)
    bumped = fp16_ptr + 64 * 128
    view = pl.make_tensor(bumped, [64, 128], [128, 1])  # noqa: F841


def test_ptr_plus_offset_lowers_to_addptr_ir():
    s = _ir_str(_ptr_plus_offset_kernel)
    assert "ptr.make_ptr" in s
    assert "ptr.addptr" in s


def test_ptr_plus_offset_result_type_is_ptr():
    prog = _parse_kernel(_ptr_plus_offset_kernel)
    found = _find_assign_by_call(prog.get_function(prog_first_name(prog)).body, "ptr.addptr")
    assert found is not None, "expected `ptr + offset` to emit a ptr.addptr assignment"
    assert isinstance(found.var.type, ir.PtrType)
    assert found.var.type.dtype == pl.DT_FP16


@pl.kernel
def _retensor_shape_kernel(src: pl.Tensor[[64, 128], pl.DT_FP16]):
    # Same dtype, new shape/stride, reusing src's pointer.
    reshaped = pl.make_tensor(src, [128, 64], [64, 1])  # noqa: F841


@pl.kernel
def _retensor_dtype_kernel(src: pl.Tensor[[64, 128], pl.DT_FP16]):
    # Reinterpret an fp16 tensor as a wider uint8 view (2x columns).
    as_u8 = pl.make_tensor(src, [64, 256], [256, 1], dtype=pl.DT_UINT8)  # noqa: F841


@pl.kernel
def _retensor_view_of_view_kernel(src: pl.Tensor[[64, 128], pl.DT_FP16]):
    # A view of a view: both derive from the same underlying pointer.
    v1 = pl.make_tensor(src, [128, 64], [64, 1])
    v2 = pl.make_tensor(v1, [64, 128], [128, 1])  # noqa: F841


def test_make_tensor_from_tensor_ir_type():
    prog = _parse_kernel(_retensor_shape_kernel)
    found = _find_assign_by_call(prog.get_function(prog_first_name(prog)).body, "ptr.make_tensor")
    assert found is not None
    t = found.var.type
    assert isinstance(t, ir.TensorType)
    assert [int(d.value) for d in t.shape] == [128, 64]
    assert t.dtype == pl.DT_FP16
    # The source pointer is recorded on the view so codegen can reuse it.
    assert t.tensor_view is not None
    assert t.tensor_view.ptr is not None


def test_make_tensor_from_tensor_reinterpret_dtype_ir():
    s = _ir_str(_retensor_dtype_kernel)
    assert "ptr.make_tensor" in s
    assert "tensor<64 x 256, uint8_t, tensor_view<, 256 x 1, ND, %src>>" in s


def test_make_tensor_view_of_view_ir():
    prog = _parse_kernel(_retensor_view_of_view_kernel)
    s = str(prog)
    assert s.count("ptr.make_tensor") == 2


def test_make_ptr_accepts_tensor_argument():
    @pl.kernel
    def k(t: pl.Tensor[[64], pl.DT_FP16]):
        bad = pl.make_ptr(t, dtype=pl.DT_FP32)  # noqa: F841

    prog = _parse_kernel(k)
    found = _find_assign_by_call(prog.get_function(prog_first_name(prog)).body, "ptr.make_ptr")
    assert found is not None
    assert isinstance(found.var.type, ir.PtrType)
    assert found.var.type.dtype == pl.DT_FP32


def prog_first_name(prog):
    """Return the (single) function name in a parsed Program."""
    return next(iter(prog.functions))


def _find_assign_by_call(stmt, op_name):
    """Find the first AssignStmt whose value is a Call to ``op_name``."""
    if isinstance(stmt, ir.AssignStmt):
        val = stmt.value
        if isinstance(val, ir.Call) and getattr(val, "name", None) == op_name:
            return stmt
        return None
    if isinstance(stmt, ir.SeqStmts):
        for s in stmt.stmts:
            found = _find_assign_by_call(s, op_name)
            if found is not None:
                return found
    for attr in ("body", "then_body", "else_body"):
        sub = getattr(stmt, attr, None)
        if isinstance(sub, ir.Stmt):
            found = _find_assign_by_call(sub, op_name)
            if found is not None:
                return found
    return None


@pl.kernel
def _addptr_int4_kernel(p: pl.Ptr[pl.DT_INT4]):
    bumped = pl.addptr(p, 8)  # noqa: F841


@pl.kernel
def _plus_offset_fp4_kernel(p: pl.Ptr[pl.DT_FP4]):
    bumped = p + 8  # noqa: F841


@pl.kernel
def _make_ptr_to_int4_then_addptr_kernel(p: pl.Ptr[pl.DT_UINT8]):
    # A legitimate uint8 pointer reinterpreted as int4 must still reject addptr.
    i4 = pl.make_ptr(p, dtype=pl.DT_INT4)
    bumped = pl.addptr(i4, 8)  # noqa: F841


def test_addptr_rejects_int4_pointer():
    with pytest.raises(Exception, match="sub-byte"):
        _parse_kernel(_addptr_int4_kernel)


def test_plus_offset_rejects_fp4_pointer():
    with pytest.raises(Exception, match="sub-byte"):
        _parse_kernel(_plus_offset_fp4_kernel)


def test_addptr_rejects_reinterpreted_int4_pointer():
    with pytest.raises(Exception, match="sub-byte"):
        _parse_kernel(_make_ptr_to_int4_then_addptr_kernel)
