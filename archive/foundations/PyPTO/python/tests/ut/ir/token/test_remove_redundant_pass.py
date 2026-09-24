# -*- coding: utf-8 -*-
# ruff: noqa: E501
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Tests for removing redundant token dependencies after MergeStmtsIntoIf."""

import pypto
from pypto import ir
from pypto.pil.compile_pipeline import compile_new_ir

from ..test_common import check_snapshot


class _Snapshot:
    def __init__(self, name, text):
        self.name = name
        self.text = text

    def __str__(self):
        return self.text


def _run_passes(func, *args):
    before_remove = []

    def capture_before_remove(program):
        compiled_func = program.functions[func.__name__]
        before_remove.append(_Snapshot(compiled_func.name, str(compiled_func)))
        return program

    dce = ir.Pass.aggressive_dce()
    canonicalize = ir.Pass.canonicalize()
    merge_stmts = ir.Pass.merge_stmts_into_if()
    pipeline = [
        ("infer_token_pass", ir.Pass.infer_token_pass()),
        ("first_canonicalize_dce", lambda p:dce(canonicalize(p))),
        ("second_canonicalize_dce", lambda p:dce(canonicalize(p))),
        ("canonicalize(merge_stmts)", lambda p:canonicalize(merge_stmts(p))),
        ("capture_before_remove_redundant_token_pass", capture_before_remove),
        ("remove_redundant_token_pass", ir.Pass.remove_redundant_token_pass()),
    ]
    after_remove = compile_new_ir(func, *args, pipeline=pipeline, create_new_logical_tensor=True)
    return before_remove[0], after_remove


def _tensor(shape, name):
    return pypto.Tensor(shape=shape, dtype=pypto.DT_FP32, name=name)




BEFORE_IR_0 = """
function foo incast(v0_logical_tensor %x@0 #dtype(float) #shape([16, 16]) #offset([0, 0]) #dynvalidshape([16, 16]), v0_logical_tensor %out@1 #dtype(float) #shape([16, 16]) #offset([0, 0]) #dynvalidshape([16, 16])) outcast() #type(Opaque) #entry(false) {
    v0_logical_tensor %$0@2 #dtype(float) #shape([16, 16]) #offset([0, 0]) #dynvalidshape([16, 16]), token %$0_w, token %x_r = !10000 ADDS(%x@0) token();
    v0_logical_tensor %out_1@1 #dtype(float) #shape([16, 16]) #offset([0, 0]) #dynvalidshape([16, 16]), token %out_1_w, token %$0_r = !10001 ASSEMBLE(%$0@2) token(%$0_w) #toOffset(Unsupported);
    return %x@0, %out_1@1;
}
"""


IR_0 = """
function foo incast(v0_logical_tensor %x@0 #dtype(float) #shape([16, 16]) #offset([0, 0]) #dynvalidshape([16, 16]), v0_logical_tensor %out@1 #dtype(float) #shape([16, 16]) #offset([0, 0]) #dynvalidshape([16, 16])) outcast() #type(Opaque) #entry(false) {
    v0_logical_tensor %$0@2 #dtype(float) #shape([16, 16]) #offset([0, 0]) #dynvalidshape([16, 16]) = !10003 ADDS(%x@0) token();
    v0_logical_tensor %out_1@1 #dtype(float) #shape([16, 16]) #offset([0, 0]) #dynvalidshape([16, 16]) = !10004 ASSEMBLE(%$0@2) token() #toOffset(Unsupported);
    return %x@0, %out_1@1;
}
"""


def test_remove_ssa_dependency():
    def foo(x, out):
        value = x + 1.0
        pypto.assemble(value, [0, 0], out)

    before_remove, after_remove = _run_passes(foo, _tensor((16, 16), "x"), _tensor((16, 16), "out"))
    check_snapshot(before_remove, BEFORE_IR_0)
    check_snapshot(after_remove, IR_0)


BEFORE_IR_1 = """
function foo incast(v0_logical_tensor %src@0 #dtype(float) #shape([8, 16]) #offset([0, 0]) #dynvalidshape([8, 16]), v0_logical_tensor %aux@1 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16]), v0_logical_tensor %result@2 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16])) outcast() #type(Opaque) #entry(false) {
    v0_logical_tensor %aux_1@1 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16]), token %aux_1_w, token %src_r = !10000 ASSEMBLE(%src@0) token() #toOffset(Unsupported);
    v0_logical_tensor %aux_3@1 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16]), token %aux_3_w, token %src_r = !10001 ASSEMBLE(%src@0) token(%aux_1_w) #toOffset(Unsupported);
    v0_logical_tensor %$0@3 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16]), token %$0_w, token %aux_3_r = !10002 ADDS(%aux_3@1) token(%aux_3_w);
    return %src@0, %aux_3@1, %$0@3;
}
"""


IR_1 = """
function foo incast(v0_logical_tensor %src@0 #dtype(float) #shape([8, 16]) #offset([0, 0]) #dynvalidshape([8, 16]), v0_logical_tensor %aux@1 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16]), v0_logical_tensor %result@2 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16])) outcast() #type(Opaque) #entry(false) {
    v0_logical_tensor %aux_1@1 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16]), token %aux_1_w = !10005 ASSEMBLE(%src@0) token() #toOffset(Unsupported);
    v0_logical_tensor %aux_3@1 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16]) = !10006 ASSEMBLE(%src@0) token() #toOffset(Unsupported);
    v0_logical_tensor %$0@3 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16]) = !10007 ADDS(%aux_3@1) token(%aux_1_w);
    return %src@0, %aux_3@1, %$0@3;
}
"""


def test_forward_disjoint_write_dependency():
    def foo(src, aux, result):
        pypto.assemble(src, [0, 0], aux)
        pypto.assemble(src, [16, 0], aux)
        result = aux + 1.0  # noqa: F841

    before_remove, after_remove = _run_passes(
        foo,
        _tensor((8, 16), "src"),
        _tensor((32, 16), "aux"),
        _tensor((32, 16), "result"),
    )
    check_snapshot(before_remove, BEFORE_IR_1)
    check_snapshot(after_remove, IR_1)


BEFORE_IR_2 = """
function foo incast(v0_logical_tensor %src@0 #dtype(float) #shape([8, 16]) #offset([0, 0]) #dynvalidshape([8, 16]), v0_logical_tensor %aux@1 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16]), v0_logical_tensor %result@2 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16])) outcast() #type(Opaque) #entry(false) {
    v0_logical_tensor %aux_1@1 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16]), token %aux_1_w, token %src_r = !10000 ASSEMBLE(%src@0) token() #toOffset(Unsupported);
    v0_logical_tensor %aux_3@1 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16]), token %aux_3_w, token %src_r = !10001 ASSEMBLE(%src@0) token(%aux_1_w) #toOffset(Unsupported);
    v0_logical_tensor %$0@3 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16]), token %$0_w, token %aux_3_r = !10002 ADDS(%aux_3@1) token(%aux_3_w);
    return %src@0, %aux_3@1, %$0@3;
}
"""


IR_2 = """
function foo incast(v0_logical_tensor %src@0 #dtype(float) #shape([8, 16]) #offset([0, 0]) #dynvalidshape([8, 16]), v0_logical_tensor %aux@1 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16]), v0_logical_tensor %result@2 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16])) outcast() #type(Opaque) #entry(false) {
    v0_logical_tensor %aux_1@1 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16]), token %aux_1_w = !10004 ASSEMBLE(%src@0) token() #toOffset(Unsupported);
    v0_logical_tensor %aux_3@1 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16]) = !10005 ASSEMBLE(%src@0) token(%aux_1_w) #toOffset(Unsupported);
    v0_logical_tensor %$0@3 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16]) = !10006 ADDS(%aux_3@1) token();
    return %src@0, %aux_3@1, %$0@3;
}
"""


def test_keep_overlapping_write_dependency():
    def foo(src, aux, result):
        pypto.assemble(src, [0, 0], aux)
        pypto.assemble(src, [4, 0], aux)
        result = aux + 1.0  # noqa: F841

    before_remove, after_remove = _run_passes(
        foo,
        _tensor((8, 16), "src"),
        _tensor((32, 16), "aux"),
        _tensor((32, 16), "result"),
    )
    check_snapshot(before_remove, BEFORE_IR_2)
    check_snapshot(after_remove, IR_2)


BEFORE_IR_3 = """
function foo incast(v0_logical_tensor %src@0 #dtype(float) #shape([8, 16]) #offset([0, 0]) #dynvalidshape([8, 16]), v0_logical_tensor %aux@1 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16]), v0_logical_tensor %shape@2 #dtype(float) #shape([-1]) #offset([0]) #dynvalidshape([v0call(RUNTIME_GetInputShapeDim, ARG_shape, 0)])) outcast() #type(Opaque) #entry(false) {
    v0_logical_tensor %aux_1@1 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16]), token %aux_1_w, token %src_r = !10000 ASSEMBLE(%src@0) token() #toOffset(Unsupported);
    v0_logical_tensor %aux_3@1 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16]), token %aux_3_w, token %src_r = !10001 ASSEMBLE(%src@0) token(%aux_1_w) #toOffset(Unsupported);
    return %src@0, %aux_3@1, %shape@2;
}
"""


IR_3 = """
function foo incast(v0_logical_tensor %src@0 #dtype(float) #shape([8, 16]) #offset([0, 0]) #dynvalidshape([8, 16]), v0_logical_tensor %aux@1 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16]), v0_logical_tensor %shape@2 #dtype(float) #shape([-1]) #offset([0]) #dynvalidshape([v0call(RUNTIME_GetInputShapeDim, ARG_shape, 0)])) outcast() #type(Opaque) #entry(false) {
    v0_logical_tensor %aux_1@1 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16]), token %aux_1_w = !10002 ASSEMBLE(%src@0) token() #toOffset(Unsupported);
    v0_logical_tensor %aux_3@1 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16]) = !10003 ASSEMBLE(%src@0) token(%aux_1_w) #toOffset(Unsupported);
    return %src@0, %aux_3@1, %shape@2;
}
"""


def test_keep_dynamic_write_dependency():
    def foo(src, aux, shape):
        pypto.assemble(src, [0, 0], aux)
        pypto.assemble(src, [shape.shape[0], 0], aux)

    before_remove, after_remove = _run_passes(
        foo,
        _tensor((8, 16), "src"),
        _tensor((32, 16), "aux"),
        _tensor((-1,), "shape"),
    )
    check_snapshot(before_remove, BEFORE_IR_3)
    check_snapshot(after_remove, IR_3)


BEFORE_IR_4 = """
function foo incast(v0_logical_tensor %a@0 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), v0_logical_tensor %aux@1 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), v0_logical_tensor %out@2 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128])) outcast() #type(Opaque) #entry(false) {
    v0_logical_tensor %$0@3 #dtype(float) #shape([64, 128]) #offset([0, 0]) #dynvalidshape([64, 128]), token %$0_w = !10000 VEC_DUP() token();
    v0_logical_tensor %aux_1@1 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), token %aux_1_w, token %$0_r = !10001 ASSEMBLE(%$0@3) token(%$0_w) #toOffset(Unsupported);
    v0_logical_tensor %$1@4 #dtype(float) #shape([64, 128]) #offset([0, 0]) #dynvalidshape([64, 128]), token %$1_w = !10002 VEC_DUP() token();
    v0_logical_tensor %aux_3@1 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), token %aux_3_w, token %$1_r = !10003 ASSEMBLE(%$1@4) token(%$1_w, %aux_1_w) #toOffset(Unsupported);
    v0_logical_tensor %$2@5 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), token %$2_w, token %a_r, token %aux_3_r = !10004 ADD(%a@0, %aux_3@1) token(%aux_3_w);
    v0_logical_tensor %out_1@2 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), token %out_1_w, token %$2_r = !10005 ASSEMBLE(%$2@5) token(%$2_w) #toOffset(Unsupported);
    return %a@0, %aux_3@1, %out_1@2;
}
"""


IR_4 = """
function foo incast(v0_logical_tensor %a@0 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), v0_logical_tensor %aux@1 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), v0_logical_tensor %out@2 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128])) outcast() #type(Opaque) #entry(false) {
    v0_logical_tensor %$0@3 #dtype(float) #shape([64, 128]) #offset([0, 0]) #dynvalidshape([64, 128]) = !10010 VEC_DUP() token();
    v0_logical_tensor %aux_1@1 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), token %aux_1_w = !10011 ASSEMBLE(%$0@3) token() #toOffset(Unsupported);
    v0_logical_tensor %$1@4 #dtype(float) #shape([64, 128]) #offset([0, 0]) #dynvalidshape([64, 128]) = !10012 VEC_DUP() token();
    v0_logical_tensor %aux_3@1 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]) = !10013 ASSEMBLE(%$1@4) token() #toOffset(Unsupported);
    v0_logical_tensor %$2@5 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]) = !10014 ADD(%a@0, %aux_3@1) token(%aux_1_w);
    v0_logical_tensor %out_1@2 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]) = !10015 ASSEMBLE(%$2@5) token() #toOffset(Unsupported);
    return %a@0, %aux_3@1, %out_1@2;
}
"""


def test_assemble_scenario_1_parallel_non_overlapping_writes():
    def foo(a, aux, out):
        pypto.assemble(pypto.full([64, 128], 1.0, pypto.DT_FP32), [0, 0], aux)
        pypto.assemble(pypto.full([64, 128], 2.0, pypto.DT_FP32), [64, 0], aux)
        pypto.assemble(a + aux, [0, 0], out)

    shape = (129, 128)
    before_remove, after_remove = _run_passes(foo, _tensor(shape, "a"), _tensor(shape, "aux"), _tensor(shape, "out"))
    check_snapshot(before_remove, BEFORE_IR_4)
    check_snapshot(after_remove, IR_4)


BEFORE_IR_5 = """
function foo incast(v0_logical_tensor %a@0 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), v0_logical_tensor %aux@1 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), v0_logical_tensor %out@2 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128])) outcast() #type(Opaque) #entry(false) {
    v0_logical_tensor %$0@3 #dtype(float) #shape([64, 128]) #offset([0, 0]) #dynvalidshape([64, 128]), token %$0_w = !10000 VEC_DUP() token();
    v0_logical_tensor %aux_1@1 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), token %aux_1_w, token %$0_r = !10001 ASSEMBLE(%$0@3) token(%$0_w) #toOffset(Unsupported);
    v0_logical_tensor %$1@4 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), token %$1_w, token %aux_1_r = !10002 ADDS(%aux_1@1) token(%aux_1_w);
    v0_logical_tensor %$2@5 #dtype(float) #shape([64, 128]) #offset([0, 0]) #dynvalidshape([64, 128]), token %$2_w = !10003 VEC_DUP() token();
    v0_logical_tensor %aux_3@1 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), token %aux_3_w, token %$2_r = !10004 ASSEMBLE(%$2@5) token(%$2_w, %aux_1_r) #toOffset(Unsupported);
    v0_logical_tensor %$3@6 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), token %$3_w, token %$1_r, token %aux_3_r = !10005 ADD(%$1@4, %aux_3@1) token(%$1_w, %aux_3_w);
    v0_logical_tensor %out_1@2 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), token %out_1_w, token %$3_r = !10006 ASSEMBLE(%$3@6) token(%$3_w) #toOffset(Unsupported);
    return %a@0, %aux_3@1, %out_1@2;
}
"""


IR_5 = """
function foo incast(v0_logical_tensor %a@0 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), v0_logical_tensor %aux@1 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), v0_logical_tensor %out@2 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128])) outcast() #type(Opaque) #entry(false) {
    v0_logical_tensor %$0@3 #dtype(float) #shape([64, 128]) #offset([0, 0]) #dynvalidshape([64, 128]) = !10012 VEC_DUP() token();
    v0_logical_tensor %aux_1@1 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]) = !10013 ASSEMBLE(%$0@3) token() #toOffset(Unsupported);
    v0_logical_tensor %$1@4 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), token %aux_1_r = !10014 ADDS(%aux_1@1) token();
    v0_logical_tensor %$2@5 #dtype(float) #shape([64, 128]) #offset([0, 0]) #dynvalidshape([64, 128]) = !10015 VEC_DUP() token();
    v0_logical_tensor %aux_3@1 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]) = !10016 ASSEMBLE(%$2@5) token(%aux_1_r) #toOffset(Unsupported);
    v0_logical_tensor %$3@6 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]) = !10017 ADD(%$1@4, %aux_3@1) token();
    v0_logical_tensor %out_1@2 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]) = !10018 ASSEMBLE(%$3@6) token() #toOffset(Unsupported);
    return %a@0, %aux_3@1, %out_1@2;
}
"""


def test_assemble_scenario_2_war_between_non_overlapping_writes():
    def foo(a, aux, out):
        pypto.assemble(pypto.full([64, 128], 1.0, pypto.DT_FP32), [0, 0], aux)
        value = aux + 1.0
        pypto.assemble(pypto.full([64, 128], 2.0, pypto.DT_FP32), [64, 0], aux)
        pypto.assemble(value + aux, [0, 0], out)

    shape = (129, 128)
    before_remove, after_remove = _run_passes(foo, _tensor(shape, "a"), _tensor(shape, "aux"), _tensor(shape, "out"))
    check_snapshot(before_remove, BEFORE_IR_5)
    check_snapshot(after_remove, IR_5)


BEFORE_IR_6 = """
function foo incast(v0_logical_tensor %a@0 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), v0_logical_tensor %aux@1 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), v0_logical_tensor %out@2 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128])) outcast() #type(Opaque) #entry(false) {
    v0_logical_tensor %$0@3 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), token %$0_w = !10000 VEC_DUP() token();
    v0_logical_tensor %aux_1@1 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), token %aux_1_w, token %$0_r = !10001 ASSEMBLE(%$0@3) token(%$0_w) #toOffset(Unsupported);
    v0_logical_tensor %$1@4 #dtype(float) #shape([64, 128]) #offset([0, 0]) #dynvalidshape([64, 128]), token %$1_w = !10002 VEC_DUP() token();
    v0_logical_tensor %aux_3@1 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), token %aux_3_w, token %$1_r = !10003 ASSEMBLE(%$1@4) token(%$1_w, %aux_1_w) #toOffset(Unsupported);
    v0_logical_tensor %$2@5 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), token %$2_w, token %a_r, token %aux_3_r = !10004 ADD(%a@0, %aux_3@1) token(%aux_3_w);
    v0_logical_tensor %out_1@2 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), token %out_1_w, token %$2_r = !10005 ASSEMBLE(%$2@5) token(%$2_w) #toOffset(Unsupported);
    return %a@0, %aux_3@1, %out_1@2;
}
"""


IR_6 = """
function foo incast(v0_logical_tensor %a@0 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), v0_logical_tensor %aux@1 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), v0_logical_tensor %out@2 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128])) outcast() #type(Opaque) #entry(false) {
    v0_logical_tensor %$0@3 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]) = !10010 VEC_DUP() token();
    v0_logical_tensor %aux_1@1 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), token %aux_1_w = !10011 ASSEMBLE(%$0@3) token() #toOffset(Unsupported);
    v0_logical_tensor %$1@4 #dtype(float) #shape([64, 128]) #offset([0, 0]) #dynvalidshape([64, 128]) = !10012 VEC_DUP() token();
    v0_logical_tensor %aux_3@1 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]) = !10013 ASSEMBLE(%$1@4) token(%aux_1_w) #toOffset(Unsupported);
    v0_logical_tensor %$2@5 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]) = !10014 ADD(%a@0, %aux_3@1) token();
    v0_logical_tensor %out_1@2 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]) = !10015 ASSEMBLE(%$2@5) token() #toOffset(Unsupported);
    return %a@0, %aux_3@1, %out_1@2;
}
"""


def test_assemble_scenario_3_overlapping_waw():
    def foo(a, aux, out):
        pypto.assemble(pypto.full([129, 128], 1.0, pypto.DT_FP32), [0, 0], aux)
        pypto.assemble(pypto.full([64, 128], 2.0, pypto.DT_FP32), [64, 0], aux)
        pypto.assemble(a + aux, [0, 0], out)

    shape = (129, 128)
    before_remove, after_remove = _run_passes(foo, _tensor(shape, "a"), _tensor(shape, "aux"), _tensor(shape, "out"))
    check_snapshot(before_remove, BEFORE_IR_6)
    check_snapshot(after_remove, IR_6)


BEFORE_IR_7 = """
function foo incast(v0_logical_tensor %a@0 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), v0_logical_tensor %aux@1 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), v0_logical_tensor %out@2 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128])) outcast() #type(Opaque) #entry(false) {
    v0_logical_tensor %$0@3 #dtype(float) #shape([32, 128]) #offset([0, 0]) #dynvalidshape([32, 128]), token %$0_w = !10000 VEC_DUP() token();
    v0_logical_tensor %aux_1@1 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), token %aux_1_w, token %$0_r = !10001 ASSEMBLE(%$0@3) token(%$0_w) #toOffset(Unsupported);
    v0_logical_tensor %$1@4 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), token %$1_w, token %aux_1_r = !10002 ADDS(%aux_1@1) token(%aux_1_w);
    v0_logical_tensor %$2@5 #dtype(float) #shape([32, 128]) #offset([0, 0]) #dynvalidshape([32, 128]), token %$2_w = !10003 VEC_DUP() token();
    v0_logical_tensor %aux_3@1 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), token %aux_3_w, token %$2_r = !10004 ASSEMBLE(%$2@5) token(%$2_w, %aux_1_r) #toOffset(Unsupported);
    v0_logical_tensor %$3@6 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), token %$3_w, token %aux_3_r = !10005 ADDS(%aux_3@1) token(%aux_3_w);
    v0_logical_tensor %$4@7 #dtype(float) #shape([64, 128]) #offset([0, 0]) #dynvalidshape([64, 128]), token %$4_w = !10006 VEC_DUP() token();
    v0_logical_tensor %aux_5@1 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), token %aux_5_w, token %$4_r = !10007 ASSEMBLE(%$4@7) token(%$4_w, %aux_3_r) #toOffset(Unsupported);
    v0_logical_tensor %$5@8 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), token %$5_w, token %$1_r, token %$3_r = !10008 ADD(%$1@4, %$3@6) token(%$1_w, %$3_w);
    v0_logical_tensor %$6@9 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), token %$6_w, token %$5_r, token %aux_5_r = !10009 ADD(%$5@8, %aux_5@1) token(%$5_w, %aux_5_w);
    v0_logical_tensor %out_1@2 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), token %out_1_w, token %$6_r = !10010 ASSEMBLE(%$6@9) token(%$6_w) #toOffset(Unsupported);
    return %a@0, %aux_5@1, %out_1@2;
}
"""


IR_7 = """
function foo incast(v0_logical_tensor %a@0 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), v0_logical_tensor %aux@1 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), v0_logical_tensor %out@2 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128])) outcast() #type(Opaque) #entry(false) {
    v0_logical_tensor %$0@3 #dtype(float) #shape([32, 128]) #offset([0, 0]) #dynvalidshape([32, 128]) = !10019 VEC_DUP() token();
    v0_logical_tensor %aux_1@1 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]) = !10020 ASSEMBLE(%$0@3) token() #toOffset(Unsupported);
    v0_logical_tensor %$1@4 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), token %aux_1_r = !10021 ADDS(%aux_1@1) token();
    v0_logical_tensor %$2@5 #dtype(float) #shape([32, 128]) #offset([0, 0]) #dynvalidshape([32, 128]) = !10022 VEC_DUP() token();
    v0_logical_tensor %aux_3@1 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]) = !10023 ASSEMBLE(%$2@5) token(%aux_1_r) #toOffset(Unsupported);
    v0_logical_tensor %$3@6 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), token %aux_3_r = !10024 ADDS(%aux_3@1) token();
    v0_logical_tensor %$4@7 #dtype(float) #shape([64, 128]) #offset([0, 0]) #dynvalidshape([64, 128]) = !10025 VEC_DUP() token();
    v0_logical_tensor %aux_5@1 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]) = !10026 ASSEMBLE(%$4@7) token(%aux_3_r) #toOffset(Unsupported);
    v0_logical_tensor %$5@8 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]) = !10027 ADD(%$1@4, %$3@6) token();
    v0_logical_tensor %$6@9 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]) = !10028 ADD(%$5@8, %aux_5@1) token();
    v0_logical_tensor %out_1@2 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]) = !10029 ASSEMBLE(%$6@9) token() #toOffset(Unsupported);
    return %a@0, %aux_5@1, %out_1@2;
}
"""


def test_assemble_scenario_4_all_versions_consumed():
    def foo(a, aux, out):
        pypto.assemble(pypto.full([32, 128], 1.0, pypto.DT_FP32), [0, 0], aux)
        value1 = aux + 1.0
        pypto.assemble(pypto.full([32, 128], 2.0, pypto.DT_FP32), [96, 0], aux)
        value2 = aux + 2.0
        pypto.assemble(pypto.full([64, 128], 3.0, pypto.DT_FP32), [16, 0], aux)
        pypto.assemble(value1 + value2 + aux, [0, 0], out)

    shape = (129, 128)
    before_remove, after_remove = _run_passes(foo, _tensor(shape, "a"), _tensor(shape, "aux"), _tensor(shape, "out"))
    check_snapshot(before_remove, BEFORE_IR_7)
    check_snapshot(after_remove, IR_7)


BEFORE_IR_8 = """
function foo incast(v0_logical_tensor %a@0 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), v0_logical_tensor %aux@1 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), v0_logical_tensor %out@2 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128])) outcast() #type(Opaque) #entry(false) {
    v0_logical_tensor %$0@3 #dtype(float) #shape([32, 128]) #offset([0, 0]) #dynvalidshape([32, 128]), token %$0_w = !10000 VEC_DUP() token();
    v0_logical_tensor %aux_1@1 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), token %aux_1_w, token %$0_r = !10001 ASSEMBLE(%$0@3) token(%$0_w) #toOffset(Unsupported);
    v0_logical_tensor %$1@4 #dtype(float) #shape([32, 128]) #offset([0, 0]) #dynvalidshape([32, 128]), token %$1_w = !10002 VEC_DUP() token();
    v0_logical_tensor %aux_3@1 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), token %aux_3_w, token %$1_r = !10003 ASSEMBLE(%$1@4) token(%$1_w, %aux_1_w) #toOffset(Unsupported);
    v0_logical_tensor %$2@5 #dtype(float) #shape([64, 128]) #offset([0, 0]) #dynvalidshape([64, 128]), token %$2_w = !10004 VEC_DUP() token();
    v0_logical_tensor %aux_5@1 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), token %aux_5_w, token %$2_r = !10005 ASSEMBLE(%$2@5) token(%$2_w, %aux_3_w) #toOffset(Unsupported);
    v0_logical_tensor %$3@6 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), token %$3_w, token %a_r, token %aux_5_r = !10006 ADD(%a@0, %aux_5@1) token(%aux_5_w);
    v0_logical_tensor %out_1@2 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), token %out_1_w, token %$3_r = !10007 ASSEMBLE(%$3@6) token(%$3_w) #toOffset(Unsupported);
    return %a@0, %aux_5@1, %out_1@2;
}
"""


IR_8 = """
function foo incast(v0_logical_tensor %a@0 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), v0_logical_tensor %aux@1 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), v0_logical_tensor %out@2 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128])) outcast() #type(Opaque) #entry(false) {
    v0_logical_tensor %$0@3 #dtype(float) #shape([32, 128]) #offset([0, 0]) #dynvalidshape([32, 128]) = !10013 VEC_DUP() token();
    v0_logical_tensor %aux_1@1 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), token %aux_1_w = !10014 ASSEMBLE(%$0@3) token() #toOffset(Unsupported);
    v0_logical_tensor %$1@4 #dtype(float) #shape([32, 128]) #offset([0, 0]) #dynvalidshape([32, 128]) = !10015 VEC_DUP() token();
    v0_logical_tensor %aux_3@1 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), token %aux_3_w = !10016 ASSEMBLE(%$1@4) token() #toOffset(Unsupported);
    v0_logical_tensor %$2@5 #dtype(float) #shape([64, 128]) #offset([0, 0]) #dynvalidshape([64, 128]) = !10017 VEC_DUP() token();
    v0_logical_tensor %aux_5@1 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]) = !10018 ASSEMBLE(%$2@5) token(%aux_1_w) #toOffset(Unsupported);
    v0_logical_tensor %$3@6 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]) = !10019 ADD(%a@0, %aux_5@1) token(%aux_3_w);
    v0_logical_tensor %out_1@2 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]) = !10020 ASSEMBLE(%$3@6) token() #toOffset(Unsupported);
    return %a@0, %aux_5@1, %out_1@2;
}
"""


def test_assemble_scenario_5_no_intermediate_version_consumed():
    def foo(a, aux, out):
        pypto.assemble(pypto.full([32, 128], 1.0, pypto.DT_FP32), [0, 0], aux)
        pypto.assemble(pypto.full([32, 128], 2.0, pypto.DT_FP32), [96, 0], aux)
        pypto.assemble(pypto.full([64, 128], 3.0, pypto.DT_FP32), [16, 0], aux)
        pypto.assemble(a + aux, [0, 0], out)

    shape = (129, 128)
    before_remove, after_remove = _run_passes(foo, _tensor(shape, "a"), _tensor(shape, "aux"), _tensor(shape, "out"))
    check_snapshot(before_remove, BEFORE_IR_8)
    check_snapshot(after_remove, IR_8)


BEFORE_IR_9 = """
function foo incast(v0_logical_tensor %a@0 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), v0_logical_tensor %aux@1 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), v0_logical_tensor %out@2 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128])) outcast() #type(Opaque) #entry(false) {
    v0_logical_tensor %$0@3 #dtype(float) #shape([32, 128]) #offset([0, 0]) #dynvalidshape([32, 128]), token %$0_w = !10000 VEC_DUP() token();
    v0_logical_tensor %aux_1@1 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), token %aux_1_w, token %$0_r = !10001 ASSEMBLE(%$0@3) token(%$0_w) #toOffset(Unsupported);
    v0_logical_tensor %$1@4 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), token %$1_w, token %aux_1_r = !10002 ADDS(%aux_1@1) token(%aux_1_w);
    v0_logical_tensor %$2@5 #dtype(float) #shape([32, 128]) #offset([0, 0]) #dynvalidshape([32, 128]), token %$2_w = !10003 VEC_DUP() token();
    v0_logical_tensor %aux_3@1 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), token %aux_3_w, token %$2_r = !10004 ASSEMBLE(%$2@5) token(%$2_w, %aux_1_r) #toOffset(Unsupported);
    v0_logical_tensor %$3@6 #dtype(float) #shape([64, 128]) #offset([0, 0]) #dynvalidshape([64, 128]), token %$3_w = !10005 VEC_DUP() token();
    v0_logical_tensor %aux_5@1 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), token %aux_5_w, token %$3_r = !10006 ASSEMBLE(%$3@6) token(%$3_w, %aux_3_w) #toOffset(Unsupported);
    v0_logical_tensor %$4@7 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), token %$4_w, token %$1_r, token %aux_5_r = !10007 ADD(%$1@4, %aux_5@1) token(%$1_w, %aux_5_w);
    v0_logical_tensor %out_1@2 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), token %out_1_w, token %$4_r = !10008 ASSEMBLE(%$4@7) token(%$4_w) #toOffset(Unsupported);
    return %a@0, %aux_5@1, %out_1@2;
}
"""


IR_9 = """
function foo incast(v0_logical_tensor %a@0 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), v0_logical_tensor %aux@1 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), v0_logical_tensor %out@2 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128])) outcast() #type(Opaque) #entry(false) {
    v0_logical_tensor %$0@3 #dtype(float) #shape([32, 128]) #offset([0, 0]) #dynvalidshape([32, 128]) = !10015 VEC_DUP() token();
    v0_logical_tensor %aux_1@1 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]) = !10016 ASSEMBLE(%$0@3) token() #toOffset(Unsupported);
    v0_logical_tensor %$1@4 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), token %aux_1_r = !10017 ADDS(%aux_1@1) token();
    v0_logical_tensor %$2@5 #dtype(float) #shape([32, 128]) #offset([0, 0]) #dynvalidshape([32, 128]) = !10018 VEC_DUP() token();
    v0_logical_tensor %aux_3@1 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), token %aux_3_w = !10019 ASSEMBLE(%$2@5) token(%aux_1_r) #toOffset(Unsupported);
    v0_logical_tensor %$3@6 #dtype(float) #shape([64, 128]) #offset([0, 0]) #dynvalidshape([64, 128]) = !10020 VEC_DUP() token();
    v0_logical_tensor %aux_5@1 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]) = !10021 ASSEMBLE(%$3@6) token(%aux_1_r) #toOffset(Unsupported);
    v0_logical_tensor %$4@7 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]) = !10022 ADD(%$1@4, %aux_5@1) token(%aux_3_w);
    v0_logical_tensor %out_1@2 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]) = !10023 ASSEMBLE(%$4@7) token() #toOffset(Unsupported);
    return %a@0, %aux_5@1, %out_1@2;
}
"""


def test_assemble_scenario_6_middle_version_not_consumed():
    def foo(a, aux, out):
        pypto.assemble(pypto.full([32, 128], 1.0, pypto.DT_FP32), [0, 0], aux)
        value = aux + 1.0
        pypto.assemble(pypto.full([32, 128], 2.0, pypto.DT_FP32), [96, 0], aux)
        pypto.assemble(pypto.full([64, 128], 3.0, pypto.DT_FP32), [16, 0], aux)
        pypto.assemble(value + aux, [0, 0], out)

    shape = (129, 128)
    before_remove, after_remove = _run_passes(foo, _tensor(shape, "a"), _tensor(shape, "aux"), _tensor(shape, "out"))
    check_snapshot(before_remove, BEFORE_IR_9)
    check_snapshot(after_remove, IR_9)


BEFORE_IR_10 = """
function foo incast(v0_logical_tensor %a@0 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), v0_logical_tensor %aux@1 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), v0_logical_tensor %out@2 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128])) outcast() #type(Opaque) #entry(false) {
    v0_logical_tensor %$0@3 #dtype(float) #shape([32, 128]) #offset([0, 0]) #dynvalidshape([32, 128]), token %$0_w = !10000 VEC_DUP() token();
    v0_logical_tensor %aux_1@1 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), token %aux_1_w, token %$0_r = !10001 ASSEMBLE(%$0@3) token(%$0_w) #toOffset(Unsupported);
    v0_logical_tensor %$1@4 #dtype(float) #shape([32, 128]) #offset([0, 0]) #dynvalidshape([32, 128]), token %$1_w = !10002 VEC_DUP() token();
    v0_logical_tensor %aux_3@1 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), token %aux_3_w, token %$1_r = !10003 ASSEMBLE(%$1@4) token(%$1_w, %aux_1_w) #toOffset(Unsupported);
    v0_logical_tensor %$2@5 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), token %$2_w, token %aux_3_r = !10004 ADDS(%aux_3@1) token(%aux_3_w);
    v0_logical_tensor %$3@6 #dtype(float) #shape([64, 128]) #offset([0, 0]) #dynvalidshape([64, 128]), token %$3_w = !10005 VEC_DUP() token();
    v0_logical_tensor %aux_5@1 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), token %aux_5_w, token %$3_r = !10006 ASSEMBLE(%$3@6) token(%$3_w, %aux_3_r) #toOffset(Unsupported);
    v0_logical_tensor %$4@7 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), token %$4_w, token %$2_r, token %aux_5_r = !10007 ADD(%$2@5, %aux_5@1) token(%$2_w, %aux_5_w);
    v0_logical_tensor %out_1@2 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), token %out_1_w, token %$4_r = !10008 ASSEMBLE(%$4@7) token(%$4_w) #toOffset(Unsupported);
    return %a@0, %aux_5@1, %out_1@2;
}
"""


IR_10 = """
function foo incast(v0_logical_tensor %a@0 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), v0_logical_tensor %aux@1 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), v0_logical_tensor %out@2 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128])) outcast() #type(Opaque) #entry(false) {
    v0_logical_tensor %$0@3 #dtype(float) #shape([32, 128]) #offset([0, 0]) #dynvalidshape([32, 128]) = !10015 VEC_DUP() token();
    v0_logical_tensor %aux_1@1 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), token %aux_1_w = !10016 ASSEMBLE(%$0@3) token() #toOffset(Unsupported);
    v0_logical_tensor %$1@4 #dtype(float) #shape([32, 128]) #offset([0, 0]) #dynvalidshape([32, 128]) = !10017 VEC_DUP() token();
    v0_logical_tensor %aux_3@1 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]) = !10018 ASSEMBLE(%$1@4) token() #toOffset(Unsupported);
    v0_logical_tensor %$2@5 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), token %aux_3_r = !10019 ADDS(%aux_3@1) token(%aux_1_w);
    v0_logical_tensor %$3@6 #dtype(float) #shape([64, 128]) #offset([0, 0]) #dynvalidshape([64, 128]) = !10020 VEC_DUP() token();
    v0_logical_tensor %aux_5@1 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]) = !10021 ASSEMBLE(%$3@6) token(%aux_3_r) #toOffset(Unsupported);
    v0_logical_tensor %$4@7 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]) = !10022 ADD(%$2@5, %aux_5@1) token();
    v0_logical_tensor %out_1@2 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]) = !10023 ASSEMBLE(%$4@7) token() #toOffset(Unsupported);
    return %a@0, %aux_5@1, %out_1@2;
}
"""


def test_assemble_scenario_7_first_version_not_consumed():
    def foo(a, aux, out):
        pypto.assemble(pypto.full([32, 128], 1.0, pypto.DT_FP32), [0, 0], aux)
        pypto.assemble(pypto.full([32, 128], 2.0, pypto.DT_FP32), [96, 0], aux)
        value = aux + 1.0
        pypto.assemble(pypto.full([64, 128], 3.0, pypto.DT_FP32), [16, 0], aux)
        pypto.assemble(value + aux, [0, 0], out)

    shape = (129, 128)
    before_remove, after_remove = _run_passes(foo, _tensor(shape, "a"), _tensor(shape, "aux"), _tensor(shape, "out"))
    check_snapshot(before_remove, BEFORE_IR_10)
    check_snapshot(after_remove, IR_10)


BEFORE_IR_11 = """
function foo incast(v0_logical_tensor %a@0 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), v0_logical_tensor %aux@1 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), v0_logical_tensor %out@2 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128])) outcast() #type(Opaque) #entry(false) {
    v0_logical_tensor %$0@3 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), token %$0_w, token %a_r, token %aux_r = !10000 ADD(%a@0, %aux@1) token();
    v0_logical_tensor %$1@4 #dtype(float) #shape([64, 128]) #offset([0, 0]) #dynvalidshape([64, 128]), token %$1_w = !10001 VEC_DUP() token();
    v0_logical_tensor %a_1@0 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), token %a_1_w, token %$1_r = !10002 ASSEMBLE(%$1@4) token(%$1_w, %a_r) #toOffset(Unsupported);
    v0_logical_tensor %$2@5 #dtype(float) #shape([64, 128]) #offset([0, 0]) #dynvalidshape([64, 128]), token %$2_w = !10003 VEC_DUP() token();
    v0_logical_tensor %aux_1@1 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), token %aux_1_w, token %$2_r = !10004 ASSEMBLE(%$2@5) token(%$2_w, %aux_r) #toOffset(Unsupported);
    v0_logical_tensor %out_1@2 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), token %out_1_w, token %$0_r = !10005 ASSEMBLE(%$0@3) token(%$0_w) #toOffset(Unsupported);
    return %a_1@0, %aux_1@1, %out_1@2;
}
"""


IR_11 = """
function foo incast(v0_logical_tensor %a@0 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), v0_logical_tensor %aux@1 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), v0_logical_tensor %out@2 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128])) outcast() #type(Opaque) #entry(false) {
    v0_logical_tensor %$0@3 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]), token %a_r, token %aux_r = !10009 ADD(%a@0, %aux@1) token();
    v0_logical_tensor %$1@4 #dtype(float) #shape([64, 128]) #offset([0, 0]) #dynvalidshape([64, 128]) = !10010 VEC_DUP() token();
    v0_logical_tensor %a_1@0 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]) = !10011 ASSEMBLE(%$1@4) token(%a_r) #toOffset(Unsupported);
    v0_logical_tensor %$2@5 #dtype(float) #shape([64, 128]) #offset([0, 0]) #dynvalidshape([64, 128]) = !10012 VEC_DUP() token();
    v0_logical_tensor %aux_1@1 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]) = !10013 ASSEMBLE(%$2@5) token(%aux_r) #toOffset(Unsupported);
    v0_logical_tensor %out_1@2 #dtype(float) #shape([129, 128]) #offset([0, 0]) #dynvalidshape([129, 128]) = !10014 ASSEMBLE(%$0@3) token() #toOffset(Unsupported);
    return %a_1@0, %aux_1@1, %out_1@2;
}
"""


def test_assemble_scenario_8_separate_writes_after_shared_read():
    def foo(a, aux, out):
        value = a + aux
        pypto.assemble(pypto.full([64, 128], 1.0, pypto.DT_FP32), [0, 0], a)
        pypto.assemble(pypto.full([64, 128], 2.0, pypto.DT_FP32), [0, 0], aux)
        pypto.assemble(value, [0, 0], out)

    shape = (129, 128)
    before_remove, after_remove = _run_passes(foo, _tensor(shape, "a"), _tensor(shape, "aux"), _tensor(shape, "out"))
    check_snapshot(before_remove, BEFORE_IR_11)
    check_snapshot(after_remove, IR_11)


BEFORE_IR_12 = """
function foo incast(v0_logical_tensor %src@0 #dtype(float) #shape([8, 16]) #offset([0, 0]) #dynvalidshape([8, 16]), v0_logical_tensor %aux@1 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16]), v0_logical_tensor %result@2 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16])) outcast() #type(Opaque) #entry(false) {
    v0_logical_tensor %aux_1@1 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16]), token %aux_1_w, token %src_r = !10000 ASSEMBLE(%src@0) token() #toOffset(Unsupported);
    v0_logical_tensor %aux_3@1 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16]), token %aux_3_w, token %src_r = !10001 ASSEMBLE(%src@0) token(%aux_1_w) #toOffset(Unsupported);
    v0_logical_tensor %aux_5@1 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16]), token %aux_5_w, token %src_r = !10002 ASSEMBLE(%src@0) token(%aux_3_w) #toOffset(Unsupported);
    v0_logical_tensor %$0@3 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16]), token %$0_w, token %aux_5_r = !10003 ADDS(%aux_5@1) token(%aux_5_w);
    return %src@0, %aux_5@1, %$0@3;
}
"""


IR_12 = """
function foo incast(v0_logical_tensor %src@0 #dtype(float) #shape([8, 16]) #offset([0, 0]) #dynvalidshape([8, 16]), v0_logical_tensor %aux@1 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16]), v0_logical_tensor %result@2 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16])) outcast() #type(Opaque) #entry(false) {
    v0_logical_tensor %aux_1@1 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16]), token %aux_1_w = !10007 ASSEMBLE(%src@0) token() #toOffset(Unsupported);
    v0_logical_tensor %aux_3@1 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16]), token %aux_3_w = !10008 ASSEMBLE(%src@0) token() #toOffset(Unsupported);
    v0_logical_tensor %aux_5@1 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16]) = !10009 ASSEMBLE(%src@0) token(%aux_1_w) #toOffset(Unsupported);
    v0_logical_tensor %$0@3 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16]) = !10010 ADDS(%aux_5@1) token(%aux_3_w);
    return %src@0, %aux_5@1, %$0@3;
}
"""


def test_rewire_four_op_token_chain():
    def foo(src, aux, result):
        # A: write aux[0:8].
        pypto.assemble(src, [0, 0], aux)
        # B: write aux[16:24].
        pypto.assemble(src, [16, 0], aux)
        # C: write aux[0:8].
        pypto.assemble(src, [0, 0], aux)
        # D: read aux.
        result = aux + 1.0  # noqa: F841

    before_remove, after_remove = _run_passes(
        foo,
        _tensor((8, 16), "src"),
        _tensor((32, 16), "aux"),
        _tensor((32, 16), "result"),
    )
    check_snapshot(before_remove, BEFORE_IR_12)
    check_snapshot(after_remove, IR_12)


BEFORE_IR_13 = """
function foo incast(v0_logical_tensor %src_a@0 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16]), v0_logical_tensor %src_b@1 #dtype(float) #shape([8, 16]) #offset([0, 0]) #dynvalidshape([8, 16]), v0_logical_tensor %src_c@2 #dtype(float) #shape([8, 16]) #offset([0, 0]) #dynvalidshape([8, 16]), v0_logical_tensor %aux@3 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16]), v0_logical_tensor %result@4 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16])) outcast() #type(Opaque) #entry(false) {
    v0_logical_tensor %aux_1@3 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16]), token %aux_1_w, token %src_a_r = !10000 ASSEMBLE(%src_a@0) token() #toOffset(Unsupported);
    v0_logical_tensor %aux_3@3 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16]), token %aux_3_w, token %src_b_r = !10001 ASSEMBLE(%src_b@1) token(%aux_1_w) #toOffset(Unsupported);
    v0_logical_tensor %aux_5@3 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16]), token %aux_5_w, token %src_c_r = !10002 ASSEMBLE(%src_c@2) token(%aux_3_w) #toOffset(Unsupported);
    v0_logical_tensor %$0@5 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16]), token %$0_w, token %aux_5_r = !10003 ADDS(%aux_5@3) token(%aux_5_w);
    return %src_a@0, %src_b@1, %src_c@2, %aux_5@3, %$0@5;
}
"""


IR_13 = """
function foo incast(v0_logical_tensor %src_a@0 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16]), v0_logical_tensor %src_b@1 #dtype(float) #shape([8, 16]) #offset([0, 0]) #dynvalidshape([8, 16]), v0_logical_tensor %src_c@2 #dtype(float) #shape([8, 16]) #offset([0, 0]) #dynvalidshape([8, 16]), v0_logical_tensor %aux@3 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16]), v0_logical_tensor %result@4 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16])) outcast() #type(Opaque) #entry(false) {
    v0_logical_tensor %aux_1@3 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16]), token %aux_1_w = !10006 ASSEMBLE(%src_a@0) token() #toOffset(Unsupported);
    v0_logical_tensor %aux_3@3 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16]), token %aux_3_w = !10007 ASSEMBLE(%src_b@1) token(%aux_1_w) #toOffset(Unsupported);
    v0_logical_tensor %aux_5@3 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16]) = !10008 ASSEMBLE(%src_c@2) token(%aux_1_w) #toOffset(Unsupported);
    v0_logical_tensor %$0@5 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16]) = !10009 ADDS(%aux_5@3) token(%aux_3_w);
    return %src_a@0, %src_b@1, %src_c@2, %aux_5@3, %$0@5;
}
"""


def test_rewire_middle_edge_to_diamond():
    def foo(src_a, src_b, src_c, aux, result):
        # A overlaps both B and C; B and C are disjoint.
        pypto.assemble(src_a, [0, 0], aux)   # A: write aux[0:32].
        pypto.assemble(src_b, [0, 0], aux)   # B: write aux[0:8].
        pypto.assemble(src_c, [24, 0], aux)  # C: write aux[24:32].
        result = aux + 1.0  # D: read the whole tensor.  # noqa: F841

    before_remove, after_remove = _run_passes(
        foo,
        _tensor((32, 16), "src_a"),
        _tensor((8, 16), "src_b"),
        _tensor((8, 16), "src_c"),
        _tensor((32, 16), "aux"),
        _tensor((32, 16), "result"),
    )
    check_snapshot(before_remove, BEFORE_IR_13)
    check_snapshot(after_remove, IR_13)


BEFORE_IR_14 = """
function foo incast(v0_logical_tensor %src@0 #dtype(float) #shape([8, 16]) #offset([0, 0]) #dynvalidshape([8, 16]), v0_logical_tensor %aux@1 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16]), v0_logical_tensor %out@2 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16])) outcast() #type(Opaque) #entry(false) {
    v0_logical_tensor %aux_1@1 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16]), token %aux_1_w, token %src_r = !10000 ASSEMBLE(%src@0) token() #toOffset(Unsupported);
    v0_logical_tensor %aux_3@1 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16]), token %aux_3_w, token %src_r = !10001 ASSEMBLE(%src@0) token(%aux_1_w) #toOffset(Unsupported);
    v0_logical_tensor %value_0@3 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16]), token %value_0_w_for = for %loop_idx_21 inrange 0, 2, 1 iter { none %value = %None; token %value_0_w_for_iter = %None; } #parallel(false) #submit_before_loop(false) #_loop_conds(Unsupported) #_config_scope(Unsupported) #unroll_times(1) {
        v0_logical_tensor %$0@3 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16]), token %$0_w, token %aux_3_r = !10002 ADDS(%aux_3@1) token(%aux_3_w);
        continue %$0@3, %$0_w;
    }
    v0_logical_tensor %out_1@2 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16]), token %out_1_w, token %value_0_r = !10003 ASSEMBLE(%value_0@3) token(%value_0_w_for) #toOffset(Unsupported);
    return %src@0, %aux_3@1, %out_1@2;
}
"""


IR_14 = """
function foo incast(v0_logical_tensor %src@0 #dtype(float) #shape([8, 16]) #offset([0, 0]) #dynvalidshape([8, 16]), v0_logical_tensor %aux@1 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16]), v0_logical_tensor %out@2 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16])) outcast() #type(Opaque) #entry(false) {
    v0_logical_tensor %aux_1@1 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16]), token %aux_1_w = !10004 ASSEMBLE(%src@0) token() #toOffset(Unsupported);
    v0_logical_tensor %aux_3@1 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16]), token %aux_3_w = !10005 ASSEMBLE(%src@0) token(%aux_1_w) #toOffset(Unsupported);
    v0_logical_tensor %value_0@3 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16]), token %value_0_w_for = for %loop_idx_21 inrange 0, 2, 1 iter { none %value = %None; token %value_0_w_for_iter = %None; } #parallel(false) #submit_before_loop(false) #_loop_conds(Unsupported) #_config_scope(Unsupported) #unroll_times(1) {
        v0_logical_tensor %$0@3 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16]), token %$0_w = !10006 ADDS(%aux_3@1) token(%aux_3_w);
        continue %$0@3, %$0_w;
    }
    v0_logical_tensor %out_1@2 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16]) = !10007 ASSEMBLE(%value_0@3) token(%value_0_w_for) #toOffset(Unsupported);
    return %src@0, %aux_3@1, %out_1@2;
}
"""


def test_keep_dependency_when_result_is_read_in_loop():
    def foo(src, aux, out):
        pypto.assemble(src, [0, 0], aux)
        pypto.assemble(src, [16, 0], aux)
        for _ in pypto.loop(2):
            value = aux + 1.0
        pypto.assemble(value, [0, 0], out)

    before_remove, after_remove = _run_passes(
        foo,
        _tensor((8, 16), "src"),
        _tensor((32, 16), "aux"),
        _tensor((32, 16), "out"),
    )
    check_snapshot(before_remove, BEFORE_IR_14)
    check_snapshot(after_remove, IR_14)


BEFORE_IR_15 = """
function foo incast(v0_logical_tensor %src@0 #dtype(float) #shape([8, 16]) #offset([0, 0]) #dynvalidshape([8, 16]), v0_logical_tensor %aux@1 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16]), v0_logical_tensor %out@2 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16])) outcast() #type(Opaque) #entry(false) {
    v0_logical_tensor %aux_7@1 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16]), token %aux_7_w_for = for %loop_idx_5 inrange 0, 2, 1 iter { v0_logical_tensor %aux_1@1 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16]) = %aux@1; token %aux_7_w_for_iter = %None; } #parallel(false) #submit_before_loop(false) #_loop_conds(Unsupported) #_config_scope(Unsupported) #unroll_times(1) {
        v0_logical_tensor %aux_3@1 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16]), token %aux_3_w, token %src_r = !10000 ASSEMBLE(%src@0) token() #toOffset(Unsupported);
        v0_logical_tensor %aux_5@1 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16]), token %aux_5_w, token %src_r = !10001 ASSEMBLE(%src@0) token(%aux_3_w) #toOffset(Unsupported);
        continue %aux_5@1, %aux_5_w;
    }
    v0_logical_tensor %$0@3 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16]), token %$0_w, token %aux_7_r = !10002 ADDS(%aux_7@1) token(%aux_7_w_for);
    v0_logical_tensor %out_1@2 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16]), token %out_1_w, token %$0_r = !10003 ASSEMBLE(%$0@3) token(%$0_w) #toOffset(Unsupported);
    return %src@0, %aux_7@1, %out_1@2;
}
"""


IR_15 = """
function foo incast(v0_logical_tensor %src@0 #dtype(float) #shape([8, 16]) #offset([0, 0]) #dynvalidshape([8, 16]), v0_logical_tensor %aux@1 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16]), v0_logical_tensor %out@2 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16])) outcast() #type(Opaque) #entry(false) {
    v0_logical_tensor %aux_7@1 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16]), token %aux_7_w_for = for %loop_idx_5 inrange 0, 2, 1 iter { v0_logical_tensor %aux_1@1 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16]) = %aux@1; token %aux_7_w_for_iter = %None; } #parallel(false) #submit_before_loop(false) #_loop_conds(Unsupported) #_config_scope(Unsupported) #unroll_times(1) {
        v0_logical_tensor %aux_3@1 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16]), token %aux_3_w = !10005 ASSEMBLE(%src@0) token() #toOffset(Unsupported);
        v0_logical_tensor %aux_5@1 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16]), token %aux_5_w = !10006 ASSEMBLE(%src@0) token(%aux_3_w) #toOffset(Unsupported);
        continue %aux_5@1, %aux_5_w;
    }
    v0_logical_tensor %$0@3 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16]) = !10007 ADDS(%aux_7@1) token(%aux_7_w_for);
    v0_logical_tensor %out_1@2 #dtype(float) #shape([32, 16]) #offset([0, 0]) #dynvalidshape([32, 16]) = !10008 ASSEMBLE(%$0@3) token() #toOffset(Unsupported);
    return %src@0, %aux_7@1, %out_1@2;
}
"""


def test_keep_dependency_when_loop_writes_are_read_after_loop():
    def foo(src, aux, out):
        for _ in pypto.loop(2):
            pypto.assemble(src, [0, 0], aux)
            pypto.assemble(src, [16, 0], aux)
        pypto.assemble(aux + 1.0, [0, 0], out)

    before_remove, after_remove = _run_passes(
        foo,
        _tensor((8, 16), "src"),
        _tensor((32, 16), "aux"),
        _tensor((32, 16), "out"),
    )
    check_snapshot(before_remove, BEFORE_IR_15)
    check_snapshot(after_remove, IR_15)
