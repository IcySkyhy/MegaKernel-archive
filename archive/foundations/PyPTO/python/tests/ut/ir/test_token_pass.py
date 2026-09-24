# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Tests for ir.Pass.token_pass (WAR/WAW token dependency insertion)."""

import pypto
from pypto import ir, pil


def _run_token_pass(func, *args):
    """Compile a kernel, wrap it in a program and run TokenPass."""
    func = pil.compile(func, *args)
    b = ir.IRBuilder()
    prog = b.create_program([func], "main", ir.Span.unknown())
    prog = ir.Pass.token_pass()(prog)
    prog = ir.Pass.aggressive_dce()(prog)
    prog = ir.Pass.canonicalize()(prog)
    return prog.functions[func.name]


def _tensor_ops(func):
    return [op for op in func.body if isinstance(op, ir.TensorOpStmt)]


def _assembles(func):
    return [op for op in _tensor_ops(func) if op.opcode == "ASSEMBLE"]


# ---------- Write-After-Write ----------


def test_waw_conflict():
    """Two overlapping Assembles into the same output -> WAW token edge."""

    def foo(x, y):
        pypto.set_vec_tile_shapes(16, 16)
        a = pypto.view(x, [16, 16], [0, 0])
        pypto.assemble(a, [0, 0], y)  # write y rows [0, 16)
        pypto.assemble(a, [8, 0], y)  # write y rows [8, 24) -> overlaps [8, 16)

    x = pypto.Tensor(shape=(32, 16), dtype=pypto.DT_FP32, name="x")
    y = pypto.Tensor(shape=(32, 16), dtype=pypto.DT_FP32, name="y")
    func = _run_token_pass(foo, x, y)

    ops = _assembles(func)
    assert len(ops) == 2
    assert ops[0].result_token
    assert ops[0].result_token[0] in ops[1].tokens


def test_waw_no_conflict():
    """Two disjoint Assembles -> no WAW token edge."""

    def foo(x, y):
        pypto.set_vec_tile_shapes(16, 16)
        a = pypto.view(x, [16, 16], [0, 0])
        pypto.assemble(a, [0, 0], y)  # write y rows [0, 16)
        pypto.assemble(a, [16, 0], y)  # write y rows [16, 32) -> disjoint

    x = pypto.Tensor(shape=(32, 16), dtype=pypto.DT_FP32, name="x")
    y = pypto.Tensor(shape=(32, 16), dtype=pypto.DT_FP32, name="y")
    func = _run_token_pass(foo, x, y)

    ops = _assembles(func)
    assert len(ops) == 2
    # No edge created: the first write produced no token consumed by the second.
    assert not ops[0].result_token


# ---------- Write-After-Read ----------


def test_war_conflict():
    """A read of y followed by an overlapping write of y -> WAR token edge."""

    def foo(x, y):
        pypto.set_vec_tile_shapes(16, 16)
        ry = pypto.view(y, [16, 16], [0, 0])  # read y rows [0, 16)
        a = pypto.view(x, [16, 16], [0, 0])
        s = pypto.add(ry, a)  # live read of ry
        pypto.assemble(s, [8, 0], y)  # write y rows [8, 24) -> overlaps [8, 16)

    x = pypto.Tensor(shape=(32, 16), dtype=pypto.DT_FP32, name="x")
    y = pypto.Tensor(shape=(32, 16), dtype=pypto.DT_FP32, name="y")
    func = _run_token_pass(foo, x, y)

    ops = _assembles(func)
    assert len(ops) == 1
    # The write must depend on the prior overlapping read via a token.
    assert len(ops[0].tokens) > 0


def test_war_no_conflict():
    """A read of y followed by a disjoint write of y -> no WAR token edge."""

    def foo(x, y):
        pypto.set_vec_tile_shapes(16, 16)
        ry = pypto.view(y, [16, 16], [0, 0])  # read y rows [0, 16)
        a = pypto.view(x, [16, 16], [0, 0])
        s = pypto.add(ry, a)  # live read of ry
        pypto.assemble(s, [16, 0], y)  # write y rows [16, 32) -> disjoint

    x = pypto.Tensor(shape=(32, 16), dtype=pypto.DT_FP32, name="x")
    y = pypto.Tensor(shape=(32, 16), dtype=pypto.DT_FP32, name="y")
    func = _run_token_pass(foo, x, y)

    ops = _assembles(func)
    assert len(ops) == 1
    assert len(ops[0].tokens) == 0


def test_scalar_dependency():
    """A read of y followed by a disjoint write of y -> no WAR token edge."""

    def foo(x, y):
        pypto.set_vec_tile_shapes(16, 16)
        off = pypto.arange(4)[0]
        a = pypto.view(x, [16, 16], [off, 0])  # a depends off, should have a token
        s = pypto.add(a, a)
        pypto.assemble(s, [off, 0], y)  # y depends off, should have a token

    x = pypto.Tensor(shape=(32, 16), dtype=pypto.DT_FP32, name="x")
    y = pypto.Tensor(shape=(32, 16), dtype=pypto.DT_FP32, name="y")
    func = _run_token_pass(foo, x, y)

    ops = _assembles(func)
    assert len(ops) == 2
    assert ops[0].result_token  # assemble used generate by get_tensor_data
    assert len(ops[1].tokens) == 1


if __name__ == "__main__":
    test_waw_conflict()
    test_waw_no_conflict()
    test_war_conflict()
    test_war_no_conflict()
    test_scalar_dependency()
