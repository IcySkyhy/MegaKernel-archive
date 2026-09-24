# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Comprehensive tests for ir.Pass.aggressive_dce."""

import logging

import pypto
from pypto import ir, pil


def _collect_stmts(stmt, cls):
    """Recursively collect all stmts of a given IR type."""
    result = []
    if isinstance(stmt, cls):
        result.append(stmt)
    if isinstance(stmt, ir.SeqStmts):
        for s in stmt.stmts:
            result.extend(_collect_stmts(s, cls))
    if isinstance(stmt, ir.ForStmt):
        result.extend(_collect_stmts(stmt.body, cls))
    if isinstance(stmt, ir.IfStmt):
        result.extend(_collect_stmts(stmt.then_body, cls))
        if stmt.else_body is not None:
            result.extend(_collect_stmts(stmt.else_body, cls))
    if isinstance(stmt, ir.WhileStmt):
        result.extend(_collect_stmts(stmt.body, cls))
    return result


def _run_dce(func, *args):
    """Build a program from a compiled function and run aggressive DCE."""
    b = ir.IRBuilder()
    func = pil.compile(func, *args)
    prog = b.create_program([func], "main", ir.Span.unknown())
    dce = ir.Pass.aggressive_dce()
    canonical = ir.Pass.canonicalize()
    logging.info("\norigin: %s\n", prog)
    prog = canonical(prog)
    logging.info("\ncanonical: %s\n", prog)
    prog = dce(prog)
    logging.info("\ndce: %s\n", prog)
    prog = dce(canonical(prog))
    logging.info("\ndce canonical: %s\n", prog)

    return prog.functions[func.name]


# ---------- TensorOpStmt chain propagation ----------


def test_dce_tensor_chain_view_adds_assemble():
    """VIEW -> ADDS -> ASSEMBLE: all three must be preserved.

    The original bug: DCE only propagated liveness through AssignStmts,
    so TensorOpStmt chains were incorrectly treated as dead.
    """

    def foo(x, y):
        for i in pypto.loop(x.shape[0] // 32):
            pypto.set_vec_tile_shapes(32, 32)
            ta = x[i:i + 32, :]
            y[i:, :] = ta + 1

    x = pypto.Tensor((-1, 32), pypto.DT_FP32, 'x')
    y = pypto.Tensor((-1, 32), pypto.DT_FP32, 'y')
    func = _run_dce(foo, x, y)

    tensor_ops = _collect_stmts(func.body, ir.TensorOpStmt)
    opcodes = [op.opcode for op in tensor_ops]
    assert 'VIEW' in opcodes, f"VIEW was incorrectly eliminated: {opcodes}"
    assert 'ADDS' in opcodes, f"ADDS was incorrectly eliminated: {opcodes}"
    assert 'ASSEMBLE' in opcodes, f"ASSEMBLE was incorrectly eliminated: {opcodes}"


def test_dce_dead_tensor_op_removed():
    """An unused TensorOpStmt should be removed by aggressive DCE."""

    def foo(x, y):
        for i in pypto.loop(x.shape[0] // 32):
            pypto.set_vec_tile_shapes(32, 32)
            ta = x[i:i + 32, :]
            _ = ta + 2  # dead op: result unused
            y[i:, :] = ta + 1

    x = pypto.Tensor((-1, 32), pypto.DT_FP32, 'x')
    y = pypto.Tensor((-1, 32), pypto.DT_FP32, 'y')
    func = _run_dce(foo, x, y)

    tensor_ops = _collect_stmts(func.body, ir.TensorOpStmt)
    adds_count = sum(1 for op in tensor_ops if op.opcode == 'ADDS')
    assert adds_count == 1, f"Expected 1 ADDS, got {adds_count}"


def test_dce_long_tensor_chain():
    """A -> B -> C -> ASSEMBLE: entire chain must be preserved.

    Tests multi-step propagation through TensorOpStmts.
    """

    def foo(x, y):
        for i in pypto.loop(x.shape[0] // 32):
            pypto.set_vec_tile_shapes(32, 32)
            ta = x[i:i + 32, :]
            tb = ta + 1  # step 1
            tc = tb + 1  # step 2
            y[i:, :] = tc  # ASSEMBLE

    x = pypto.Tensor((-1, 32), pypto.DT_FP32, 'x')
    y = pypto.Tensor((-1, 32), pypto.DT_FP32, 'y')
    func = _run_dce(foo, x, y)

    tensor_ops = _collect_stmts(func.body, ir.TensorOpStmt)
    opcodes = [op.opcode for op in tensor_ops]
    adds_count = sum(1 for c in opcodes if c == 'ADDS')
    assert adds_count == 2, f"Expected 2 ADDS, got {adds_count} in {opcodes}"
    assert 'VIEW' in opcodes, f"VIEW was incorrectly eliminated: {opcodes}"
    assert 'ASSEMBLE' in opcodes, f"ASSEMBLE was incorrectly eliminated: {opcodes}"


def test_dce_branching_tensor_chain():
    """A -> B -> C -> ASSEMBLE and A -> D (dead): only A,B,C,ASSEMBLE kept."""

    def foo(x, y):
        for i in pypto.loop(x.shape[0] // 32):
            pypto.set_vec_tile_shapes(32, 32)
            ta = x[i:i + 32, :]
            tb = ta + 1
            tc = tb + 1
            _ = ta + 99  # dead branch: result unused
            y[i:, :] = tc

    x = pypto.Tensor((-1, 32), pypto.DT_FP32, 'x')
    y = pypto.Tensor((-1, 32), pypto.DT_FP32, 'y')
    func = _run_dce(foo, x, y)

    tensor_ops = _collect_stmts(func.body, ir.TensorOpStmt)
    opcodes = [op.opcode for op in tensor_ops]
    adds_count = sum(1 for c in opcodes if c == 'ADDS')
    assert adds_count == 2, f"Expected 2 ADDS (dead branch removed), got {adds_count} in {opcodes}"


# ---------- ContinueStmt / BreakStmt live-root ----------


def test_dce_keeps_tensor_op():
    """A TensorOpStmt whose result is used by ContinueStmt must survive."""

    def foo(x, y):
        for i in pypto.loop(x.shape[0] // 32):
            pypto.set_vec_tile_shapes(32, 32)
            ta = x[i:i + 32, :]
            tb = ta + 1
            y[i:, :] = tb

    x = pypto.Tensor((-1, 32), pypto.DT_FP32, 'x')
    y = pypto.Tensor((-1, 32), pypto.DT_FP32, 'y')
    func = _run_dce(foo, x, y)

    tensor_ops = _collect_stmts(func.body, ir.TensorOpStmt)
    opcodes = [op.opcode for op in tensor_ops]
    assert 'VIEW' in opcodes, f"VIEW eliminated but used by continue: {opcodes}"


# ---------- Nested control flow ----------


def test_dce_nested_for_dead_inner():
    """Dead tensor ops inside a for-loop body should be eliminated."""

    def foo(x, y):
        for i in pypto.loop(x.shape[0] // 32):
            pypto.set_vec_tile_shapes(32, 32)
            ta = x[i:i + 32, :]
            _ = ta + 99  # dead: result unused
            y[i:, :] = ta + 1

    x = pypto.Tensor((-1, 32), pypto.DT_FP32, 'x')
    y = pypto.Tensor((-1, 32), pypto.DT_FP32, 'y')
    func = _run_dce(foo, x, y)

    tensor_ops = _collect_stmts(func.body, ir.TensorOpStmt)
    opcodes = [op.opcode for op in tensor_ops]
    adds_count = sum(1 for c in opcodes if c == 'ADDS')
    assert adds_count == 1, f"Expected 1 ADDS, got {adds_count} in {opcodes}"


def test_dce_if_else_branches():
    """Dead code in if/else branches should be eliminated independently."""

    def foo(x, y):
        for i in pypto.loop(x.shape[0] // 32):
            pypto.set_vec_tile_shapes(32, 32)
            ta = x[i:i + 32, :]
            if i < 2:
                _ = ta + 99  # dead in then-branch
                tb = ta + 1
                y[i:, :] = tb
            else:
                _ = ta + 88  # dead in else-branch
                tb = ta + 2
                y[i:, :] = tb

    x = pypto.Tensor((-1, 32), pypto.DT_FP32, 'x')
    y = pypto.Tensor((-1, 32), pypto.DT_FP32, 'y')
    func = _run_dce(foo, x, y)

    tensor_ops = _collect_stmts(func.body, ir.TensorOpStmt)
    opcodes = [op.opcode for op in tensor_ops]
    # Dead ADDS (ta+99, ta+88) removed; live ADDS (ta+1, ta+2) kept
    adds_count = sum(1 for c in opcodes if c == 'ADDS')
    assert adds_count == 2, f"Expected 2 ADDS, got {adds_count} in {opcodes}"


# ---------- Mixed AssignStmt + TensorOpStmt ----------


def test_dce_nested_if_else():
    """Dead code in nested if/else should be eliminated at every level."""

    def foo(x, y):
        for i in pypto.loop(x.shape[0] // 32):
            pypto.set_vec_tile_shapes(32, 32)
            ta = x[i:i + 32, :]
            if i < 2:
                _ = ta + 99  # dead in then-branch
                if i == 0:
                    _ = ta + 77  # dead in nested then
                    tb = ta + 1
                else:
                    _ = ta + 66  # dead in nested else
                    tb = ta + 2
                y[i:, :] = tb
            else:
                _ = ta + 88  # dead in else-branch
                tb = ta + 3
                y[i:, :] = tb

    x = pypto.Tensor((-1, 32), pypto.DT_FP32, 'x')
    y = pypto.Tensor((-1, 32), pypto.DT_FP32, 'y')
    func = _run_dce(foo, x, y)

    tensor_ops = _collect_stmts(func.body, ir.TensorOpStmt)
    opcodes = [op.opcode for op in tensor_ops]
    # Dead ADDS (ta+99, ta+77, ta+66, ta+88) removed; live ADDS (ta+1, ta+2, ta+3) kept
    adds_count = sum(1 for c in opcodes if c == 'ADDS')
    assert adds_count == 3, f"Expected 3 ADDS, got {adds_count} in {opcodes}"


def test_dce_nested_loop_if_else():
    """Dead code in if/else inside nested loops should be eliminated."""

    def foo(x, y):
        for i in pypto.loop(x.shape[0] // 32):
            pypto.set_vec_tile_shapes(32, 32)
            ta = x[i * 32:(i + 1) * 32, :]
            for j in pypto.loop(2):
                pypto.set_vec_tile_shapes(16, 16)
                tb = ta[j * 16:(j + 1) * 16, :]
                if j == 0:
                    _ = tb + 99  # dead in inner then
                    tc = tb + 1
                else:
                    _ = tb + 88  # dead in inner else
                    tc = tb + 2
                y[i * 32 + j * 16:i * 32 + (j + 1) * 16, :] = tc

    x = pypto.Tensor((-1, 32), pypto.DT_FP32, 'x')
    y = pypto.Tensor((-1, 32), pypto.DT_FP32, 'y')
    func = _run_dce(foo, x, y)

    tensor_ops = _collect_stmts(func.body, ir.TensorOpStmt)
    opcodes = [op.opcode for op in tensor_ops]
    # Each inner iteration has: VIEW, dead ADDS removed, live ADDS kept, ASSEMBLE
    adds_count = sum(1 for c in opcodes if c == 'ADDS')
    assert adds_count == 2, f"Expected 2 ADDS, got {adds_count} in {opcodes}"

    """Dead code in if/else branch containing break should be eliminated."""

    def foo(x, y):
        for i in pypto.loop(x.shape[0] // 32):
            pypto.set_vec_tile_shapes(32, 32)
            ta = x[i:i + 32, :]
            if i < 2:
                _ = ta + 99  # dead
                tb = ta + 1
                y[i:, :] = tb
            else:
                _ = ta + 88  # dead
                tb = ta + 2
                y[i:, :] = tb

    x = pypto.Tensor((-1, 32), pypto.DT_FP32, 'x')
    y = pypto.Tensor((-1, 32), pypto.DT_FP32, 'y')
    func = _run_dce(foo, x, y)

    tensor_ops = _collect_stmts(func.body, ir.TensorOpStmt)
    opcodes = [op.opcode for op in tensor_ops]
    adds_count = sum(1 for c in opcodes if c == 'ADDS')
    assert adds_count == 2, f"Expected 2 ADDS, got {adds_count} in {opcodes}"


# ---------- Mixed AssignStmt + TensorOpStmt ----------


def test_dce_mixed_assign_and_tensor():
    """Dead scalar assignments and dead tensor ops should both be removed."""

    def foo(x, y, n):
        for i in pypto.loop(n):
            pypto.set_vec_tile_shapes(32, 32)
            ta = x[i * 32:(i + 1) * 32, :]
            dead_scalar = i * 42  # dead scalar assign
            _ = dead_scalar
            _ = ta + 99  # dead tensor op
            tb = ta + 1
            y[i * 32:(i + 1) * 32, :] = tb

    x = pypto.Tensor((-1, 32), pypto.DT_FP32, 'x')
    y = pypto.Tensor((-1, 32), pypto.DT_FP32, 'y')
    func = _run_dce(foo, x, y, 10)

    tensor_ops = _collect_stmts(func.body, ir.TensorOpStmt)
    opcodes = [op.opcode for op in tensor_ops]
    adds_count = sum(1 for c in opcodes if c == 'ADDS')
    assert adds_count == 1, f"Expected 1 ADDS (dead one removed), got {adds_count} in {opcodes}"


def test_dce_no_assemble_stmt():
    """Dead scalar assignments and dead tensor ops should both be removed."""

    def foo(x, y, n):
        for i in pypto.loop(n):
            pypto.set_vec_tile_shapes(32, 32)
            if i > 0:
                ta = x[i * 32:(i + 1) * 32, :]
            else:
                ta = x[i * 32:(i + 1) * 32, :]
            dead_scalar = i * 42  # dead scalar assign
            _ = dead_scalar
            _ = ta + 99  # dead tensor op
            _tb = ta + 1

    x = pypto.Tensor((-1, 32), pypto.DT_FP32, 'x')
    y = pypto.Tensor((-1, 32), pypto.DT_FP32, 'y')
    func = _run_dce(foo, x, y, 10)
    assert len(func.body.stmts) == 1, "Expected only return stmt after DCE"


def test_forstmt_attrs_and_step_name_preserved():
    """ForStmt attrs and step-based path func naming must survive DCE/canonicalize/TransformStmts."""

    def foo(x, z):
        pypto.set_vec_tile_shapes(16, 16)
        for i in pypto.loop(0, 4, name="LOOP_TEST", parallel=True, unroll_list=[2]):
            x_view = pypto.view(x, [16, 32], [i * 16, 0])
            pypto.assemble(x_view, [i * 16, 0], z)

    x = pypto.Tensor(shape=[64, 32], dtype=pypto.DT_FP32, name="x")
    z = pypto.Tensor(shape=[64, 32], dtype=pypto.DT_FP32, name="z")

    b = ir.IRBuilder()
    func = pil.compile(foo, x, z)
    prog = b.create_program([func], "main", ir.Span.unknown())
    prog = ir.Pass.canonicalize()(prog)
    prog = ir.Pass.aggressive_dce()(prog)
    prog = ir.Pass.create_root_functions()(prog)

    # 1. ForStmt exists (attrs preserved, no crash)
    dyn_func = prog.functions[func.name]
    for_stmts = _collect_stmts(dyn_func.body, ir.ForStmt)
    assert len(for_stmts) >= 1, "Expected at least one ForStmt after create_root_functions"

    # 2. path function name contains step value
    prog_str = str(prog)
    assert "_Unroll2" in prog_str, f"Expected '_Unroll2' (step=2) in program: {prog_str}"


def test_tensor_move():

    def foo(a, b):
        last = pypto.full(a.shape, 1.0, a.dtype)
        for _ in pypto.loop(10):
            a = last + 1
            last[:] = a + 1
            b[0:, 0:] = last
        b[:] = a + 1

    x = pypto.Tensor((32, 32), pypto.DT_FP32, 'x')
    y = pypto.Tensor((32, 32), pypto.DT_FP32, 'y')
    func = _run_dce(foo, x, y)
    for_stmt = func.body[1]
    assert isinstance(for_stmt, ir.ForStmt)
    assert "last" in [v.iterVar.name for v in for_stmt.iter_args]
    add_stmt = func.body[2]
    assert isinstance(add_stmt, ir.TensorOpStmt)
    assert add_stmt.opcode == "ADDS"
    assert add_stmt.result[0].name == 'b_0'


def test_tensor_move2():

    def foo(a, b):
        last = pypto.full(a.shape, 1.0, a.dtype)
        for _ in pypto.loop(10):
            a = last + 1
            last.move(a + 1)
            b[0:, 0:] = last
        b.move(last + 1)

    x = pypto.Tensor((32, 32), pypto.DT_FP32, 'x')
    y = pypto.Tensor((32, 32), pypto.DT_FP32, 'y')
    func = _run_dce(foo, x, y)
    for_stmt = func.body[1]
    assert isinstance(for_stmt, ir.ForStmt)
    add_stmt = func.body[2]
    assert isinstance(add_stmt, ir.TensorOpStmt)
    assert add_stmt.opcode == "ADDS"
    assert add_stmt.result[0].name == 'b_0'


def test_tensor_move3():

    def bar(a):
        a.fill_(1.0)

    def foo(a):
        bar(a)

    x = pypto.Tensor((32, 32), pypto.DT_FP32, 'x')
    func = _run_dce(foo, x)
    assert len(func.body) == 2  # VEC_DUP + RETURN


def test_dce_reshape_inplace1():
    def foo(a, b):
        a1 = pypto.reshape(a, [32, 32], inplace=True)
        for _ in pypto.loop(1):
            a1[0:, 0:] = b + 1  # a1 reshaped from input, could not be deleted
    x = pypto.Tensor((32, 32), pypto.DT_FP32)
    y = pypto.Tensor((32, 32), pypto.DT_FP32)
    func = _run_dce(foo, x, y)
    for_stmt = func.body[1]
    assert isinstance(for_stmt, ir.ForStmt)
    stmt = for_stmt.body[1]
    assert isinstance(stmt, ir.TensorOpStmt)
    assert stmt.opcode == "ASSEMBLE"


def test_dce_reshape_inplace3():
    def foo(a, b):
        a1 = a + 1
        a2 = pypto.reshape(a1, [32, 32], inplace=True)
        a2[:] = b + 1  # a2 not reshaped from input, could be deleted
    x = pypto.Tensor((32, 32), pypto.DT_FP32)
    y = pypto.Tensor((32, 32), pypto.DT_FP32)
    func = _run_dce(foo, x, y)
    assert len(func.body) == 1


def test_dce_reshape_inplace4():
    def foo(t0, t1, out):
        pypto.set_vec_tile_shapes(32, 32)
        out[:] = pypto.add(t0, t1)
        for _ in pypto.loop(2, name="L02", idx_name="idx3"):
            t0[:] = pypto.add(t0, t1)
            out[:] = pypto.add(t0, out)

    x = pypto.Tensor((32, 32), pypto.DT_FP32)
    y = pypto.Tensor((32, 32), pypto.DT_FP32)
    z = pypto.Tensor((32, 32), pypto.DT_FP32)
    func = _run_dce(foo, x, y, z)
    for_stmt = func.body[1]
    stmt3 = for_stmt.body[-1]
    assert isinstance(stmt3, ir.ContinueStmt)
    assert len(stmt3.value) == 2  # t0 and out both


# ---------- Standalone helper carry propagation ----------
def _store_carry_helper(a, out):
    for i in pypto.loop(a.shape[0] // 32):
        pypto.set_vec_tile_shapes(32, 32)
        tile = a[i * 32:(i + 1) * 32, :]
        out[:] = tile + 1


def _store_carry_helper1(a, out):
    _store_carry_helper(a, out)


def test_canonicalize_keeps_helper_returnvar():
    """Ensure helper modified tensor return vars survive canonicalize."""

    def foo(a, out):
        _store_carry_helper(a, out)

    a = pypto.Tensor((64, 32), pypto.DT_FP32, 'a')
    out = pypto.Tensor((32, 32), pypto.DT_FP32, 'out')

    b = ir.IRBuilder()
    func = pil.compile(foo, a, out)
    prog = b.create_program([func], "main", ir.Span.unknown())
    prog = ir.Pass.canonicalize()(prog)

    main_func = prog.functions[func.name]
    for_stmts = _collect_stmts(main_func.body, ir.ForStmt)
    assert len(for_stmts) == 1
    assert len(for_stmts[0].return_vars) > 0, \
        "carry return_vars removed by canonicalize — scope propagation missing"


def test_canonicalize_keeps_helper_returnvar2():
    """Carry must propagate through a chain of standalone helper calls."""

    def foo(a, out):
        _store_carry_helper1(a, out)

    a = pypto.Tensor((64, 32), pypto.DT_FP32, 'a')
    out = pypto.Tensor((32, 32), pypto.DT_FP32, 'out')

    b = ir.IRBuilder()
    func = pil.compile(foo, a, out)
    prog = b.create_program([func], "main", ir.Span.unknown())
    prog = ir.Pass.canonicalize()(prog)

    main_func = prog.functions[func.name]
    for_stmts = _collect_stmts(main_func.body, ir.ForStmt)
    assert len(for_stmts) == 1
    assert len(for_stmts[0].return_vars) > 0, \
        "carry return_vars removed by canonicalize — nested propagation missing"


def test_canonicalize_keeps_helper_returnvar3():
    """Carry must propagate when caller variable names differ from helper params."""

    def foo(x, y):
        _store_carry_helper(x, y)

    x = pypto.Tensor((64, 32), pypto.DT_FP32, 'x')
    y = pypto.Tensor((32, 32), pypto.DT_FP32, 'y')

    b = ir.IRBuilder()
    func = pil.compile(foo, x, y)
    prog = b.create_program([func], "main", ir.Span.unknown())
    prog = ir.Pass.canonicalize()(prog)

    main_func = prog.functions[func.name]
    for_stmts = _collect_stmts(main_func.body, ir.ForStmt)
    assert len(for_stmts) == 1
    assert len(for_stmts[0].return_vars) > 0, \
        "carry return_vars removed by canonicalize — non-same-name propagation missing"
