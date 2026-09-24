#Copyright(c) 2026 Huawei Technologies Co., Ltd.
#This program is free software, you can redistribute it and / or modify it under the terms and conditions of
#CANN Open Software License Agreement Version 2.0(the "License").
#Please refer to the License for details.You may not use this file except in compliance with the License.
#THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
#INCLUDING BUT NOT LIMITED TO NON - INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
#See LICENSE in the root of the software repository for the full text of the License.
"""
Tests targeting RootFunctionBuilder (ir_func_builder.cpp) coverage gaps.
"""

import pypto
from pypto import ir, pil


def _collect_stmts(stmt, cls):
    result = []
    if stmt is None:
        return result
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
    return result


def _run_root_only(func, *args):
    """Run canonicalize + dce + create_root_functions (skip finalize).

    Returns (program, dyn_func) so the caller can inspect the IR
    structure with CALL placeholders before finalization.
    """
    b = ir.IRBuilder()
    compiled = pil.compile(func, *args)
    prog = b.create_program([compiled], "main", ir.Span.unknown())
    prog = ir.Pass.canonicalize()(prog)
    prog = ir.Pass.aggressive_dce()(prog)
    prog = ir.Pass.create_root_functions()(prog)
    return prog, prog.functions[compiled.name]


def _assert_func_counts(prog, expected_hiddenfunc, expected_funcs=None):
    """Assert the program contains exactly *expected_hiddenfunc* hidden funcs
    and the expected total number of functions.

    Each pure-tensor-op segment in TransformStmts produces one hiddenFunc
    (named with ``_hiddenfunc`` suffix).  Each hiddenFunc is paired with a
    pathFunc, so the total function count is:

        len(prog.functions) = 1 (dynFunc) + 2 * hiddenfunc_count
    """
    if expected_funcs is None:
        expected_funcs = 1 + 2 * expected_hiddenfunc
    hiddenfunc_names = [k for k in prog.functions if "_hiddenfunc" in k]
    actual_hidden = len(hiddenfunc_names)
    actual_funcs = len(prog.functions)
    assert actual_hidden == expected_hiddenfunc, (
        f"Expected {expected_hiddenfunc} hidden funcs, got {actual_hidden}.\nProgram:\n{prog}"
    )
    assert actual_funcs == expected_funcs, (
        f"Expected {expected_funcs} total functions, got {actual_funcs}.\nProgram:\n{prog}"
    )


def test_incast_outcast_correctness():
    """Loop body: view(a) + view(b) -> add -> assemble into c.

    After create_root_functions the dynFunc body should contain a
    ForStmt wrapping a CALL to a path func.  Raw VIEW/ADD/ASSEMBLE
    must NOT leak into the dynFunc body.
    """

    def foo(a, b, c):
        pypto.set_vec_tile_shapes(16, 16)
        for i in pypto.loop(2):
            av = pypto.view(a, [16, 16], [i * 16, 0])
            bv = pypto.view(b, [16, 16], [i * 16, 0])
            cv = av + bv
            pypto.assemble(cv, [i * 16, 0], c)

    a = pypto.Tensor([32, 16], pypto.DT_FP32, name="a")
    b = pypto.Tensor([32, 16], pypto.DT_FP32, name="b")
    c = pypto.Tensor([32, 16], pypto.DT_FP32, name="c")
    prog, dyn_func = _run_root_only(foo, a, b, c)

    _assert_func_counts(prog, 1)

    for_stmts = _collect_stmts(dyn_func.body, ir.ForStmt)
    assert len(for_stmts) >= 1, "Expected ForStmt after create_root_functions"

    tensor_ops = _collect_stmts(for_stmts[0].body, ir.TensorOpStmt)
    opcodes = {op.opcode for op in tensor_ops}
    assert "CALL" in opcodes, f"Expected CALL to path func, got {opcodes}"
    assert "VIEW" not in opcodes, "VIEW leaked into dynFunc body"
    assert "ASSEMBLE" not in opcodes, "ASSEMBLE leaked into dynFunc body"


def test_multiple_path_funcs():
    def foo(a, b, c):
        pypto.set_vec_tile_shapes(16, 16)
        tmp = a + b
        for i in pypto.loop(2):
            cv = pypto.view(tmp, [16, 16], [i * 16, 0])
            pypto.assemble(cv, [i * 16, 0], c)

    a = pypto.Tensor([32, 16], pypto.DT_FP32, name="a")
    b = pypto.Tensor([32, 16], pypto.DT_FP32, name="b")
    c = pypto.Tensor([32, 16], pypto.DT_FP32, name="c")
    prog, _ = _run_root_only(foo, a, b, c)

    _assert_func_counts(prog, 2)


def test_if_without_else():
    """An if with no else branch inside a loop — exercises
    TransformStmts IfStmt path where elseBody_ is None.

    Loop body splits into 3 segments: [view] [if-then] [assemble]
    => 3 hidden funcs.
    """

    def foo(x, out):
        pypto.set_vec_tile_shapes(16, 16)
        for i in pypto.loop(4):
            xv = pypto.view(x, [16, 16], [i * 16, 0])
            if i == 0:
                xv = xv + 1.0
            pypto.assemble(xv, [i * 16, 0], out)

    x = pypto.Tensor([64, 16], pypto.DT_FP32, name="x")
    out = pypto.Tensor([64, 16], pypto.DT_FP32, name="out")
    prog, _ = _run_root_only(foo, x, out)

    _assert_func_counts(prog, 3)


def test_intermediate_tensor_cross_path_func():
    """A tensor produced in one path func (pre-loop) and consumed in
    another path func (in-loop).  The producer must list it as outcast
    and the consumer as incast.

    Pre-loop op segment + loop body segment => 2 hidden funcs.
    """

    def foo(a, b, c):
        pypto.set_vec_tile_shapes(16, 16)
        tmp = a + b
        for i in pypto.loop(2):
            tv = pypto.view(tmp, [16, 16], [0, 0])
            pypto.assemble(tv, [i * 16, 0], c)

    a = pypto.Tensor([16, 16], pypto.DT_FP32, name="a")
    b = pypto.Tensor([16, 16], pypto.DT_FP32, name="b")
    c = pypto.Tensor([32, 16], pypto.DT_FP32, name="c")
    prog, _ = _run_root_only(foo, a, b, c)

    _assert_func_counts(prog, 2)


def test_multi_level_nested_loop():
    """Two-level nested loop with tensor ops at inner level only.

    Outer loop body = [inner ForStmt] -> recurses -> inner body is pure
    => 1 hidden func.
    """

    def foo(x, out):
        pypto.set_vec_tile_shapes(16, 16)
        for i in pypto.loop(2):
            for j in pypto.loop(2):
                xv = pypto.view(x, [16, 16], [(i * 2 + j) * 16, 0])
                pypto.assemble(xv, [(i * 2 + j) * 16, 0], out)

    x = pypto.Tensor([64, 16], pypto.DT_FP32, name="x")
    out = pypto.Tensor([64, 16], pypto.DT_FP32, name="out")
    prog, dyn_func = _run_root_only(foo, x, out)

    _assert_func_counts(prog, 1)

    for_stmts = _collect_stmts(dyn_func.body, ir.ForStmt)
    assert len(for_stmts) >= 2, f"Expected >=2 nested ForStmts, got {len(for_stmts)}"


def test_func_param_as_assemble_dst():
    """Assemble directly into a function parameter (not an intermediate
    tensor).  Exercises FindOrCreateSlot(isAssembleOut=true) for a
    paramRawMagics_ entry.

    Single loop body => 1 hidden func.
    """

    def foo(x, out):
        pypto.set_vec_tile_shapes(16, 16)
        for i in pypto.loop(2):
            xv = pypto.view(x, [16, 16], [i * 16, 0])
            pypto.assemble(xv, [i * 16, 0], out)

    x = pypto.Tensor([32, 16], pypto.DT_FP32, name="x")
    out = pypto.Tensor([32, 16], pypto.DT_FP32, name="out")
    prog, _ = _run_root_only(foo, x, out)

    _assert_func_counts(prog, 1)


def test_two_segments_separated_by_if():
    """Loop body: [op, op] -> if {...} -> [op, op].

    SplitIntoTensorOpSegments splits at the IfStmt into 3 segments:
    [view+add] [if-then] [add+assemble] => 3 hidden funcs.
    The IfStmt must be preserved in the dynFunc body.
    """

    def foo(x, out):
        pypto.set_vec_tile_shapes(16, 16)
        for i in pypto.loop(2):
            xv = pypto.view(x, [16, 16], [i * 16, 0])
            a1 = xv + 1.0
            if i == 0:
                a1 = a1 + 2.0
            a2 = a1 + 3.0
            pypto.assemble(a2, [i * 16, 0], out)

    x = pypto.Tensor([32, 16], pypto.DT_FP32, name="x")
    out = pypto.Tensor([32, 16], pypto.DT_FP32, name="out")
    prog, dyn_func = _run_root_only(foo, x, out)

    _assert_func_counts(prog, 3)

    for_stmts = _collect_stmts(dyn_func.body, ir.ForStmt)
    assert len(for_stmts) >= 1
    if_stmts = _collect_stmts(for_stmts[0].body, ir.IfStmt)
    assert len(if_stmts) >= 1, "Expected IfStmt preserved in dynFunc loop body"


def test_assemble_dedup_and_param_exclude():
    """Two assembles into the same intermediate tensor (aux) should be
    deduplicated in constructAssembleSlotList.  An assemble into a
    function parameter (out) should be excluded entirely.

    Pre-loop op segment + loop body segment => 2 hidden funcs.
    """

    def foo(a, out):
        pypto.set_vec_tile_shapes(16, 16)
        aux = pypto.tensor([16, 16], pypto.DT_FP32, name="aux")
        pypto.assemble(pypto.full([16, 16], 0.0, pypto.DT_FP32), [0, 0], aux)
        pypto.assemble(a, [0, 0], aux)
        for i in pypto.loop(2):
            av = pypto.view(aux, [16, 16], [0, 0])
            pypto.assemble(av, [i * 8, 0], out)

    a = pypto.Tensor([16, 16], pypto.DT_FP32, name="a")
    out = pypto.Tensor([16, 16], pypto.DT_FP32, name="out")
    prog, _ = _run_root_only(foo, a, out)

    _assert_func_counts(prog, 2)


def test_sequential_loops():
    """Two sequential loops, each producing its own path func.

    2 loop bodies => 2 hidden funcs.
    """

    def foo(x, out):
        pypto.set_vec_tile_shapes(16, 16)
        for i in pypto.loop(2):
            xv = pypto.view(x, [16, 16], [i * 16, 0])
            pypto.assemble(xv, [i * 16, 0], out)
        for i in pypto.loop(2):
            xv = pypto.view(x, [16, 16], [i * 16, 0])
            pypto.assemble(xv, [i * 16 + 32, 0], out)

    x = pypto.Tensor([64, 16], pypto.DT_FP32, name="x")
    out = pypto.Tensor([64, 16], pypto.DT_FP32, name="out")
    prog, _ = _run_root_only(foo, x, out)

    _assert_func_counts(prog, 2)


def test_pure_tensor_op_single_path_func():
    def foo(a, b, out):
        pypto.set_vec_tile_shapes(16, 16)
        c = a + b
        pypto.assemble(c, [0, 0], out)

    a = pypto.Tensor([16, 16], pypto.DT_FP32, name="a")
    b = pypto.Tensor([16, 16], pypto.DT_FP32, name="b")
    out = pypto.Tensor([16, 16], pypto.DT_FP32, name="out")
    prog, _ = _run_root_only(foo, a, b, out)

    _assert_func_counts(prog, 1)


def test_if_else_both_branches():
    """if/else inside a loop — both branches produce tensor ops.

    Loop body splits into 4 segments:
    [view+view] [if-then] [if-else] [assemble] => 4 hidden funcs.
    """

    def foo(x, y, out):
        pypto.set_vec_tile_shapes(16, 16)
        for i in pypto.loop(2):
            xv = pypto.view(x, [16, 16], [i * 16, 0])
            yv = pypto.view(y, [16, 16], [i * 16, 0])
            if i == 0:
                tmp = xv + yv
            else:
                tmp = xv - yv
            pypto.assemble(tmp, [i * 16, 0], out)

    x = pypto.Tensor([32, 16], pypto.DT_FP32, name="x")
    y = pypto.Tensor([32, 16], pypto.DT_FP32, name="y")
    out = pypto.Tensor([32, 16], pypto.DT_FP32, name="out")
    prog, dyn_func = _run_root_only(foo, x, y, out)

    _assert_func_counts(prog, 4)

    for_stmts = _collect_stmts(dyn_func.body, ir.ForStmt)
    assert len(for_stmts) >= 1
    if_stmts = _collect_stmts(for_stmts[0].body, ir.IfStmt)
    assert len(if_stmts) >= 1, "Expected IfStmt in loop body"


def test_create_tensor():

    def foo(x, y, out):
        pypto.set_vec_tile_shapes(16, 16)
        dim0 = x.shape[0]
        tmp = pypto.tensor([16, 32], pypto.DT_FP32, "max_update")

        for i in pypto.loop(dim0 // 16):

            x_view = pypto.view(x, [16, 32], [i * 16, 0])
            y_view = pypto.view(y, [16, 32], [i * 16, 0])
            a = x_view * y_view
            pypto.assemble(a, [0, 0], tmp)
            b = tmp * 2
            pypto.assemble(b, [i * 16, 0], out)

    x = pypto.Tensor([32, 16], pypto.DT_FP32, name="x")
    y = pypto.Tensor([32, 16], pypto.DT_FP32, name="y")
    out = pypto.Tensor([32, 16], pypto.DT_FP32, name="out")
    b = ir.IRBuilder()
    compiled = pil.compile(foo, x, y, out)
    prog = b.create_program([compiled], "main", ir.Span.unknown())
    prog = ir.Pass.canonicalize()(prog)
    prog = ir.Pass.aggressive_dce()(prog)
    func = prog.functions[compiled.name]

    assert func.body[0].opcode == "TENSOR_ALLOC"

    prog = ir.Pass.create_root_functions()(prog)
    _assert_func_counts(prog, 2)
