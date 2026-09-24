# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Tests for dynValidShape clone in SubstituteStmt (IR snapshot golden).

When a VIEW with a dynamic toValidShape (containing a yield-var SymbolicScalar) is
sunk into both if/else branches by merge_stmts, the output LogicalTensor must be
cloned per-branch so that each branch keeps its own dynValidShape. The snapshot
captures the merged if-tree in golden .pypto files; regenerate them after an
intentional change by printing str(func) (the tests print the merged IR with -s).
"""

from pathlib import Path

import pypto

from ..test_common import check_snapshot, run_merge_pass

_GOLDEN_DIR = Path(__file__).parent

IR_THEN_ELSE = (_GOLDEN_DIR / "test_dyn_valid_shape_clone_then_else.pypto").read_text()
IR_NO_CROSS = (_GOLDEN_DIR / "test_dyn_valid_shape_no_cross_branch_contamination.pypto").read_text()


def test_view_dyn_valid_shape_clone_then_else():
    """Each branch yields a SymbolicScalar as vs (min(n,m) / min(n,16)): after merge_stmts
    sinks the VIEW into both branches, the else-branch keeps its own cloned VIEW var
    (distinct from the then-branch's), with its own dynamic dynValidShape."""

    def foo_then_else(x, y, z):
        pypto.set_vec_tile_shapes(16, 16)
        n = pypto.symbolic_scalar("n")
        m = pypto.symbolic_scalar("m")
        for i in pypto.loop(2):
            xv = pypto.view(x, [16, 16], [0, 0])
            yv = pypto.view(y, [16, 16], [0, 0])
            if i == 0:
                t = xv + yv
                vs = pypto.min(n, m)
            else:
                t = xv - yv
                vs = pypto.min(n, 16)
            out_view = pypto.view(t, [16, 16], [0, 0], valid_shape=[vs, 16])
            pypto.assemble(out_view, [0, 0], z)

    x = pypto.Tensor([32, 32], pypto.DT_FP32, 'x')
    y = pypto.Tensor([32, 32], pypto.DT_FP32, 'y')
    z = pypto.Tensor([32, 32], pypto.DT_FP32, 'z')
    func = run_merge_pass(foo_then_else, x, y, z)
    print("\nmerged IR:\n%s" % func)
    check_snapshot(func, IR_THEN_ELSE)


def test_view_dyn_valid_shape_no_cross_branch_contamination():
    """After merge_stmts, the else-branch VIEW must keep its own dynValidShape
    (min(n,16)), independent from the then-branch's (min(n,m)): the snapshot shows
    distinct per-branch VIEW vars instead of a shared poisoned tensor."""

    def foo_no_cross(x, z):
        pypto.set_vec_tile_shapes(16, 16)
        n = pypto.symbolic_scalar("n")
        m = pypto.symbolic_scalar("m")
        for i in pypto.loop(2):
            xv = pypto.view(x, [16, 16], [0, 0])
            if i == 0:
                vs = pypto.min(n, m)
            else:
                vs = pypto.min(n, 16)
            out_view = pypto.view(xv, [16, 16], [0, 0], valid_shape=[vs, 16])
            pypto.assemble(out_view, [0, 0], z)

    x = pypto.Tensor([32, 32], pypto.DT_FP32, 'x')
    z = pypto.Tensor([32, 32], pypto.DT_FP32, 'z')
    func = run_merge_pass(foo_no_cross, x, z)
    print("\nmerged IR:\n%s" % func)
    check_snapshot(func, IR_NO_CROSS)
