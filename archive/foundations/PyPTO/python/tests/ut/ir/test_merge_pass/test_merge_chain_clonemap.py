# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from pathlib import Path

import pypto
from pypto import ir, pil

from ..test_common import check_snapshot

_GOLDEN_DIR = Path(__file__).parent

IR = (_GOLDEN_DIR / "test_merge_chain_clonemap.pypto").read_text()


def test_merge_chain_clonemap():
    """Chained AppendIntoIfStmt produces a cloneMap chain (A->B->C->D)."""

    def foo(x, y, z):
        n = pypto.symbolic_scalar("n")
        m = pypto.symbolic_scalar("m")
        pypto.set_vec_tile_shapes(16, 16)
        for i in pypto.loop(2):
            if n > 0:
                t0 = x + y
            else:
                t0 = x - y
            if m > 0:
                t1 = t0 + y
            else:
                t1 = t0 - y
            s0 = t1 + x
            pypto.assemble(s0, [0, 0], z)

    x = pypto.Tensor([32, 32], pypto.DT_FP32, 'x')
    y = pypto.Tensor([32, 32], pypto.DT_FP32, 'y')
    z = pypto.Tensor([32, 32], pypto.DT_FP32, 'z')

    b = ir.IRBuilder()
    func = pil.compile(foo, x, y, z, create_new_logical_tensor=True)
    prog = b.create_program([func], "main", ir.Span.unknown())
    merge = ir.Pass.merge_stmts_into_if()
    prog = merge(prog)

    func = prog.functions[func.name]

    # check_snapshot verifies the full IR string against golden.
    # Key difference from pre-fix IR: continue references $6_1@7 (outer returnVar)
    # instead of $6_0@7 (intermediate inner returnVar, not defined in any outer yield).
    check_snapshot(func, IR)
