# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import pypto
from pypto import ir, pil

from .test_common import _ssa_verify


def test_canonicalize_keeps_carry_used_by_nested_loop_init():
    def kernel(src, out):
        pypto.set_vec_tile_shapes(16, 16)
        out_alias = out
        for _ in pypto.loop(2):
            for _ in pypto.loop(2):
                pypto.assemble(pypto.view(src, [16, 16], [0, 0]), [0, 0], out_alias)

    src = pypto.Tensor([16, 16], pypto.DT_FP32, name="src")
    out = pypto.Tensor([16, 16], pypto.DT_FP32, name="out")
    func = pil.compile(kernel, src, out, create_new_logical_tensor=True)
    builder = ir.IRBuilder()
    program = builder.create_program([func], "main", ir.Span.unknown())
    verifier = ir.IRVerifier.create_default()

    _ssa_verify(verifier, program, "original")
    program = ir.Pass.canonicalize()(program)
    _ssa_verify(verifier, program, "canonicalize")

    main_func = program.functions[func.name]
    outer_loop = next(stmt for stmt in main_func.body.stmts if isinstance(stmt, ir.ForStmt))
    assert "out_alias" in [arg.iterVar.name for arg in outer_loop.iter_args]
