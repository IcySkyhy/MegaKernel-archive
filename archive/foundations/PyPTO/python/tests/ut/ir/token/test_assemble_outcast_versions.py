# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Root-function outcast tests for Assemble LogicalTensor versions."""

from pathlib import Path

import pypto
from pypto import ir, pil

from ..test_common import _ssa_verify, check_snapshot

_GOLDEN_DIR = Path(__file__).parent

IR = (_GOLDEN_DIR / "test_assemble_outcast_versions.pypto").read_text()


def _run_root_builder(func, *args):
    """Compile a kernel and run canonicalize + dce + create_root_functions, returning the program
    so callers can inspect hiddenfuncs and outcast versions."""
    compiled = pil.compile(func, *args, create_new_logical_tensor=True)
    builder = ir.IRBuilder()
    program = builder.create_program([compiled], "main", ir.Span.unknown())
    verifier = ir.IRVerifier.create_default()
    _ssa_verify(verifier, program, "original")
    program = ir.Pass.aggressive_dce()(ir.Pass.canonicalize()(program))
    _ssa_verify(verifier, program, "dce")
    program = ir.Pass.create_root_functions()(program)
    _ssa_verify(verifier, program, "root_functions")
    return program


def test_assemble_outcast_versions_share_raw_tensor():
    """MakeOutcasts preserves distinct Assemble versions on one RawTensor."""

    def foo(a, aux, out):
        pypto.set_vec_tile_shapes(16, 16)
        pypto.assemble(pypto.full([16, 16], 1.0, pypto.DT_FP32), [0, 0], aux)
        pypto.assemble(pypto.full([16, 16], 2.0, pypto.DT_FP32), [16, 0], aux)
        out[:] = a + aux

    a = pypto.Tensor([32, 16], pypto.DT_FP32, name="a")
    aux = pypto.Tensor([32, 16], pypto.DT_FP32, name="aux")
    out = pypto.Tensor([32, 16], pypto.DT_FP32, name="out")
    program = _run_root_builder(foo, a, aux, out)

    check_snapshot(program, IR)
