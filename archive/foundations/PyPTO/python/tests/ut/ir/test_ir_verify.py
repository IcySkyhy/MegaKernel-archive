# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
import torch

import pypto
from pypto import ir, pil
from pypto._build_online import BuildOnlineCalculatorManager

verify_options = {
    "enable_pass_verify": True,
    "pass_verify_save_tensor": True,
}


def _run_dce(func, golden, *args):
    """Compile a function through the shared new-IR pipeline."""
    in_out_tensors = [*args]

    b = ir.IRBuilder()
    func = pil.compile(func, *args)
    prog = b.create_program([func], "main", ir.Span.unknown())
    dce = ir.Pass.aggressive_dce()
    canonical = ir.Pass.canonicalize()
    merge = ir.Pass.merge_stmts_into_if()
    create_pf = ir.Pass.create_root_functions()
    finalize = ir.Pass.finalize_dynamic_function()
    prog = dce(canonical(prog))
    prog = dce(canonical(prog))
    prog = canonical(merge(prog))
    prog = create_pf(prog)

    pypto.set_verify_golden_data(in_out_tensors=in_out_tensors, goldens=[None, None, golden])
    prog = finalize(prog)
    return prog.functions[func.name]


def generate_f3_golden(a, b):
    z = torch.zeros([32, 32], dtype=torch.float32)
    z[0:16, :] = a[0:16, :] + b[0:16, :]
    z[16:32, :] = a[0:16, :] - b[0:16, :]
    return z


def test_ir_verify():
    def foo(x, y, z):
        pypto.set_vec_tile_shapes(16, 16)
        x_view = pypto.view(x, [16, 32], [0, 0])
        y_view = pypto.view(y, [16, 32], [0, 0])
        for i in pypto.loop(2):
            if i == 0:
                tmp = x_view + y_view
            else:
                tmp = x_view - y_view
            pypto.assemble(tmp, [i * 16, 0], z)

    shape = [32, 32]
    a = torch.rand(shape, dtype=torch.float32)
    b = torch.rand(shape, dtype=torch.float32)
    c = torch.rand(shape, dtype=torch.float32)
    golden = generate_f3_golden(a, b)
    x = pypto.from_torch(a)
    y = pypto.from_torch(b)
    z = pypto.from_torch(c)

    original_options = pypto.get_verify_options()
    pypto.set_verify_options(**verify_options)
    try:
        BuildOnlineCalculatorManager().build_and_load_calculator()

        _run_dce(foo, golden, x, y, z)
    finally:
        pypto.set_verify_options(**original_options)
