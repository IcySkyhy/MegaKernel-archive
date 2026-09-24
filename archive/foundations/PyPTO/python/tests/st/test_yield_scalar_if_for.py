#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Test yield scalar in controlFlow-level constructs.

Covers if/else inside for loop body yielding both constant and symbolic
expression scalars, used as inner loop bounds.
"""

import os

from numpy.testing import assert_allclose
import torch

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))


@pypto.frontend.jit()
def yield_const_and_symbolic_kernel(
    x: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32),
    out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32),
):
    """if/else yields constant (step) and symbolic expression (stop), both as loop bounds."""
    m = x.shape[0]
    n = 128

    pypto.set_vec_tile_shapes(1, 128)

    for b_idx in pypto.loop(0, m, 32, name="OUTER_LOOP", idx_name="b_idx"):
        if pypto.cond(b_idx < m // 2):
            step_val = 1
            stop_val = b_idx + 32
        else:
            step_val = 2
            stop_val = b_idx + 16

        for s_idx in pypto.loop(b_idx, stop_val, step_val, name="INNER_LOOP", idx_name="s_idx"):
            x_view = pypto.view(x, [1, n], [s_idx, 0])
            result = pypto.add(x_view, pypto.full([1, n], 1.0, pypto.DT_FP32))
            pypto.assemble(result, [s_idx, 0], out)


def test_yield_const_and_symbolic():
    """Verify VALUE_step_val = 1; and VALUE_stop_val = expr; in controlFlow cpp."""
    torch.npu.set_device(ST_DEVICE_ID)

    m, n = 64, 128
    dtype = torch.float32

    x = torch.randn(m, n, dtype=dtype).npu()
    out = torch.zeros(m, n, dtype=dtype).npu()

    yield_const_and_symbolic_kernel(x, out)

    torch.npu.synchronize()

    golden = torch.zeros(m, n, dtype=dtype)
    for b_idx in range(0, m, 32):
        if b_idx < m // 2:
            step_val, stop_val = 1, b_idx + 32
        else:
            step_val, stop_val = 2, b_idx + 16
        for s_idx in range(b_idx, stop_val, step_val):
            golden[s_idx, :] = x[s_idx, :] + 1.0
    assert_allclose(out.cpu().numpy(), golden.cpu().numpy(), atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    test_yield_const_and_symbolic()
