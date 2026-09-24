#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Minimal interleave_rope verify case (compile + pass_verify + golden, no NPU)."""

import os
import sys

import torch

import pypto

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _ops.interleave_rope_golden import interleave_rope_golden  # noqa: E402
from _verify_check import assert_pass_verify_ok, set_verify_goldens  # noqa: E402

S_TILE_1 = 64
D = 64
HALF = 32


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
def interleave_rope_kernel_n1_bf16_sim(
    x: pypto.Tensor([pypto.DYNAMIC, 1, pypto.DYNAMIC, 64], pypto.DT_BF16),
    cos: pypto.Tensor([pypto.DYNAMIC, 1, pypto.DYNAMIC, 64], pypto.DT_BF16),
    sin: pypto.Tensor([pypto.DYNAMIC, 1, pypto.DYNAMIC, 64], pypto.DT_BF16),
    out: pypto.Tensor([pypto.DYNAMIC, 1, pypto.DYNAMIC, 64], pypto.DT_BF16),
):
    pypto.experimental.set_operation_options(combine_axis=True)
    pypto.set_vec_tile_shapes(1, 1, S_TILE_1, D)
    b_dim = x.shape[0]
    s_dim = x.shape[2]
    s_loops = (s_dim + S_TILE_1 - 1) // S_TILE_1
    for b in pypto.loop(b_dim, name="b_loop"):
        for s_blk in pypto.loop(s_loops, name="s_loop"):
            s_off = s_blk * S_TILE_1
            valid_s = (s_dim - s_off).min(S_TILE_1)
            vshape = [1, 1, valid_s, D]
            x_t = pypto.view(x, [1, 1, S_TILE_1, D], [b, 0, s_off, 0], valid_shape=vshape)
            c_t = pypto.view(cos, [1, 1, S_TILE_1, D], [b, 0, s_off, 0], valid_shape=vshape)
            s_t = pypto.view(sin, [1, 1, S_TILE_1, D], [b, 0, s_off, 0], valid_shape=vshape)
            x_e = pypto.gathermask(x_t, pattern_mode=1)
            x_o = pypto.gathermask(x_t, pattern_mode=2)
            c_e = pypto.gathermask(c_t, pattern_mode=1)
            c_o = pypto.gathermask(c_t, pattern_mode=2)
            s_e = pypto.gathermask(s_t, pattern_mode=1)
            s_o = pypto.gathermask(s_t, pattern_mode=2)
            pypto.set_vec_tile_shapes(1, 1, S_TILE_1, HALF)
            ye = pypto.sub(pypto.mul(x_e, c_e), pypto.mul(x_o, s_e))
            yo = pypto.add(pypto.mul(x_e, s_o), pypto.mul(x_o, c_o))
            pypto.assemble(ye, [b, 0, s_off, 0], out)
            pypto.assemble(yo, [b, 0, s_off, HALF], out)
            pypto.set_vec_tile_shapes(1, 1, S_TILE_1, D)


def test_interleave_rope_minimal():
    """shape [4,1,2,64] bf16; pass_verify vs interleave_rope golden."""
    pypto.set_verify_options(
        enable_pass_verify=True,
        pass_verify_pass_filter=["all"],
        pass_verify_error_tol=[0.008, 0.001]
    )
    shape = (4, 1, 2, 64)
    torch.manual_seed(42)
    x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    cos = torch.randn(shape, dtype=torch.float32).clamp_(-1.0, 1.0).to(torch.bfloat16)
    sin = torch.randn(shape, dtype=torch.float32).clamp_(-1.0, 1.0).to(torch.bfloat16)
    y_out = torch.empty_like(x)
    y_g = interleave_rope_golden(x, cos, sin)
    set_verify_goldens([None, None, None, y_g])
    interleave_rope_kernel_n1_bf16_sim(x, cos, sin, y_out)
    assert_pass_verify_ok()
