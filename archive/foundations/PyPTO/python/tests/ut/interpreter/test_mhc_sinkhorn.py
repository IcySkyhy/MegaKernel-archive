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
"""Minimal mhc_sinkhorn verify case (compile + pass_verify + golden, no NPU)."""

import os
import sys

import torch

import pypto

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _ops.mhc_sinkhorn_golden import mhc_sinkhorn_golden  # noqa: E402
from _verify_check import assert_pass_verify_ok, set_verify_goldens  # noqa: E402


@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.SIM, "device_sched_mode": 1},
    pass_options={"vec_nbuffer_setting": {-2: 1, -1: 2}},
)
def mhc_sinkhorn_kernel_sim(
    x: pypto.Tensor([pypto.DYNAMIC, 4, 4], pypto.DT_FP32),
    out: pypto.Tensor([pypto.DYNAMIC, 4, 4], pypto.DT_FP32),
    eps: float,
    num_iters: int,
):
    pypto.experimental.set_operation_options(combine_axis=True)
    t = x.shape[0]
    hc = x.shape[1]
    unroll_list = [1, 8]

    x_flat = pypto.reshape(x, [t, hc * hc], inplace=True)
    s = 8

    for s_idx, unroll_length in pypto.loop_unroll(
        0, (t + s - 1) // s, 1, name="tLoop", idx_name="tIdx", unroll_list=unroll_list
    ):
        tile_t = unroll_length * s
        t_idx = s_idx * s
        t_valid = (t - t_idx).min(tile_t)

        pypto.set_vec_tile_shapes(256, 16)
        comb_flag = pypto.view(x_flat, [tile_t, hc * hc], [t_idx, 0], valid_shape=[t_valid, hc * hc])
        comb_flag = pypto.transpose(comb_flag, 1, 0)
        comb_flag = pypto.reshape(comb_flag, [hc, hc, tile_t], inplace=True)

        pypto.set_vec_tile_shapes(4, 4, 256)
        row_max = pypto.amax(comb_flag, 1, True)
        comb_flag = pypto.exp(comb_flag - row_max)

        row_sum = pypto.sum(comb_flag, 1, True)
        comb_flag = comb_flag / (row_sum + eps)
        comb_flag = comb_flag + eps

        col_sum = pypto.sum(comb_flag, 0, True)
        comb_flag = comb_flag / (col_sum + eps)

        for _ in range(num_iters - 1):
            row_sum = comb_flag.sum(1, keepdim=True)
            comb_flag = comb_flag / (row_sum + eps)
            col_sum = comb_flag.sum(0, keepdim=True)
            comb_flag = comb_flag / (col_sum + eps)

        comb_flag = pypto.reshape(comb_flag, [hc * hc, tile_t])
        pypto.set_vec_tile_shapes(16, 256)
        comb_flag = pypto.transpose(comb_flag, 1, 0)
        pypto.set_vec_tile_shapes(256, 16)

        comb_flag = pypto.reshape(comb_flag, [tile_t, hc, hc])
        out[t_idx:, :, :] = comb_flag


def test_mhc_sinkhorn_minimal():
    """bs=8, N=4; pass_verify vs sinkhorn golden."""
    torch.manual_seed(0)
    x = torch.randn((8, 4, 4), dtype=torch.float32)
    result = torch.empty_like(x)
    y_g = mhc_sinkhorn_golden(x, eps=1e-6, num_iters=20)
    set_verify_goldens([None, y_g])
    mhc_sinkhorn_kernel_sim(x, result, 1e-6, 20)
    assert_pass_verify_ok()
