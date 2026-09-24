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
"""Minimal sum_lstm verify case (compile + pass_verify + golden, no NPU)."""

import os
import sys
from typing import Optional, Tuple

import torch

import pypto

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _ops.sum_lstm import LstmConfig, LstmTileConfig, sum_lstm_compute  # noqa: E402
from _verify_check import assert_pass_verify_ok, set_verify_goldens  # noqa: E402

BATCH_SIZE = 32
D_GATE = 4096
D_GATE_4 = 16384


def _rms_norm_golden(x: torch.Tensor, eps: float) -> torch.Tensor:
    x = x.to(torch.float32)
    mean_square = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(mean_square + eps)


def _gelu_approx_sigmoid_golden(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(1.702 * x)


def _sum_lstm_golden(
    states_4d: torch.Tensor,
    z4_4d: torch.Tensor,
    prev_cell: torch.Tensor,
    alpha: float,
    eps_cell: float,
    eps_state: float,
    w_cell: Optional[torch.Tensor] = None,
    b_cell: Optional[torch.Tensor] = None,
    w_state: Optional[torch.Tensor] = None,
    b_state: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    fused = states_4d + alpha * z4_4d
    chunk_size = fused.shape[-1] // 4
    pre_f, pre_i, pre_o, pre_c = torch.split(fused, chunk_size, dim=-1)
    f_gate = torch.sigmoid(pre_f)
    i_gate = torch.sigmoid(pre_i)
    c_cand_norm = _rms_norm_golden(pre_c, eps_cell)
    if w_cell is not None:
        c_cand_norm = c_cand_norm * w_cell
    if b_cell is not None:
        c_cand_norm = c_cand_norm + b_cell
    c_act = _gelu_approx_sigmoid_golden(c_cand_norm)
    c_new = prev_cell * f_gate + c_act * i_gate
    h_temp = _rms_norm_golden(c_new, eps_state)
    if w_state is not None:
        h_temp = h_temp * w_state
    if b_state is not None:
        h_temp = h_temp + b_state
    h_act = _gelu_approx_sigmoid_golden(h_temp)
    o_gate = torch.sigmoid(pre_o)
    h_new = h_act * o_gate
    return h_new.to(states_4d.dtype), c_new.to(states_4d.dtype)


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.SIM, "device_sched_mode": 1})
def sum_lstm_kernel_sim(
    states_4d: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP16),
    z4_4d: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP16),
    prev_cell: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP16),
    w_cell: pypto.Tensor([...], pypto.DT_FP16),
    b_cell: pypto.Tensor([...], pypto.DT_FP16),
    w_state: pypto.Tensor([...], pypto.DT_FP16),
    b_state: pypto.Tensor([...], pypto.DT_FP16),
    h_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP16),
    c_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP16),
    config: LstmConfig,
):
    tile_cfg = LstmTileConfig()
    sum_lstm_compute(
        states_4d, z4_4d, prev_cell,
        w_cell, b_cell, w_state, b_state,
        config, tile_cfg,
        h_out, c_out,
    )


def test_sum_lstm_minimal():
    """BATCH=32, D_GATE=4096 fp16; pass_verify vs sum_lstm golden."""
    torch.manual_seed(0)
    states = torch.randn(BATCH_SIZE, D_GATE_4, dtype=torch.float16)
    z4 = torch.randn(BATCH_SIZE, D_GATE_4, dtype=torch.float16)
    prev = torch.randn(BATCH_SIZE, D_GATE, dtype=torch.float16)
    w_c = torch.randn(D_GATE, dtype=torch.float16)
    b_c = torch.randn(D_GATE, dtype=torch.float16)
    w_s = torch.randn(D_GATE, dtype=torch.float16)
    b_s = torch.randn(D_GATE, dtype=torch.float16)
    h_out = torch.zeros(BATCH_SIZE, D_GATE, dtype=torch.float16)
    c_out = torch.zeros(BATCH_SIZE, D_GATE, dtype=torch.float16)
    cfg = LstmConfig(alpha=0.1, eps_cell=1e-6, eps_state=1e-6)
    h_g, c_g = _sum_lstm_golden(
        states, z4, prev, cfg.alpha, cfg.eps_cell, cfg.eps_state, w_c, b_c, w_s, b_s
    )
    set_verify_goldens([None] * 7 + [h_g, c_g])
    sum_lstm_kernel_sim(states, z4, prev, w_c, b_c, w_s, b_s, h_out, c_out, cfg)
    assert_pass_verify_ok()
