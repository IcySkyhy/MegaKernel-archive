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
"""Minimal apply_adam_w_v2 verify case (compile + pass_verify + golden, no NPU)."""

import os
import sys

import torch

import pypto

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _ops.apply_adam_w_v2_golden import apply_adam_w_v2_golden  # noqa: E402
from _verify_check import assert_pass_verify_ok, set_verify_goldens  # noqa: E402

_PARAMS = dict(beta1=0.9, beta2=0.999, lr=1e-3, weight_decay=0.01, eps=1e-8, step=1)
_TILE_CONFIG = [32, 32, 512]


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
def apply_adam_w_v2_kernel_bf16_sim(
    weight: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_BF16),
    grad: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_BF16),
    m: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),
    v: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),
    weight_out: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_BF16),
    m_out: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),
    v_out: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),
    beta1: float,
    one_m_b1: float,
    beta2: float,
    one_m_b2: float,
    bc1: float,
    bc2: float,
    lr: float,
    weight_decay: float,
    eps: float,
    tile_config: list,
):
    m_dim = weight.shape[0]
    n_dim = weight.shape[1]
    m_tile = tile_config[0]
    pypto.set_vec_tile_shapes(tile_config[1], tile_config[2])
    m_loops = pypto.ceildiv(m_dim, m_tile)
    n_tile = 1024
    n_loops = pypto.ceildiv(n_dim, n_tile)

    for m_idx in pypto.loop(m_loops, name="adamw_m_loop_bf16", idx_name="m_idx", unroll_list=[1]):
        m_offset = m_idx * m_tile
        valid_m = (m_dim - m_offset).min(m_tile)
        for n_idx in pypto.loop(n_loops, name="adamw_n_loop_bf16", idx_name="n_idx", unroll_list=[1]):
            n_offset = n_idx * n_tile
            valid_n = (n_dim - n_offset).min(n_tile)
            valid_shape = [valid_m, valid_n]

            w_tile = pypto.view(weight, [m_tile, n_tile], [m_offset, n_offset], valid_shape=valid_shape)
            g_tile = pypto.view(grad, [m_tile, n_tile], [m_offset, n_offset], valid_shape=valid_shape)
            m_state = pypto.view(m, [m_tile, n_tile], [m_offset, n_offset], valid_shape=valid_shape)
            v_state = pypto.view(v, [m_tile, n_tile], [m_offset, n_offset], valid_shape=valid_shape)

            w_f32 = pypto.cast(w_tile, pypto.DT_FP32)
            g_f32 = pypto.cast(g_tile, pypto.DT_FP32)
            m_new = pypto.add(pypto.mul(m_state, beta1), pypto.mul(g_f32, one_m_b1))
            grad_sq = pypto.mul(g_f32, g_f32)
            v_new = pypto.add(pypto.mul(v_state, beta2), pypto.mul(grad_sq, one_m_b2))
            m_hat = pypto.div(m_new, bc1, precision_type=pypto.PrecisionType.INTRINSIC)
            v_hat = pypto.div(v_new, bc2, precision_type=pypto.PrecisionType.INTRINSIC)
            denom = pypto.add(pypto.sqrt(v_hat), eps)
            update = pypto.add(
                pypto.div(m_hat, denom, precision_type=pypto.PrecisionType.INTRINSIC),
                pypto.mul(w_f32, weight_decay),
            )
            w_new_f32 = pypto.sub(w_f32, pypto.mul(update, lr))
            w_new_bf16 = pypto.cast(w_new_f32, pypto.DT_BF16)

            pypto.assemble(w_new_bf16, [m_offset, n_offset], weight_out)
            pypto.assemble(m_new, [m_offset, n_offset], m_out)
            pypto.assemble(v_new, [m_offset, n_offset], v_out)


def test_apply_adam_w_v2_minimal():
    """bf16 shape [5, 17]; pass_verify vs AdamW golden."""
    shape = (5, 17)
    torch.manual_seed(42)
    weight = (torch.randn(shape, dtype=torch.float32) * 0.02).to(torch.bfloat16)
    grad = (torch.randn(shape, dtype=torch.float32) * 0.01).to(torch.bfloat16)
    m = torch.randn(shape, dtype=torch.float32) * 1e-3
    v = torch.randn(shape, dtype=torch.float32).abs() * 1e-6

    beta1 = _PARAMS["beta1"]
    beta2 = _PARAMS["beta2"]
    lr = _PARAMS["lr"]
    weight_decay = _PARAMS["weight_decay"]
    eps = _PARAMS["eps"]
    step = _PARAMS["step"]
    bc1 = 1.0 - (beta1 ** step)
    bc2 = 1.0 - (beta2 ** step)

    w_g, m_g, v_g = apply_adam_w_v2_golden(
        weight, grad, m, v, beta1, beta2, lr, weight_decay, eps, step
    )
    weight_out = torch.empty_like(weight)
    m_out = torch.empty_like(m)
    v_out = torch.empty_like(v)
    set_verify_goldens([None, None, None, None, w_g, m_g, v_g])
    apply_adam_w_v2_kernel_bf16_sim(
        weight, grad, m, v, weight_out, m_out, v_out,
        float(beta1), float(1.0 - beta1), float(beta2), float(1.0 - beta2),
        float(bc1), float(bc2), float(lr), float(weight_decay), float(eps),
        list(_TILE_CONFIG),
    )
    assert_pass_verify_ok()
