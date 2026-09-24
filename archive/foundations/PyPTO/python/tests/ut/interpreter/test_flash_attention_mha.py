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
"""Minimal flash_attention_mha verify case (Ascend950 compile + pass_verify + golden, no NPU)."""

import math
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _ops.flash_attention_mha_impl import (  # noqa: E402
    FlashAttentionTileShapeConfig,
    flash_attention_varlen_forward_kernel,
)
from _verify_check import assert_pass_verify_ok, set_verify_goldens  # noqa: E402

_TILE_950 = FlashAttentionTileShapeConfig(
    q_tile=128,
    k_tile=128,
    c1_cube_tile=[[128, 128], [128, 128], [128, 128]],
    v1_tile=[128, 128],
    c2_cube_tile=[[128, 128], [128, 128], [128, 128]],
    v2_tile=[128, 128],
)


def _attention_forward_golden_noflash(q, k, v, scale):
    scores = torch.matmul(q.to(torch.float32), k.transpose(1, 0).to(torch.float32)) * scale
    m = scores.amax(dim=-1, keepdim=True)
    p_unnorm = torch.exp(scores - m)
    softmax_l = p_unnorm.sum(dim=-1, keepdim=True)
    p_norm = p_unnorm / softmax_l
    o = torch.matmul(p_norm.to(torch.bfloat16).to(torch.float32), v.to(torch.float32)).to(torch.bfloat16)
    return o, m, softmax_l


def _create_inputs(batch_size, s1_size, s2_size, num_heads, head_dim):
    q_seqlens = [s1_size] * batch_size
    kv_seqlens = [s2_size] * batch_size
    total_q = sum(q_seqlens)
    total_kv = sum(kv_seqlens)
    torch.manual_seed(42)
    q = torch.randn(total_q, num_heads, head_dim, dtype=torch.bfloat16) + 0.5
    k = torch.randn(total_kv, num_heads, head_dim, dtype=torch.bfloat16) + 0.5
    v = torch.randn(total_kv, num_heads, head_dim, dtype=torch.bfloat16) + 0.5
    cu_seqlens_q = torch.tensor([0] + list(np.cumsum(q_seqlens)), dtype=torch.int32)
    cu_seqlens_k = torch.tensor([0] + list(np.cumsum(kv_seqlens)), dtype=torch.int32)
    return q, k, v, cu_seqlens_q, cu_seqlens_k


def _compute_goldens(q, k, v, batch_size, num_heads, dim, s1_size, s2_size):
    scale = 1.0 / math.sqrt(dim)
    out_g = torch.empty(batch_size * s1_size, num_heads * dim, dtype=torch.bfloat16)
    l_g = torch.empty(batch_size * s1_size, num_heads, dtype=torch.float32)
    m_g = torch.empty(batch_size * s1_size, num_heads, dtype=torch.float32)
    q_off = k_off = 0
    for _ in range(batch_size):
        for h in range(num_heads):
            h_off = h * dim
            o, m, softmax_l = _attention_forward_golden_noflash(
                q[q_off:q_off + s1_size, h, :],
                k[k_off:k_off + s2_size, h, :],
                v[k_off:k_off + s2_size, h, :],
                scale,
            )
            out_g[q_off:q_off + s1_size, h_off:h_off + dim] = o
            m_g[q_off:q_off + s1_size, h:h + 1] = m
            l_g[q_off:q_off + s1_size, h:h + 1] = softmax_l
        q_off += s1_size
        k_off += s2_size
    return out_g, l_g, m_g


@pytest.mark.soc("950")
def test_flash_attention_mha_minimal():
    """b=8,h=16,s=32,dim=32; pass_verify vs attention golden."""
    batch_size, num_heads, s1_size, s2_size, dim = 8, 16, 32, 32, 32
    hidden_dim = num_heads * dim
    q, k, v, cu_q, cu_k = _create_inputs(batch_size, s1_size, s2_size, num_heads, dim)
    total_q = batch_size * s1_size
    out = torch.empty(total_q, hidden_dim, dtype=torch.bfloat16)
    l_out = torch.empty(total_q, num_heads, dtype=torch.float32)
    m_out = torch.empty(total_q, num_heads, dtype=torch.float32)
    out_g, l_g, m_g = _compute_goldens(q, k, v, batch_size, num_heads, dim, s1_size, s2_size)
    set_verify_goldens([None, None, None, out_g, l_g, m_g, None, None])
    flash_attention_varlen_forward_kernel(q, k, v, out, l_out, m_out, cu_q, cu_k, _TILE_950)
    assert_pass_verify_ok()
