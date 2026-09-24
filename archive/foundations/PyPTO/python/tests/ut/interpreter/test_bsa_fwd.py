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
"""Minimal BSA forward verify case (Ascend950 compile + pass_verify + golden, no NPU)."""

import math
import os
import sys

import pytest
import torch

import pypto  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _ops.bsa import bsa_fwd_impl as _bsa_impl  # noqa: E402
from _ops.bsa.bsa_common import DEFAULT_CONFIG  # noqa: E402
from _ops.bsa.bsa_fwd_golden import BSAForwardInputs, bsa_forward_golden  # noqa: E402
from _ops.bsa.bsa_fwd_impl import BSAForwardCallInputs, block_sparse_attention_forward  # noqa: E402
from _verify_check import assert_pass_verify_ok, set_verify_goldens  # noqa: E402

_CFG = DEFAULT_CONFIG


def _generate_block_sparse_mask(batch, head_num_q, num_q_blocks, num_kv_blocks, sparsity, seed):
    torch.manual_seed(seed)
    mask = torch.rand(batch, head_num_q, num_q_blocks, num_kv_blocks) < sparsity
    empty = ~mask.any(dim=-1)
    if empty.any():
        forced = torch.randint(0, num_kv_blocks, empty.shape)
        b_idx, h_idx, u_idx = torch.where(empty)
        mask[b_idx, h_idx, u_idx, forced[b_idx, h_idx, u_idx]] = True
    return mask


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_bsa_fwd_minimal(monkeypatch):
    """S256 Sparse50; pass_verify vs BSA O golden (l/m skipped — host LSE form differs)."""
    b, hq, hkv, sq, skv, sparsity = 1, 4, 2, 256, 256, 0.5
    torch.manual_seed(42)
    num_qb = math.ceil(sq / _CFG.block_shape_x)
    num_kb = math.ceil(skv / _CFG.block_shape_y)
    q = torch.empty(b, hq, sq, _CFG.head_dim, dtype=_CFG.torch_dtype).uniform_(-1, 1)
    k = torch.empty(b, hkv, skv, _CFG.head_dim, dtype=_CFG.torch_dtype).uniform_(-1, 1)
    v = torch.empty(b, hkv, skv, _CFG.head_dim, dtype=_CFG.torch_dtype).uniform_(-1, 1)
    mask = _generate_block_sparse_mask(b, hq, num_qb, num_kb, sparsity, seed=42)
    asq = torch.full([b], sq, dtype=torch.int64)
    askv = torch.full([b], skv, dtype=torch.int64)

    golden = bsa_forward_golden(
        BSAForwardInputs(
            query=q,
            key=k,
            value=v,
            block_sparse_mask=mask,
            block_shape_x=_CFG.block_shape_x,
            block_shape_y=_CFG.block_shape_y,
            actual_seq_lengths=asq,
            actual_seq_lengths_kv=askv,
            scale_value=_CFG.softmax_scale,
            cfg=_CFG,
        )
    )
    bx = _CFG.block_shape_x
    sq_pad = math.ceil(sq / bx) * bx
    d = _CFG.head_dim
    o_pad = torch.zeros(b * hq, sq_pad, d, dtype=_CFG.torch_dtype)
    o_pad[:, :sq, :] = golden.o.reshape(b * hq, sq, d)

    real_dispatch = _bsa_impl._dispatch_fwd_kernel

    def _dispatch_with_golden(call_inputs, prepared, sparse_kv_result):
        n_hint = 7
        n_in = n_hint + 5
        set_verify_goldens([None] * n_in + [o_pad.contiguous(), None, None])
        return real_dispatch(call_inputs, prepared, sparse_kv_result)

    monkeypatch.setattr(_bsa_impl, "_dispatch_fwd_kernel", _dispatch_with_golden)

    block_sparse_attention_forward(
        BSAForwardCallInputs(
            query=q,
            key=k,
            value=v,
            block_sparse_mask=mask,
            actual_seq_lengths=asq,
            actual_seq_lengths_kv=askv,
            block_shape=None,
            cfg=_CFG,
        )
    )
    assert_pass_verify_ok()
