#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Dynamic cell-match ST test (multi-L tmp->d_emb kernel).

Runs the same JIT kernel with sequence lengths 64, 2000, and 125 sequentially in one
process to exercise dynamic cell-match metadata pool reuse and resize across launches.

Do NOT add ``from __future__ import annotations`` — it breaks @jit parameter parsing.
"""

import os

from numpy.testing import assert_allclose
import torch

import pypto

B_STATIC = 1
H_STATIC = 1
D_STATIC = 16
UNROLL = [1]

# Run back-to-back in one process (order matters for dynCm pool grow/shrink).
L_STATIC_VALUES = [64, 2000, 125]

RTOL = 1e-3
ATOL = 1e-3


def _golden_d_emb_only(dy: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (dy[:, :, 0, :].reshape(-1, dy.shape[-1]) @ weight[0].T).reshape(dy.shape[0], dy.shape[1], dy.shape[-1])


def _linear_dx_only(dy, weight):
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    return pypto.matmul(dy, weight, pypto.DT_FP32, b_trans=True)


@pypto.frontend.jit()
def k_tmp_to_d_emb(
    dy: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, H_STATIC, D_STATIC], pypto.DT_FP32),
    weight: pypto.Tensor([H_STATIC, D_STATIC, D_STATIC], pypto.DT_FP32),
    d_emb: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, D_STATIC], pypto.DT_FP32),
):
    b, l, _, _ = dy.shape  # noqa: E741
    h_idx = 0
    tmp = pypto.tensor([b, l, H_STATIC, D_STATIC], d_emb.dtype, "tmp")
    for l_idx, tile in pypto.loop_unroll(0, l, 1, unroll_list=UNROLL):
        pypto.set_vec_tile_shapes(1, 64, 1, 256)
        dy_v = dy[0, l_idx:l_idx + tile, h_idx]
        dx = _linear_dx_only(dy_v, weight[h_idx])
        pypto.set_vec_tile_shapes(1, 64, 1, 512)
        tmp[0, l_idx:l_idx + tile, h_idx] = dx
    for b_idx in pypto.loop(0, b, 1, name="b_loop_1"):
        for l_idx, tile in pypto.loop_unroll(0, l, 1, name="l_loop_2", unroll_list=UNROLL):
            pypto.set_vec_tile_shapes(1, 64, 1, 512)
            d_emb[b_idx, l_idx:l_idx + tile] = tmp[b_idx, l_idx:l_idx + tile, h_idx]


def _run_case(l_static: int, device: str, seed: int = 44) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed + l_static)
    dy = torch.randn(B_STATIC, l_static, H_STATIC, D_STATIC, dtype=torch.float32, device=device)
    # x = torch.randn(B_STATIC, l_static, D_STATIC, dtype=torch.float32, device=device)
    weight = torch.randn(H_STATIC, D_STATIC, D_STATIC, dtype=torch.float32, device=device)

    ref_emb = _golden_d_emb_only(dy, weight)
    d_emb = torch.zeros(B_STATIC, l_static, D_STATIC, dtype=torch.float32, device=device)
    k_tmp_to_d_emb(dy, weight, d_emb)
    torch.npu.synchronize()
    return d_emb, ref_emb


@pypto.options(pass_options={"enable_slice": True})
def test_cellmatch_case():
    """Multiple L values in one process: dynCm pool alloc, resize, and reuse."""
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", "0"))
    torch.npu.set_device(device_id)
    device = f"npu:{device_id}"

    for l_static in L_STATIC_VALUES:
        d_emb, ref_emb = _run_case(l_static, device)
        assert_allclose(
            d_emb.cpu().numpy(),
            ref_emb.cpu().numpy(),
            rtol=RTOL,
            atol=ATOL,
            err_msg=f"L_STATIC={l_static} d_emb mismatch",
        )
