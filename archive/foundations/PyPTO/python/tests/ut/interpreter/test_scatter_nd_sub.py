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
"""Minimal scatter_nd_sub verify case (compile + pass_verify + golden, no NPU)."""

import os
import sys

import torch

import pypto

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _verify_check import assert_pass_verify_ok, set_verify_goldens  # noqa: E402


@pypto.frontend.jit(
    runtime_options={
        "run_mode": pypto.RunMode.SIM,
        "stitch_function_max_num": 128,
    }
)
def scatter_nd_sub_kernel(
    target: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    indices: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_INT32),
    updates: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32),
):
    for bs_idx, tile_batch in pypto.loop_unroll(
        0, indices.shape[0], 1, name="LOOP_SCATTER_ND_SUB_L0",
        idx_name="bs_idx", unroll_list=[2048, 1024, 512, 256, 1],
    ):
        b_offset = bs_idx
        b_offset_end = bs_idx + tile_batch
        indices_temp = indices[b_offset:b_offset_end, ...]
        pypto.set_vec_tile_shapes(32, 16)
        indices_temp_new = pypto.reshape(indices_temp, [indices_temp.shape[0]])
        indices_tuple = (indices_temp_new,)
        pypto.set_vec_tile_shapes(32, 16)
        neg_updates = pypto.mul(updates[b_offset:b_offset_end], -1)
        pypto.index_put_(target, indices_tuple, neg_updates, True)


def _scatter_nd_sub_golden(target, indices, updates):
    out = target.clone()
    idx = indices.reshape(-1).long()
    out.index_add_(0, idx, -updates)
    return out


def test_scatter_nd_sub_minimal():
    """target[10,16], indices[1024,1], updates[1024,16]; pass_verify vs golden."""
    pypto.set_verify_options(
        enable_pass_verify=True,
        pass_verify_pass_filter=["CodegenPreproc"]
    )
    torch.manual_seed(0)
    target = torch.rand((10, 16), dtype=torch.float32) * 10
    indices = torch.randint(0, 10, (1024, 1), dtype=torch.int32)
    updates = torch.rand((1024, 16), dtype=torch.float32) * 2
    golden = _scatter_nd_sub_golden(target, indices, updates)
    # In-place write to arg0 → golden slot 0
    set_verify_goldens([golden, None, None])
    scatter_nd_sub_kernel(target, indices, updates)
    assert_pass_verify_ok()
