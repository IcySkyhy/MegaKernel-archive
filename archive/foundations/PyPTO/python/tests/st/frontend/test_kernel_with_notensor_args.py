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
"""测试 pypto.frontend.jit 对非 pypto.Tensor 参数的支持。"""

import os

import torch

import pypto


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def add_kernel(
    a: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
    b: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
    out: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
    scalar=0,
):
    pypto.set_vec_tile_shapes(16, 16)
    out.move(a + b + scalar)


# =============================================================================
# 通过 kwargs 传入 non-tensor 参数
# =============================================================================
@pypto.options(pass_options={"enable_slice": True})
def test_add_with_kwargs():
    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    torch.npu.set_device(int(device_id))
    a = torch.ones(32, 32, dtype=torch.int32, device=f"npu:{device_id}")
    b = torch.ones(32, 32, dtype=torch.int32, device=f"npu:{device_id}")
    r = torch.zeros(32, 32, dtype=torch.int32, device=f"npu:{device_id}")
    add_kernel(a, b, r, scalar=1)
    assert r.shape == (32, 32)
    assert torch.allclose(r.cpu().float(), torch.ones(32, 32) * 2 + 1)


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def add_npu_with_tiling(
    a: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
    b: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
    out: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
    tiling=None,
    scalar=1,
):
    pypto.set_vec_tile_shapes(tiling, tiling)
    out.move(a + b + scalar)


# # =============================================================================
# # 混合情况
# # =============================================================================
@pypto.options(pass_options={"enable_slice": True})
def test_add_npu_with_tiling():
    device_id = os.environ.get("TILE_FWK_DEVICE_ID", 0)
    torch.npu.set_device(int(device_id))
    a = torch.ones(32, 32, dtype=torch.int32, device=f"npu:{device_id}")
    b = torch.ones(32, 32, dtype=torch.int32, device=f"npu:{device_id}")

    r1 = torch.zeros(32, 32, dtype=torch.int32, device=f"npu:{device_id}")
    add_npu_with_tiling(a, b, r1, 32, scalar=2)
    assert torch.allclose(r1.cpu().float(), torch.ones(32, 32) * 2 + 2)
    r2 = torch.zeros(32, 32, dtype=torch.int32, device=f"npu:{device_id}")
    add_npu_with_tiling(a, b, r2, 16)
    assert torch.allclose(r2.cpu().float(), torch.ones(32, 32) * 2 + 1)
