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

import pytest
import torch

import pypto


# =============================================================================
# @jit @function 场景验证
# =============================================================================
@pytest.mark.skip("Failed for sync")
@pypto.options(pass_options={"enable_slice": True})
def test_add_with_kwargs_run():
    @pypto.frontend.function
    def sub_add_kernel(
        a: pypto.Tensor([], pypto.DT_INT32),
        b: pypto.Tensor(dtype=pypto.DT_INT32),
        c: pypto.Tensor([...], pypto.DT_INT32),
        d: pypto.Tensor([pypto.STATIC, 32], pypto.DT_INT32),
        res: pypto.Tensor([32, 32], pypto.DT_INT32),
        scalar=0,
    ):
        res.move(a + b + c + d + scalar)

    @pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
    def add_kernel(
        a: pypto.Tensor(dtype=pypto.DT_INT32),
        b: pypto.Tensor([], pypto.DT_INT32),
        c: pypto.Tensor([...], pypto.DT_INT32),
        d: pypto.Tensor([pypto.STATIC, 32], pypto.DT_INT32),
        res: pypto.Tensor([32, 32], pypto.DT_INT32),
        scalar=0,
    ):
        pypto.set_vec_tile_shapes(16, 16)
        sub_add_kernel(a, b, c, d, res, scalar)

    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    torch.npu.set_device(int(device_id))
    a = torch.ones(32, 32, dtype=torch.int32, device=f"npu:{device_id}")
    b = torch.ones(32, 32, dtype=torch.int32, device=f"npu:{device_id}")
    c = torch.ones(32, 32, dtype=torch.int32, device=f"npu:{device_id}")
    d = torch.ones(32, 32, dtype=torch.int32, device=f"npu:{device_id}")
    res = torch.ones(32, 32, dtype=torch.int32, device=f"npu:{device_id}")
    add_kernel(a, b, c, d, res, scalar=1)
    assert res.shape == (32, 32)
    assert torch.allclose(res.cpu().float(), torch.ones(32, 32) * 4 + 1)


# =============================================================================
# @fucntion检验场景验证 固定轴
# =============================================================================
@pypto.options(pass_options={"enable_slice": True})
def test_add_with_kwargs_check_stable():
    @pypto.frontend.function
    def sub_add_kernel(
        a: pypto.Tensor([], pypto.DT_INT32),
        b: pypto.Tensor(dtype=pypto.DT_INT32),
        c: pypto.Tensor([...], pypto.DT_INT32),
        d: pypto.Tensor([pypto.STATIC, 32], pypto.DT_INT32),
        res: pypto.Tensor([32, 32], pypto.DT_INT32),
        scalar=0,
    ):
        res.move(a + b + c + d + scalar)

    @pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
    def add_kernel(
        a: pypto.Tensor(dtype=pypto.DT_INT32),
        b: pypto.Tensor([], pypto.DT_INT32),
        c: pypto.Tensor([...], pypto.DT_INT32),
        d: pypto.Tensor([pypto.STATIC, 32], pypto.DT_INT32),
        res: pypto.Tensor([32, 32], pypto.DT_INT32),
        scalar=0,
    ):
        pypto.set_vec_tile_shapes(16, 16)
        sub_add_kernel(a, b, c, d, res, scalar)

    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    torch.npu.set_device(int(device_id))
    a = torch.ones(32, 32, dtype=torch.int32, device=f"npu:{device_id}")
    b = torch.ones(32, 32, dtype=torch.int32, device=f"npu:{device_id}")
    c = torch.ones(32, 32, dtype=torch.int32, device=f"npu:{device_id}")
    d = torch.ones(32, 31, dtype=torch.int32, device=f"npu:{device_id}")
    res = torch.ones(32, 32, dtype=torch.int32, device=f"npu:{device_id}")
    try:
        add_kernel(a, b, c, d, res, scalar=1)
    except Exception as e:
        assert "does not match the shape" in str(e)


# =============================================================================
# @fucntion检验场景验证 dtype
# =============================================================================
@pypto.options(pass_options={"enable_slice": True})
def test_add_with_kwargs_check_dtype():
    @pypto.frontend.function
    def sub_add_kernel(
        a: pypto.Tensor([], pypto.DT_INT32),
        b: pypto.Tensor(dtype=pypto.DT_INT32),
        c: pypto.Tensor([...], pypto.DT_INT32),
        d: pypto.Tensor([pypto.STATIC, 32], pypto.DT_INT32),
        res: pypto.Tensor([32, 32], pypto.DT_INT32),
        scalar=0,
    ):
        res.move(a + b + c + d + scalar)

    @pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
    def add_kernel(
        a: pypto.Tensor(dtype=pypto.DT_INT32),
        b: pypto.Tensor([], pypto.DT_INT32),
        c: pypto.Tensor([...], pypto.DT_INT32),
        d: pypto.Tensor([pypto.STATIC, 32], pypto.DT_INT32),
        res: pypto.Tensor([32, 32], pypto.DT_INT32),
        scalar=0,
    ):
        pypto.set_vec_tile_shapes(16, 16)
        sub_add_kernel(a, b, c, d, res, scalar)

    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    torch.npu.set_device(int(device_id))
    a = torch.ones(32, 32, dtype=torch.float32, device=f"npu:{device_id}")
    b = torch.ones(32, 32, dtype=torch.int32, device=f"npu:{device_id}")
    c = torch.ones(32, 32, dtype=torch.int32, device=f"npu:{device_id}")
    d = torch.ones(32, 32, dtype=torch.int32, device=f"npu:{device_id}")
    res = torch.ones(32, 32, dtype=torch.int32, device=f"npu:{device_id}")
    try:
        add_kernel(a, b, c, d, res, scalar=1)
    except Exception as e:
        assert "does not match the dtype" in str(e)
