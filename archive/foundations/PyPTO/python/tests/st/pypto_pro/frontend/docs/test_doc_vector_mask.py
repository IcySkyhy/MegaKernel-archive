# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Vector mask APIs (reset_mask / set_mask_count / set_mask_norm / set_vec_mask).

Uses make_tile_group + auto_mutex style.

Doc: docs/zh/pypto_pro/api/SIMD-API/计算API/Memory矢量计算/向量掩码/

Verifies all mask APIs compile and run without error, and that block-level
ops remain correct after mask manipulation (mask state not corrupted).
"""

import logging
import os

import pypto_pro.language as pl
import pytest
import torch

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"


def _require_a5(device):
    try:
        torch.npu.set_device(device)
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
    name = torch.npu.get_device_name()
    if "Ascend950" not in name:
        pytest.skip(f"Current device is {name}, not A5 (Ascend950). Skip.")


# ===========================================================================
# reset_mask —— 重置掩码为全 1，然后做 add 验证结果正确
# ===========================================================================
@pl.jit(auto_mutex=True)
def reset_mask_kernel(
    a: pl.Tensor[[64, 64], pl.DT_FP32],
    b: pl.Tensor[[64, 64], pl.DT_FP32],
    out: pl.Tensor[[64, 64], pl.DT_FP32],
):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_b = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_b = tile_b.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.reset_mask()
        pl.add(cur_out, cur_a, cur_b)
        pl.store(out, cur_out, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_reset_mask():
    device = ST_DEVICE
    _require_a5(device)
    torch.manual_seed(0)
    a = torch.randn(64, 64, device=device, dtype=torch.float32)
    b = torch.randn(64, 64, device=device, dtype=torch.float32)
    out = torch.zeros(64, 64, device=device, dtype=torch.float32)
    reset_mask_kernel(a, b, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a + b, rtol=1e-2, atol=1e-2)
    logging.info("reset_mask result equal!")


# ===========================================================================
# set_vec_mask —— 设置全 1 掩码（-1 即全 1），然后做 add 验证结果正确
# ===========================================================================
@pl.jit(auto_mutex=True)
def set_vec_mask_kernel(
    a: pl.Tensor[[64, 64], pl.DT_FP32],
    b: pl.Tensor[[64, 64], pl.DT_FP32],
    out: pl.Tensor[[64, 64], pl.DT_FP32],
):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_b = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_b = tile_b.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.set_vec_mask(-1, -1)
        pl.add(cur_out, cur_a, cur_b)
        pl.store(out, cur_out, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_set_vec_mask():
    device = ST_DEVICE
    _require_a5(device)
    torch.manual_seed(0)
    a = torch.randn(64, 64, device=device, dtype=torch.float32)
    b = torch.randn(64, 64, device=device, dtype=torch.float32)
    out = torch.zeros(64, 64, device=device, dtype=torch.float32)
    set_vec_mask_kernel(a, b, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a + b, rtol=1e-2, atol=1e-2)
    logging.info("set_vec_mask result equal!")


# ===========================================================================
# set_mask_count + set_mask_norm —— 切换掩码模式后恢复 norm，然后做 add 验证结果正确
# ===========================================================================
@pl.jit(auto_mutex=True)
def mask_count_norm_kernel(
    a: pl.Tensor[[64, 64], pl.DT_FP32],
    b: pl.Tensor[[64, 64], pl.DT_FP32],
    out: pl.Tensor[[64, 64], pl.DT_FP32],
):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_b = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_b = tile_b.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.set_mask_count()
        pl.set_vec_mask(0, 64 * 64)
        pl.set_mask_norm()
        pl.set_vec_mask(-1, -1)
        pl.add(cur_out, cur_a, cur_b)
        pl.store(out, cur_out, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_mask_count_norm():
    device = ST_DEVICE
    _require_a5(device)
    torch.manual_seed(0)
    a = torch.randn(64, 64, device=device, dtype=torch.float32)
    b = torch.randn(64, 64, device=device, dtype=torch.float32)
    out = torch.zeros(64, 64, device=device, dtype=torch.float32)
    mask_count_norm_kernel(a, b, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a + b, rtol=1e-2, atol=1e-2)
    logging.info("set_mask_count/set_mask_norm result equal!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    test_reset_mask()
    test_set_vec_mask()
    test_mask_count_norm()
    logging.info("\nAll vector mask examples passed!")
