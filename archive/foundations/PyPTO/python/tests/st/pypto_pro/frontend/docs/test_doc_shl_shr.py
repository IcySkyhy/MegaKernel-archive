# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""UT for doc examples — shl / shr element-wise shift ops.

Verifies kernel examples from:
  docs/zh/pypto_pro/api/SIMD-API/operation/memory_vector_computation/elementwise/shl.md
  docs/zh/pypto_pro/api/SIMD-API/operation/memory_vector_computation/elementwise/shr.md
"""

import logging
import os

import pypto_pro.language as pl
import pytest
import torch

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


@pl.jit(auto_mutex=True)
def shl_kernel(a: pl.Tensor[[64, 64], pl.DT_INT32],
               b: pl.Tensor[[64, 64], pl.DT_INT32],
               out: pl.Tensor[[64, 64], pl.DT_INT32]):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_b = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_b = tile_b.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.shl(cur_out, cur_a, cur_b)
        pl.store(out, cur_out, [0, 0])


@pl.jit(auto_mutex=True)
def shr_kernel(a: pl.Tensor[[64, 64], pl.DT_INT32],
               b: pl.Tensor[[64, 64], pl.DT_INT32],
               out: pl.Tensor[[64, 64], pl.DT_INT32]):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_b = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_b = tile_b.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.shr(cur_out, cur_a, cur_b)
        pl.store(out, cur_out, [0, 0])


@pytest.mark.soc("950")
def test_shl():
    device = ST_DEVICE
    _require_a5(device)
    torch.manual_seed(0)
    a = torch.arange(1, 64 * 64 + 1, device=device, dtype=torch.int32).reshape(64, 64)
    b = (torch.arange(64 * 64, device=device, dtype=torch.int32).reshape(64, 64) % 8) + 1
    out = torch.zeros((64, 64), device=device, dtype=torch.int32)
    shl_kernel(a, b, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, torch.bitwise_left_shift(a, b), rtol=0, atol=0)
    logging.info("test_shl passed!")


@pytest.mark.soc("950")
def test_shr():
    device = ST_DEVICE
    _require_a5(device)
    torch.manual_seed(0)
    a = (torch.arange(64 * 64, device=device, dtype=torch.int32).reshape(64, 64) + 2) * 4
    b = (torch.arange(64 * 64, device=device, dtype=torch.int32).reshape(64, 64) % 8) + 1
    out = torch.zeros((64, 64), device=device, dtype=torch.int32)
    shr_kernel(a, b, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, torch.bitwise_right_shift(a, b), rtol=0, atol=0)
    logging.info("test_shr passed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    test_shl()
    test_shr()
    logging.info("\nAll shift op tests passed!")
