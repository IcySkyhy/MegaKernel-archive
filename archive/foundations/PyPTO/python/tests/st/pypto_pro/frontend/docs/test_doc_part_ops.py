# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""UT for doc examples — partmax / partmin / partmul fused ops.

Verifies kernel examples from:
  docs/zh/pypto_pro/api/SIMD-API/operation/memory_vector_computation/fused_vector_computation/partmax.md
  docs/zh/pypto_pro/api/SIMD-API/operation/memory_vector_computation/fused_vector_computation/partmin.md
  docs/zh/pypto_pro/api/SIMD-API/operation/memory_vector_computation/fused_vector_computation/partmul.md
"""

import logging
import os

import pypto_pro.language as pl
import pytest
import torch

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

M = 64
N = 128


def _require_a5(device):
    try:
        torch.npu.set_device(device)
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
    name = torch.npu.get_device_name()
    if "Ascend950" not in name:
        pytest.skip(f"Current device is {name}, not A5 (Ascend950). Skip.")


@pl.jit(auto_mutex=True)
def partmax_kernel(x: pl.Tensor[[M, N], pl.DT_FP16],
                   y: pl.Tensor[[M, N], pl.DT_FP16],
                   out: pl.Tensor[[M, N], pl.DT_FP16]):
    tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16,
                     target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    gx = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    gy = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[1])
    gout = pl.make_tile_group(type=tt, addrs=0x10000, mutex_ids=[2])
    with pl.section_vector():
        tx, ty, tout = gx.current(), gy.current(), gout.current()
        pl.set_validshape(tx, [64, 128])
        pl.load(tx, x, [0, 0])
        pl.set_validshape(ty, [32, 128])
        pl.load(ty, y, [0, 0])
        pl.set_validshape(tout, [64, 128])
        pl.partmax(tout, tx, ty)
        pl.store(out, tout, [0, 0])


@pl.jit(auto_mutex=True)
def partmin_kernel(x: pl.Tensor[[M, N], pl.DT_FP16],
                   y: pl.Tensor[[M, N], pl.DT_FP16],
                   out: pl.Tensor[[M, N], pl.DT_FP16]):
    tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16,
                     target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    gx = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    gy = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[1])
    gout = pl.make_tile_group(type=tt, addrs=0x10000, mutex_ids=[2])
    with pl.section_vector():
        tx, ty, tout = gx.current(), gy.current(), gout.current()
        pl.set_validshape(tx, [64, 128])
        pl.load(tx, x, [0, 0])
        pl.set_validshape(ty, [32, 128])
        pl.load(ty, y, [0, 0])
        pl.set_validshape(tout, [64, 128])
        pl.partmin(tout, tx, ty)
        pl.store(out, tout, [0, 0])


@pl.jit(auto_mutex=True)
def partmul_kernel(x: pl.Tensor[[M, N], pl.DT_FP16],
                   y: pl.Tensor[[M, N], pl.DT_FP16],
                   out: pl.Tensor[[M, N], pl.DT_FP16]):
    tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16,
                     target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    gx = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    gy = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[1])
    gout = pl.make_tile_group(type=tt, addrs=0x10000, mutex_ids=[2])
    with pl.section_vector():
        tx, ty, tout = gx.current(), gy.current(), gout.current()
        pl.set_validshape(tx, [64, 128])
        pl.load(tx, x, [0, 0])
        pl.set_validshape(ty, [32, 128])
        pl.load(ty, y, [0, 0])
        pl.set_validshape(tout, [64, 128])
        pl.partmul(tout, tx, ty)
        pl.store(out, tout, [0, 0])


@pytest.mark.soc("950")
def test_partmax():
    _require_a5(ST_DEVICE)
    torch.manual_seed(0)
    x = torch.randn(M, N, device=ST_DEVICE, dtype=torch.float16)
    y = torch.randn(M, N, device=ST_DEVICE, dtype=torch.float16)
    out = torch.empty(M, N, device=ST_DEVICE, dtype=torch.float16)
    partmax_kernel(x, y, out)
    torch.npu.synchronize()
    ref = x.clone()
    ref[:32, :] = torch.maximum(x[:32, :], y[:32, :])
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
    logging.info("test_partmax passed!")


@pytest.mark.soc("950")
def test_partmin():
    _require_a5(ST_DEVICE)
    torch.manual_seed(0)
    x = torch.randn(M, N, device=ST_DEVICE, dtype=torch.float16)
    y = torch.randn(M, N, device=ST_DEVICE, dtype=torch.float16)
    out = torch.empty(M, N, device=ST_DEVICE, dtype=torch.float16)
    partmin_kernel(x, y, out)
    torch.npu.synchronize()
    ref = x.clone()
    ref[:32, :] = torch.minimum(x[:32, :], y[:32, :])
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
    logging.info("test_partmin passed!")


@pytest.mark.soc("950")
def test_partmul():
    _require_a5(ST_DEVICE)
    torch.manual_seed(0)
    x = torch.randn(M, N, device=ST_DEVICE, dtype=torch.float16)
    y = torch.randn(M, N, device=ST_DEVICE, dtype=torch.float16)
    out = torch.empty(M, N, device=ST_DEVICE, dtype=torch.float16)
    partmul_kernel(x, y, out)
    torch.npu.synchronize()
    ref = x.clone()
    ref[:32, :] = x[:32, :] * y[:32, :]
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
    logging.info("test_partmul passed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    test_partmax()
    test_partmin()
    test_partmul()
    logging.info("\nAll part op tests passed!")
