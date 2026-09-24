# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Sync control APIs: sync_src/dst, barriers, sync_all, mutex, cross-core, layout transform.

Doc: docs/zh/pypto_pro/api/SIMD-API/计算API/同步控制/

Verifies pipeline sync, barriers (bar_all/bar_m), global sync,
manual mutex, cross-core communication, and matmul layout transform.
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
# sync_src / sync_dst —— 搬运与计算间的流水线同步
#   load 后 MTE2→V，计算后 V→MTE3。
# ===========================================================================
@pl.jit()
def sync_src_dst_kernel(
    a: pl.Tensor[[64, 64], pl.DT_FP32],
    b: pl.Tensor[[64, 64], pl.DT_FP32],
    out: pl.Tensor[[64, 64], pl.DT_FP32],
):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile(tt, addr=0x0000, size=16384)
    tile_b = pl.make_tile(tt, addr=0x4000, size=16384)
    tile_out = pl.make_tile(tt, addr=0x8000, size=16384)
    with pl.section_vector():
        pl.load(tile_a, a, [0, 0])
        pl.load(tile_b, b, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.add(tile_out, tile_a, tile_b)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, tile_out, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_sync_src_dst():
    device = ST_DEVICE
    _require_a5(device)
    torch.manual_seed(0)
    a = torch.randn(64, 64, device=device, dtype=torch.float32)
    b = torch.randn(64, 64, device=device, dtype=torch.float32)
    out = torch.zeros(64, 64, device=device, dtype=torch.float32)
    sync_src_dst_kernel(a, b, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a + b, rtol=1e-2, atol=1e-2)
    logging.info("sync_src/sync_dst result equal!")


# ===========================================================================
# bar_all —— 全流水线 barrier（循环体顶部同步）
#   在循环内 load 前插入 bar_all，确保上一轮 store 完成。
# ===========================================================================
@pl.jit()
def bar_all_kernel(
    x: pl.Tensor[[128, 64], pl.DT_FP16],
    out: pl.Tensor[[128, 64], pl.DT_FP16],
):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile_x = pl.make_tile(tt, addr=0x0000, size=8192)
    tile_out = pl.make_tile(tt, addr=0x2000, size=8192)
    with pl.section_vector():
        for i in pl.range(0, 128, 64):
            pl.system.bar_all()
            pl.load(tile_x, x, [i, 0])
            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
            pl.add(tile_out, tile_x, tile_x)
            pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
            pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
            pl.store(out, tile_out, [i, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_bar_all():
    device = ST_DEVICE
    _require_a5(device)
    torch.manual_seed(0)
    x = torch.rand(128, 64, device=device, dtype=torch.float16)
    out = torch.empty(128, 64, device=device, dtype=torch.float16)
    bar_all_kernel(x, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, x + x, rtol=1e-2, atol=1e-2)
    logging.info("bar_all result equal!")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    test_sync_src_dst()
    test_bar_all()
    logging.info("\nAll sync-control examples passed!")
