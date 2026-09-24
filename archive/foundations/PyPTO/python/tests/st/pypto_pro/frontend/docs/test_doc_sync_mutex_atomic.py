# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""sync_all, mutex_lock/mutex_unlock, and store(atomic="add").

Doc: docs/zh/pypto_pro/api/SIMD-API/计算API/{同步控制/sync_all, 同步控制/mutex_lock_mutex_unlock, 原子操作/store_atomic}

Pure vector mode. Verifies full-pipeline sync, manual mutex for load/store,
and atomic add write-back to GM.
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
# sync_all —— 全流水同步替代 sync_src/sync_dst
#   纯 vector kernel 用 core_type="aiv_only"，避免同步不存在的 cube 核。
# ===========================================================================
@pl.jit(auto_mutex=False)
def sync_all_kernel(
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
        pl.system.sync_all(core_type=pl.SyncCoreType.AIV_ONLY)
        pl.add(tile_out, tile_a, tile_b)
        pl.system.sync_all(core_type=pl.SyncCoreType.AIV_ONLY)
        pl.store(out, tile_out, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_sync_all():
    device = ST_DEVICE
    _require_a5(device)
    torch.manual_seed(0)
    a = torch.randn(64, 64, device=device, dtype=torch.float32)
    b = torch.randn(64, 64, device=device, dtype=torch.float32)
    out = torch.zeros(64, 64, device=device, dtype=torch.float32)
    sync_all_kernel(a, b, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a + b, rtol=1e-2, atol=1e-2)
    logging.info("sync_all result equal!")


# ===========================================================================
# mutex_lock / mutex_unlock —— 手动互斥锁保护 load/store
#   mutex 设计用于 make_tile_group 缓冲区，单 buffer 也可手动加锁。
# ===========================================================================
@pl.jit(auto_mutex=False)
def mutex_kernel(
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

        pl.system.mutex_lock(pipe=pl.PipeType.MTE2, mutex_id=0)
        pl.load(cur_a, a, [0, 0])
        pl.system.mutex_unlock(pipe=pl.PipeType.MTE2, mutex_id=0)

        pl.system.mutex_lock(pipe=pl.PipeType.MTE2, mutex_id=1)
        pl.load(cur_b, b, [0, 0])
        pl.system.mutex_unlock(pipe=pl.PipeType.MTE2, mutex_id=1)

        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.add(cur_out, cur_a, cur_b)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)

        pl.system.mutex_lock(pipe=pl.PipeType.MTE3, mutex_id=2)
        pl.store(out, cur_out, [0, 0])
        pl.system.mutex_unlock(pipe=pl.PipeType.MTE3, mutex_id=2)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_mutex_lock_unlock():
    device = ST_DEVICE
    _require_a5(device)
    torch.manual_seed(0)
    a = torch.randn(64, 64, device=device, dtype=torch.float32)
    b = torch.randn(64, 64, device=device, dtype=torch.float32)
    out = torch.zeros(64, 64, device=device, dtype=torch.float32)
    mutex_kernel(a, b, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a + b, rtol=1e-2, atol=1e-2)
    logging.info("mutex_lock/mutex_unlock result equal!")


# ===========================================================================
# store(atomic="add") —— 原子累加写回 GM
#   照搬 lightning_indexer_grad 范式：FP32 小 tile [1, 128]，循环逐行累加。
#   load 后 sync MTE2→MTE3（等 load 完成再 store），
#   store 后 sync MTE3→MTE2（等 store 完成再下一轮 load 复用 buffer）。
#   单核验证：out 初始为 0，atomic add 后 out == a。
# ===========================================================================
D = 128


@pl.jit()
def atomic_add_kernel(
    a: pl.Tensor[[64, D], pl.DT_FP32],
    out: pl.Tensor[[64, D], pl.DT_FP32],
):
    tt = pl.TileType(shape=[1, D], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    with pl.section_vector():
        for i in pl.range(0, 64):
            cur_a = tile_a.current()
            pl.load(cur_a, a, [i, 0])
            pl.store(out, cur_a, [i, 0], atomic=pl.AtomicType.AtomicAdd)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_store_atomic():
    device = ST_DEVICE
    _require_a5(device)
    torch.manual_seed(0)
    a = torch.randn(64, D, device=device, dtype=torch.float32)
    out = torch.zeros(64, D, device=device, dtype=torch.float32)
    atomic_add_kernel(a, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a, rtol=1e-2, atol=1e-2)
    logging.info("store(atomic='add') result equal!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    test_sync_all()
    test_mutex_lock_unlock()
    test_store_atomic()
    logging.info("\nAll sync_all/mutex/atomic examples passed!")
