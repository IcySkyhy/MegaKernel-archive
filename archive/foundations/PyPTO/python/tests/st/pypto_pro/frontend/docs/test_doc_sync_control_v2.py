# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Sync control APIs (batch 2): bar_m, set_mm_layout_transform, cross-core sync.

Doc: docs/zh/pypto_pro/api/SIMD-API/计算API/同步控制/

Verifies cube-pipeline barrier between matmuls, layout transform for K
accumulation, and vector-cube cross-core communication.
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
# bar_m —— cube 流水线内 barrier
#   两次 matmul 之间插入 bar_m，确保前一次完成后再开始下一次。
#   第二次 matmul 覆盖 acc（非 matmul_acc），out = a @ b。
# ===========================================================================
TILE = 64


@pl.jit(auto_mutex=True)
def bar_m_kernel(
    a: pl.Tensor[[TILE, TILE], pl.DT_FP16],
    b: pl.Tensor[[TILE, TILE], pl.DT_FP16],
    out: pl.Tensor[[TILE, TILE], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x0000,
        mutex_ids=[0],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x2000,
        mutex_ids=[1],
    )
    a_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000,
        mutex_ids=[2],
    )
    b_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000,
        mutex_ids=[3],
    )
    c_l0c = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
        addrs=0x0000,
        mutex_ids=[4],
    )

    with pl.section_cube():
        cur_a = a_l1.current()
        cur_b = b_l1.current()
        al = a_l0a.current()
        br = b_l0b.current()
        ac = c_l0c.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.move(al, cur_a)
        pl.move(br, cur_b)
        pl.matmul(ac, al, br)
        pl.system.bar_m()
        pl.matmul(ac, al, br)
        pl.store(out, ac, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_bar_m():
    device = ST_DEVICE
    _require_a5(device)
    torch.manual_seed(0)
    a = torch.randn(TILE, TILE, device=device, dtype=torch.float16)
    b = torch.randn(TILE, TILE, device=device, dtype=torch.float16)
    out = torch.zeros(TILE, TILE, device=device, dtype=torch.float32)
    bar_m_kernel(a, b, out)
    torch.npu.synchronize()
    ref = torch.matmul(a.float(), b.float())
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
    logging.info("bar_m result equal!")


# ===========================================================================
# set_mm_layout_transform —— matmul K 累加必需的布局变换开关
#   照搬 test_doc_matrix_compute.py 的 matmul_acc_kernel。
# ===========================================================================
TILE_ACC = 128
K_SIZE_ACC = 256


@pl.jit(auto_mutex=True)
def mm_layout_kernel(
    a: pl.Tensor[[TILE_ACC, K_SIZE_ACC], pl.DT_FP16],
    b: pl.Tensor[[K_SIZE_ACC, TILE_ACC], pl.DT_FP16],
    c: pl.Tensor[[TILE_ACC, TILE_ACC], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_ACC, TILE_ACC], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x00000,
        mutex_ids=[0, 1],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_ACC, TILE_ACC], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x10000,
        mutex_ids=[2, 3],
    )
    a_left = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_ACC, TILE_ACC], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=0x0000,
        mutex_ids=[4, 5],
    )
    b_right = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_ACC, TILE_ACC], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000,
        mutex_ids=[6, 7],
    )
    acc = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_ACC, TILE_ACC], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, fractal=1024),
        addrs=0x0000,
        mutex_ids=[8],
    )

    with pl.section_cube():
        pl.system.set_mm_layout_transform(enabled=True)
        ac = acc.current()
        for k in pl.range(0, K_SIZE_ACC, TILE_ACC):
            cur_a = a_l1.next()
            cur_b = b_l1.next()
            al = a_left.next()
            br = b_right.next()
            pl.load(cur_a, a, [0, k])
            pl.load(cur_b, b, [k, 0])
            pl.move(al, cur_a)
            pl.move(br, cur_b)
            if k == 0:
                pl.matmul(ac, al, br, phase=pl.AccPhase.Partial)
            else:
                pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Final)
        pl.store(c, ac, [0, 0], phase=pl.STPhase.Final)
        pl.system.set_mm_layout_transform(enabled=False)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_set_mm_layout_transform():
    device = ST_DEVICE
    _require_a5(device)
    torch.manual_seed(0)
    a = torch.randn(TILE_ACC, K_SIZE_ACC, device=device, dtype=torch.float16)
    b = torch.randn(K_SIZE_ACC, TILE_ACC, device=device, dtype=torch.float16)
    c = torch.zeros(TILE_ACC, TILE_ACC, device=device, dtype=torch.float32)
    mm_layout_kernel(a, b, c)
    torch.npu.synchronize()
    ref = torch.matmul(a.float(), b.float())
    torch.testing.assert_close(c, ref, rtol=1e-2, atol=1e-2)
    logging.info("set_mm_layout_transform result equal!")


# ===========================================================================
# set_cross_core / wait_cross_core —— vector 与 cube 跨核通信
#   照搬 test_doc_memory_movement.py 的 insert_matmul_kernel。
# ===========================================================================
@pl.jit()
def cross_core_kernel(
    x: pl.Tensor[[64, 64], pl.DT_FP32],
    y: pl.Tensor[[64, 64], pl.DT_FP32],
    rhs: pl.Tensor[[64, 64], pl.DT_FP32],
    out: pl.Tensor[[64, 64], pl.DT_FP32],
):
    v1_mat = pl.make_tile(
        pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addr=0x10000,
        size=16384,
    )

    with pl.section_vector():
        sub_index = pl.get_subblock_idx()
        off = sub_index * 32

        tile_x = pl.make_tile(
            pl.TileType(shape=[32, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addr=0x0000, size=8192
        )
        tile_y = pl.make_tile(
            pl.TileType(shape=[32, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addr=0x2000, size=8192
        )
        tile_sum = pl.make_tile(
            pl.TileType(shape=[32, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addr=0x4000, size=8192
        )
        tile_nz = pl.make_tile(
            pl.TileType(shape=[32, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.NZ),
            addr=0x6000,
            size=8448,
        )

        pl.load(tile_x, x, [off, 0])
        pl.load(tile_y, y, [off, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.add(tile_sum, tile_x, tile_y)
        pl.move(tile_nz, tile_sum)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=2)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=2)
        pl.insert(v1_mat, tile_nz, [off, 0])
        pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=2)

    with pl.section_cube():
        rhs_mat = pl.make_tile(
            pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
            addr=0x0000,
            size=16384,
        )
        v1_left = pl.make_tile(
            pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
            addr=0x0000,
            size=16384,
        )
        rhs_right = pl.make_tile(
            pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Right, layout=pl.ZN),
            addr=0x0000,
            size=16384,
        )
        c_l0c = pl.make_tile(
            pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024),
            addr=0x0000,
            size=16384,
        )

        pl.load(rhs_mat, rhs, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.move(rhs_right, rhs_mat)
        pl.system.wait_cross_core(pipe=pl.PipeType.MTE1, event_id=2, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
        pl.move(v1_left, v1_mat)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        pl.matmul(c_l0c, v1_left, rhs_right)
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.store(out, c_l0c, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_set_cross_core_wait_cross_core():
    device = ST_DEVICE
    _require_a5(device)
    torch.manual_seed(0)
    x = torch.randn(64, 64, device=device, dtype=torch.float32)
    y = torch.randn(64, 64, device=device, dtype=torch.float32)
    rhs = torch.randn(64, 64, device=device, dtype=torch.float32)
    out = torch.zeros(64, 64, device=device, dtype=torch.float32)
    cross_core_kernel(x, y, rhs, out)
    torch.npu.synchronize()
    ref = torch.matmul((x + y).float(), rhs.float())
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
    logging.info("set_cross_core/wait_cross_core result equal!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    test_bar_m()
    test_set_mm_layout_transform()
    test_set_cross_core_wait_cross_core()
    logging.info("\nAll sync-control batch-2 examples passed!")
