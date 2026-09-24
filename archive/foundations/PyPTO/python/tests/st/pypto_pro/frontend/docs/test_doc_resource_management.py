# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Resource management APIs (make_tile / addptr / make_tensor / make_tile_group).

Doc: docs/zh/pypto_pro/api/SIMD-API/计算API/资源管理/

Verifies tile/tensor creation and pointer arithmetic. make_tile_group tested
within a minimal matmul context (requires auto_mutex).
"""

import logging
import os

import pypto_pro.language as pl
import pytest
import torch

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

TILE = 128


def _require_a5(device):
    try:
        torch.npu.set_device(device)
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
    name = torch.npu.get_device_name()
    if "Ascend950" not in name:
        pytest.skip(f"Current device is {name}, not A5 (Ascend950). Skip.")


# ===========================================================================
# make_tile.md —— 在 UB 用 make_tile 分配输入/输出做 element-wise 加法
#   纯 vector kernel，同步用 sync_src/sync_dst 手写。
# ===========================================================================
@pl.jit()
def make_tile_add_kernel(
    a: pl.Tensor[[64, 128], pl.DT_FP16],
    b: pl.Tensor[[64, 128], pl.DT_FP16],
    out: pl.Tensor[[64, 128], pl.DT_FP16],
):
    tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    # size缺省，由tt推导为64 * 128 * 2 = 16384字节
    tile_a = pl.make_tile(tt, addr=0x0000)
    tile_b = pl.make_tile(tt, addr=0x4000)
    tile_out = pl.make_tile(tt, addr=0x8000)

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
def test_make_tile_add_kernel():
    device = ST_DEVICE
    _require_a5(device)
    torch.manual_seed(0)
    shape = [64, 128]
    a = torch.rand(shape, device=device, dtype=torch.float16)
    b = torch.rand(shape, device=device, dtype=torch.float16)
    out = torch.empty(shape, device=device, dtype=torch.float16)

    make_tile_add_kernel(a, b, out)
    torch.npu.synchronize()

    torch.testing.assert_close(out, a + b, rtol=1e-2, atol=1e-2)
    logging.info("make_tile add result equal!")


# ===========================================================================
# make_tile_group.md —— 该接口仅在 matmul + auto_mutex 语境有通过先例
#   最小 matmul：L1 用 next() 双缓冲，L0A/L0B/Acc 单 mutex_id 用 current()。
#   （克隆 a5/matmul/test_matmul_add_buffer_manager.py 的 cube 段）
# ===========================================================================
MM_M, MM_K, MM_N = 256, 128, 256


@pl.jit(auto_mutex=True)
def tile_group_matmul_kernel(
    a: pl.Tensor[[MM_M, MM_K], pl.DT_FP16],
    b: pl.Tensor[[MM_K, MM_N], pl.DT_FP16],
    c: pl.Tensor[[MM_M, MM_N], pl.DT_FP32],
):
    # L1 双缓冲（next() 轮转）
    a_l1_db = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, MM_K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x00000,
        mutex_ids=[0, 1],
    )
    b_l1_db = pl.make_tile_group(
        type=pl.TileType(shape=[MM_K, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x10000,
        mutex_ids=[2, 3],
    )
    # L0A / L0B / Acc 单 tile group（current()）
    a_left = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, MM_K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000,
        mutex_ids=[4],
    )
    b_right = pl.make_tile_group(
        type=pl.TileType(shape=[MM_K, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000,
        mutex_ids=[5],
    )
    acc = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
        addrs=0x0000,
        mutex_ids=[6],
    )

    with pl.section_cube():
        for i in pl.range(0, MM_M, TILE):
            for j in pl.range(0, MM_N, TILE):
                cur_a = a_l1_db.next()  # 双缓冲轮转
                cur_b = b_l1_db.next()
                al = a_left.current()  # 单 tile group
                br = b_right.current()
                ac = acc.current()
                pl.load(cur_a, a, [i, 0])
                pl.load(cur_b, b, [0, j])
                pl.move(al, cur_a)
                pl.move(br, cur_b)
                pl.matmul(ac, al, br)
                pl.store(c, ac, [i, j])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_tile_group_matmul_kernel():
    device = ST_DEVICE
    _require_a5(device)
    torch.manual_seed(0)
    a = torch.randn(MM_M, MM_K, device=device, dtype=torch.float16)
    b = torch.randn(MM_K, MM_N, device=device, dtype=torch.float16)
    c = torch.zeros(MM_M, MM_N, device=device, dtype=torch.float32)

    tile_group_matmul_kernel(a, b, c)
    torch.npu.synchronize()

    ref = torch.matmul(a.float(), b.float())
    torch.testing.assert_close(c, ref, rtol=1e-2, atol=1e-2)
    logging.info("make_tile_group matmul result equal!")


# ===========================================================================
# make_tile_group.md 4-buffer —— L1 用 4 块 tile 轮转，验证 N-buffer 可用
#   mutex_ids 长度 4，next() 按 cursor % 4 自动轮转。
#   （与 2-buffer 相同的 matmul 逻辑，仅 L1 缓冲深度不同）
# ===========================================================================
@pl.jit(auto_mutex=True)
def tile_group_4buf_matmul_kernel(
    a: pl.Tensor[[MM_M, MM_K], pl.DT_FP16],
    b: pl.Tensor[[MM_K, MM_N], pl.DT_FP16],
    c: pl.Tensor[[MM_M, MM_N], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, MM_K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x00000,
        mutex_ids=[0, 1, 2, 3],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[MM_K, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x20000,
        mutex_ids=[4, 5, 6, 7],
    )
    a_left = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, MM_K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000,
        mutex_ids=[8],
    )
    b_right = pl.make_tile_group(
        type=pl.TileType(shape=[MM_K, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000,
        mutex_ids=[9],
    )
    acc = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
        addrs=0x0000,
        mutex_ids=[10],
    )

    with pl.section_cube():
        for i in pl.range(0, MM_M, TILE):
            for j in pl.range(0, MM_N, TILE):
                cur_a = a_l1.next()
                cur_b = b_l1.next()
                al = a_left.current()
                br = b_right.current()
                ac = acc.current()
                pl.load(cur_a, a, [i, 0])
                pl.load(cur_b, b, [0, j])
                pl.move(al, cur_a)
                pl.move(br, cur_b)
                pl.matmul(ac, al, br)
                pl.store(c, ac, [i, j])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_tile_group_4buf_matmul_kernel():
    device = ST_DEVICE
    _require_a5(device)
    torch.manual_seed(0)
    a = torch.randn(MM_M, MM_K, device=device, dtype=torch.float16)
    b = torch.randn(MM_K, MM_N, device=device, dtype=torch.float16)
    c = torch.zeros(MM_M, MM_N, device=device, dtype=torch.float32)

    tile_group_4buf_matmul_kernel(a, b, c)
    torch.npu.synchronize()

    ref = torch.matmul(a.float(), b.float())
    torch.testing.assert_close(c, ref, rtol=1e-2, atol=1e-2)
    logging.info("make_tile_group 4-buffer matmul result equal!")


# ===========================================================================
# addptr.md / make_tensor.md / Ptr.md —— 用 Ptr 接收 workspace 裸指针，
#   addptr 切出 workspace，make_tensor 用非连续行 stride 包装成 tensor view，
#   分两次完成 a*2 写回 out，并覆盖非零 offset 的 load/store。
#   vector kernel 开 `auto_mutex`，同步由 `make_tile_group` 自动管理。
# ===========================================================================
@pl.jit(auto_mutex=True)
def workspace_kernel(
    a: pl.Tensor[[64, 128], pl.DT_FP16],
    workspace: pl.Ptr[pl.DT_FP16],
    out: pl.Tensor[[64, 128], pl.DT_FP16],
):
    ws_buf_ptr = pl.addptr(workspace, 64 * 128)
    ws_buf = pl.make_tensor(ws_buf_ptr, [64, 128], [256, 1])

    tt = pl.TileType(shape=[32, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])

    with pl.section_vector():
        t = tile.current()
        pl.load(t, a, [0, 0])
        pl.add(t, t, t)
        pl.store(ws_buf, t, [0, 0])
        pl.load(t, ws_buf, [0, 0])
        pl.store(out, t, [0, 0])

        pl.load(t, a, [32, 0])
        pl.add(t, t, t)
        pl.store(ws_buf, t, [32, 0])
        pl.load(t, ws_buf, [32, 0])
        pl.store(out, t, [32, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_workspace_kernel():
    device = ST_DEVICE
    _require_a5(device)
    torch.manual_seed(0)
    shape = [64, 128]
    a = torch.rand(shape, device=device, dtype=torch.float16)
    out = torch.empty(shape, device=device, dtype=torch.float16)
    workspace = torch.full((64 * 128 + 64 * 256,), -7.0, device=device, dtype=torch.float16)

    workspace_kernel(a, workspace, out)
    torch.npu.synchronize()

    torch.testing.assert_close(out, a * 2, rtol=1e-2, atol=1e-2)
    pitched_workspace = workspace[64 * 128:].view(64, 256)
    torch.testing.assert_close(pitched_workspace[:, 128:], torch.full_like(pitched_workspace[:, 128:], -7.0))
    logging.info("addptr/make_tensor workspace result equal!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    test_make_tile_add_kernel()
    test_tile_group_matmul_kernel()
    test_tile_group_4buf_matmul_kernel()
    test_workspace_kernel()
    logging.info("\nAll resource-management doc examples passed!")
