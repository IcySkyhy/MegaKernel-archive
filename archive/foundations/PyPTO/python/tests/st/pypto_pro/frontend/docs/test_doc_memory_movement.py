# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Memory data movement APIs (load/store/load_tile/store_tile/move/insert/ssbuf).

Doc: docs/zh/pypto_pro/api/SIMD-API/计算API/Memory数据搬运/

Verifies GM-UB data transfer: pure vector load/store, cube-path move
(GM->L1->L0->matmul->GM), insert with cross-core sync, and ssbuf copy.
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
# load.md / store.md —— GM -> UB -> 相加 -> 写回 GM（纯 vector，auto_mutex）
# ===========================================================================
@pl.jit(auto_mutex=True)
def add_kernel(
    a: pl.Tensor[[64, 64], pl.DT_FP16],
    b: pl.Tensor[[64, 64], pl.DT_FP16],
    out: pl.Tensor[[64, 64], pl.DT_FP16],
):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_b = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[2])

    with pl.section_vector():
        cur_a = tile_a.current()
        cur_b = tile_b.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.add(cur_out, cur_a, cur_b)
        pl.store(out, cur_out, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_add_kernel():
    device = ST_DEVICE
    _require_a5(device)
    torch.manual_seed(0)
    shape = [64, 64]
    a = torch.rand(shape, device=device, dtype=torch.float16)
    b = torch.rand(shape, device=device, dtype=torch.float16)
    out = torch.empty(shape, device=device, dtype=torch.float16)

    add_kernel(a, b, out)
    torch.npu.synchronize()

    torch.testing.assert_close(out, a + b, rtol=1e-2, atol=1e-2)
    logging.info("load/store add_kernel result equal!")


# ===========================================================================
# load_tile.md —— GM 按 tile 块索引规整切分，逐块 load 后翻倍写回对应位置
#   用块索引 [ti,0] 定位（内部换算绝对偏移 [ti*64,0]）。循环内每次迭代独立，
#   用双缓冲 ping-pong（.next() 轮转），auto_mutex 自动管理搬运与计算间同步。
# ===========================================================================
@pl.jit(auto_mutex=True)
def load_tile_kernel(
    x: pl.Tensor[[256, 64], pl.DT_FP16],  # 4 个 64x64 的块
    out: pl.Tensor[[256, 64], pl.DT_FP16],
):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    x_db = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0, 1])
    out_db = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[2, 3])

    with pl.section_vector():
        for ti in pl.range(0, 4, 1):
            cur_x = x_db.next()
            cur_out = out_db.next()
            pl.load_tile(cur_x, x, [ti, 0])
            pl.add(cur_out, cur_x, cur_x)  # 翻倍，验证 load_tile 取到了正确的块
            pl.store_tile(out, cur_out, [ti, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_load_tile_kernel():
    device = ST_DEVICE
    _require_a5(device)
    torch.manual_seed(0)
    x = torch.rand([256, 64], device=device, dtype=torch.float16)
    out = torch.empty([256, 64], device=device, dtype=torch.float16)

    load_tile_kernel(x, out)
    torch.npu.synchronize()

    torch.testing.assert_close(out, x + x, rtol=1e-2, atol=1e-2)
    logging.info("load_tile load_tile_kernel result equal!")


# ===========================================================================
# store_tile.md —— 同一块 UB 结果按块索引逐块写到 GM 不同位置（纯 vector）
#   无跨迭代依赖（只读一次、多次写不同位置），auto_mutex 自动处理 load→store 同步。
# ===========================================================================
@pl.jit(auto_mutex=True)
def store_tile_kernel(
    x: pl.Tensor[[64, 64], pl.DT_FP16],
    out: pl.Tensor[[256, 64], pl.DT_FP16],  # 4 个 64x64 的块
):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile_x = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])

    with pl.section_vector():
        cur_x = tile_x.current()
        pl.load(cur_x, x, [0, 0])
        for ti in pl.range(0, 4, 1):
            pl.store_tile(out, cur_x, [ti, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_store_tile_kernel():
    device = ST_DEVICE
    _require_a5(device)
    torch.manual_seed(0)
    x = torch.rand([64, 64], device=device, dtype=torch.float16)
    out = torch.empty([256, 64], device=device, dtype=torch.float16)

    store_tile_kernel(x, out)
    torch.npu.synchronize()

    ref = x.repeat(4, 1)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
    logging.info("store_tile store_tile_kernel result equal!")


# ===========================================================================
# move.md —— 完整 matmul：GM->L1 (load), L1->L0A/L0B (move), cube, L0C->GM (store)
#   单次 matmul（K=64 不分块），照 buffer_manager：tile_group + auto_mutex，无 phase/fractal。
# ===========================================================================
@pl.jit(auto_mutex=True)
def matmul_move_kernel(
    a: pl.Tensor[[64, 64], pl.DT_FP16],
    b: pl.Tensor[[64, 64], pl.DT_FP16],
    out: pl.Tensor[[64, 64], pl.DT_FP32],
):
    tt_mat = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
    tt_left = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left)
    tt_right = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right)
    tt_acc = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc)

    a_l1 = pl.make_tile_group(type=tt_mat, addrs=0x0000, mutex_ids=[0])
    b_l1 = pl.make_tile_group(type=tt_mat, addrs=0x2000, mutex_ids=[1])
    a_l0a = pl.make_tile_group(type=tt_left, addrs=0x0000, mutex_ids=[2])
    b_l0b = pl.make_tile_group(type=tt_right, addrs=0x0000, mutex_ids=[3])
    c_l0c = pl.make_tile_group(type=tt_acc, addrs=0x0000, mutex_ids=[4])

    with pl.section_cube():
        cur_a_l1 = a_l1.current()
        cur_b_l1 = b_l1.current()
        cur_a_l0a = a_l0a.current()
        cur_b_l0b = b_l0b.current()
        cur_c_l0c = c_l0c.current()
        pl.load(cur_a_l1, a, [0, 0])  # GM -> L1
        pl.load(cur_b_l1, b, [0, 0])
        pl.move(cur_a_l0a, cur_a_l1)  # L1 -> L0A
        pl.move(cur_b_l0b, cur_b_l1)  # L1 -> L0B
        pl.matmul(cur_c_l0c, cur_a_l0a, cur_b_l0b)
        pl.store(out, cur_c_l0c, [0, 0])  # L0C -> GM（源在 Acc，走 FIX 流水）


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_matmul_move_kernel():
    device = ST_DEVICE
    _require_a5(device)
    torch.manual_seed(0)
    a = torch.randn([64, 64], device=device, dtype=torch.float16)
    b = torch.randn([64, 64], device=device, dtype=torch.float16)
    out = torch.zeros([64, 64], device=device, dtype=torch.float32)

    matmul_move_kernel(a, b, out)
    torch.npu.synchronize()

    ref = torch.matmul(a.float(), b.float())
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
    logging.info("move matmul_move_kernel result equal!")


# ===========================================================================
# insert.md —— UB 计算结果 move 成 NZ，再 insert 拼入 L1 NZ 缓冲，供 cube matmul
#   严格对照已验证范式 a5/matmul/test_single_dst.py（FP32 + subcore 切 32 行 +
#   ND->NZ move + insert + 手写 cross-core，组内流水同步交给 auto_mutex）。
#   计算：v1 = x + y（经 UB->L1 NZ 拼接），out = v1 @ rhs。
# ===========================================================================
@pl.jit(auto_mutex=True)
def insert_matmul_kernel(
    x: pl.Tensor[[64, 64], pl.DT_FP32],
    y: pl.Tensor[[64, 64], pl.DT_FP32],
    rhs: pl.Tensor[[64, 64], pl.DT_FP32],
    out: pl.Tensor[[64, 64], pl.DT_FP32],
):
    v1_mat_group = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x10000,
        mutex_ids=[0],
    )

    with pl.section_vector():
        sub_index = pl.get_subblock_idx()
        off = sub_index * 32

        tile_x_group = pl.make_tile_group(
            type=pl.TileType(shape=[32, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addrs=0x0000,
            mutex_ids=[1],
        )
        tile_y_group = pl.make_tile_group(
            type=pl.TileType(shape=[32, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addrs=0x2000,
            mutex_ids=[2],
        )
        tile_sum_group = pl.make_tile_group(
            type=pl.TileType(shape=[32, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addrs=0x4000,
            mutex_ids=[3],
        )
        tile_nz_group = pl.make_tile_group(
            type=pl.TileType(shape=[32, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.NZ),
            addrs=0x6000,
            mutex_ids=[4],
        )
        v1_mat = v1_mat_group.current()
        tile_x = tile_x_group.current()
        tile_y = tile_y_group.current()
        tile_sum = tile_sum_group.current()
        tile_nz = tile_nz_group.current()

        pl.load(tile_x, x, [off, 0])
        pl.load(tile_y, y, [off, 0])

        pl.add(tile_sum, tile_x, tile_y)
        pl.move(tile_nz, tile_sum)  # ND -> NZ

        pl.insert(v1_mat, tile_nz, [off, 0])  # UB -> L1 NZ2NZ
        pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=2)

    with pl.section_cube():
        rhs_mat_group = pl.make_tile_group(
            type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
            addrs=0x0000,
            mutex_ids=[5],
        )
        v1_left_group = pl.make_tile_group(
            type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
            addrs=0x0000,
            mutex_ids=[6],
        )
        rhs_right_group = pl.make_tile_group(
            type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Right, layout=pl.ZN),
            addrs=0x0000,
            mutex_ids=[7],
        )
        c_l0c_group = pl.make_tile_group(
            type=pl.TileType(
                shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024
            ),
            addrs=0x0000,
            mutex_ids=[8],
        )
        v1_mat = v1_mat_group.current()
        rhs_mat = rhs_mat_group.current()
        v1_left = v1_left_group.current()
        rhs_right = rhs_right_group.current()
        c_l0c = c_l0c_group.current()

        pl.load(rhs_mat, rhs, [0, 0])
        pl.move(rhs_right, rhs_mat)

        pl.system.wait_cross_core(pipe=pl.PipeType.MTE1, event_id=2, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
        pl.move(v1_left, v1_mat)

        pl.matmul(c_l0c, v1_left, rhs_right)

        pl.store(out, c_l0c, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_insert_matmul_kernel():
    device = ST_DEVICE
    _require_a5(device)
    torch.manual_seed(0)
    x = torch.randn([64, 64], device=device, dtype=torch.float32)
    y = torch.randn([64, 64], device=device, dtype=torch.float32)
    rhs = torch.randn([64, 64], device=device, dtype=torch.float32)
    out = torch.zeros([64, 64], device=device, dtype=torch.float32)

    insert_matmul_kernel(x, y, rhs, out)
    torch.npu.synchronize()

    ref = torch.matmul((x + y).float(), rhs.float())
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
    logging.info("insert insert_matmul_kernel result equal!")


# ===========================================================================
# ssbuf_store.md / ssbuf_load.md —— 跨核传少量元数据（vector 写, cube 读）
#   摘自已验证的 a5/datacopy/test_ssbuf_copy.py，无 golden（仅 printf 观察）。
# ===========================================================================
@pl.jit()
def ssbuf_copy_kernel(x: pl.Tensor[[1], pl.DT_INT32]):
    message = pl.struct("Message", batch=0, block=0, offset=0)

    with pl.section_vector():
        message.batch = 8
        message.block = 1
        message.offset = 32768
        if pl.get_subblock_idx() == 0:
            pl.ssbuf_store(message, 0)
            pl.system.set_cross_core(pipe=pl.PipeType.S, event_id=15)

    with pl.section_cube():
        pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=15, sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)
        pl.ssbuf_load(message, 0)
        pl.printf("Get ssbuf message: batch=%d, block=%d, offset=%d", message.batch, message.block, message.offset)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_ssbuf_copy_kernel():
    device = ST_DEVICE
    _require_a5(device)
    x = torch.tensor([3], dtype=torch.int32).to(device)
    ssbuf_copy_kernel(x)
    torch.npu.synchronize()
    logging.info("ssbuf_copy_kernel ran (check printf: batch=8, block=1, offset=32768).")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    test_add_kernel()
    test_load_tile_kernel()
    test_store_tile_kernel()
    test_matmul_move_kernel()
    test_insert_matmul_kernel()
    test_ssbuf_copy_kernel()
    logging.info("\nAll memory-movement doc examples passed!")
