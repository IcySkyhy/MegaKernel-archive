# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Matrix compute APIs (matmul / matmul_acc).

Doc: docs/zh/pypto_pro/api/SIMD-API/计算API/矩阵计算/

Verifies single matmul (auto_mutex + make_tile_group) and K-dimension
accumulated matmul_acc (phase partial/final + layout transform).
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
# matmul.md —— 单次 matmul，K 不分块（克隆 test_matmul_add_buffer_manager.py 的 cube 段）
#   C[M,N] = A[M,K] @ B[K,N]，K = TILE 一次性装入，每个 [i,j] 块一次 matmul。
# ===========================================================================
M_SIZE = 256
K_SIZE_MM = 128  # K 恰好一个 tile，无需分块累加
N_SIZE = 256


@pl.jit(auto_mutex=True)
def matmul_kernel(
    a: pl.Tensor[[M_SIZE, K_SIZE_MM], pl.DT_FP16],
    b: pl.Tensor[[K_SIZE_MM, N_SIZE], pl.DT_FP16],
    c: pl.Tensor[[M_SIZE, N_SIZE], pl.DT_FP32],
):
    # L1 双缓冲（next() 轮转）
    a_l1_db = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, K_SIZE_MM], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x00000,
        mutex_ids=[0, 1],
    )
    b_l1_db = pl.make_tile_group(
        type=pl.TileType(shape=[K_SIZE_MM, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x10000,
        mutex_ids=[2, 3],
    )
    # L0A / L0B / Acc 单 tile group（current()）
    a_left = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, K_SIZE_MM], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000,
        mutex_ids=[4],
    )
    b_right = pl.make_tile_group(
        type=pl.TileType(shape=[K_SIZE_MM, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000,
        mutex_ids=[5],
    )
    acc = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
        addrs=0x0000,
        mutex_ids=[6],
    )

    with pl.section_cube():
        for i in pl.range(0, M_SIZE, TILE):  # M 维分块
            for j in pl.range(0, N_SIZE, TILE):  # N 维分块
                cur_a = a_l1_db.next()
                cur_b = b_l1_db.next()
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
def test_matmul_kernel():
    device = ST_DEVICE
    _require_a5(device)
    torch.manual_seed(0)
    a = torch.randn(M_SIZE, K_SIZE_MM, device=device, dtype=torch.float16)
    b = torch.randn(K_SIZE_MM, N_SIZE, device=device, dtype=torch.float16)
    c = torch.zeros(M_SIZE, N_SIZE, device=device, dtype=torch.float32)

    matmul_kernel(a, b, c)
    torch.npu.synchronize()

    ref = torch.matmul(a.float(), b.float())
    torch.testing.assert_close(c, ref, rtol=1e-2, atol=1e-2)
    logging.info("matmul result equal!")


# ===========================================================================
# matmul_acc.md —— K 维分块累加（克隆 test_matmul_perf_asw_4k 的累加配方）
#   C[128,128] = A[128,K_SIZE] @ B[K_SIZE,128]，K 分 K_SIZE//TILE 块累加。
#   单输出 tile（M=N=128），突出 matmul_acc 的 K 累加链。
# ===========================================================================
K_SIZE_ACC = 256  # 分 2 个 TILE 块累加


@pl.jit(auto_mutex=True)
def matmul_acc_kernel(
    a: pl.Tensor[[TILE, K_SIZE_ACC], pl.DT_FP16],
    b: pl.Tensor[[K_SIZE_ACC, TILE], pl.DT_FP16],
    c: pl.Tensor[[TILE, TILE], pl.DT_FP32],
):
    # L1 / L0 双缓冲（next() 轮转）
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x00000,
        mutex_ids=[0, 1],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x10000,
        mutex_ids=[2, 3],
    )
    a_left = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=0x0000,
        mutex_ids=[4, 5],
    )
    b_right = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000,
        mutex_ids=[6, 7],
    )
    # Acc：K 累加要求 fractal=1024
    acc = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, fractal=1024),
        addrs=0x0000,
        mutex_ids=[8],
    )

    with pl.section_cube():
        pl.system.set_mm_layout_transform(enabled=True)
        ac = acc.current()
        for k in pl.range(0, K_SIZE_ACC, TILE):  # K 维分块（累加）
            cur_a = a_l1.next()
            cur_b = b_l1.next()
            al = a_left.next()
            br = b_right.next()
            pl.load(cur_a, a, [0, k])
            pl.load(cur_b, b, [k, 0])
            pl.move(al, cur_a)
            pl.move(br, cur_b)
            if k == 0:
                pl.matmul(ac, al, br, phase=pl.AccPhase.Partial)  # 首块写入累加器
            else:
                pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Final)  # 末块累加（K=2 块）
        pl.store(c, ac, [0, 0], phase=pl.STPhase.Final)
        pl.system.set_mm_layout_transform(enabled=False)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_matmul_acc_kernel():
    device = ST_DEVICE
    _require_a5(device)
    torch.manual_seed(0)
    a = torch.randn(TILE, K_SIZE_ACC, device=device, dtype=torch.float16)
    b = torch.randn(K_SIZE_ACC, TILE, device=device, dtype=torch.float16)
    c = torch.zeros(TILE, TILE, device=device, dtype=torch.float32)

    matmul_acc_kernel(a, b, c)
    torch.npu.synchronize()

    ref = torch.matmul(a.float(), b.float())
    torch.testing.assert_close(c, ref, rtol=1e-2, atol=1e-2)
    logging.info("matmul_acc result equal!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    test_matmul_kernel()
    test_matmul_acc_kernel()
    logging.info("\nAll matrix-computation doc examples passed!")
