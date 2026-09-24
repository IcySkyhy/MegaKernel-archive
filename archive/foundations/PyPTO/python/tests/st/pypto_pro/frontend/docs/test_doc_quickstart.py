# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Verify doc code from docs/guide/入门教程/快速入门/基于SIMD编程/.

make_tile_group + auto_mutex style.

Tests:
  - HelloWorld.md  : pl.printf + A[x] = v subscript
  - Add算子快速入门.md : make_tile_group + auto_mutex, no manual sync
  - Matmul算子快速入门.md : make_tile_group + auto_mutex, cube path
  - CV融合算子快速入门.md : make_tile_group + auto_mutex, fused Matmul+Softmax
"""

import os

import pypto_pro.language as pl
from pypto_pro.language import Vf
import pytest
import torch
import torch_npu  # noqa: F401 — registers npu backend

import pypto

vf = Vf

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"


def _require_a5():
    try:
        torch.npu.set_device(ST_DEVICE)
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
    name = torch.npu.get_device_name()
    if "Ascend950" not in name:
        pytest.skip(f"Current device is {name}, not A5 (Ascend950). Skip.")


# =============================================================================
# HelloWorld
# =============================================================================
@pl.jit()
def hello_world_kernel(out: pl.Tensor[[1], pl.DT_INT32]):
    with pl.section_vector():
        pl.printf("Hello World!!!\n")
        out[0] = 1


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_hello_world():
    _require_a5()
    device = ST_DEVICE
    out = torch.zeros(1, device=device, dtype=torch.int32)
    hello_world_kernel(out)
    torch.npu.synchronize()
    assert out[0].item() == 1, f"expected out[0]=1, got {out[0].item()}"


# =============================================================================
# Add 算子快速入门 — make_tile_group + auto_mutex（无需手写同步）
# =============================================================================
@pl.jit(auto_mutex=True)
def add_kernel(
    a: pl.Tensor[[64, 64], pl.DT_FP16], b: pl.Tensor[[64, 64], pl.DT_FP16], out: pl.Tensor[[64, 64], pl.DT_FP16]
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
    _require_a5()
    device = ST_DEVICE
    torch.manual_seed(0)
    a = torch.rand(64, 64, device=device, dtype=torch.float16)
    b = torch.rand(64, 64, device=device, dtype=torch.float16)
    out = torch.empty(64, 64, device=device, dtype=torch.float16)

    add_kernel(a, b, out)
    torch.npu.synchronize()

    torch.testing.assert_close(out, a + b, rtol=1e-2, atol=1e-2)


# =============================================================================
# Matmul 算子快速入门 —— make_tile_group + auto_mutex, cube path
# =============================================================================
@pl.jit(auto_mutex=True)
def matmul_example(
    a: pl.Tensor[[pl.DYNAMIC, 128], pl.DT_FP16],
    b: pl.Tensor[[128, pl.DYNAMIC], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    num_cores = pl.get_block_num()
    core_id = pl.get_block_idx()
    m = a.shape[0]
    n = b.shape[1]

    with pl.section_cube():
        a_mat_4_buffer = pl.make_tile_group(
            type=pl.TileType(shape=[128, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
            addrs=0,
            mutex_ids=[0, 1, 10, 11],
        )
        b_mat_4_buffer = pl.make_tile_group(
            type=pl.TileType(shape=[128, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
            addrs=0x20000,
            mutex_ids=[2, 3, 12, 13],
        )
        a_left_db = pl.make_tile_group(
            type=pl.TileType(shape=[128, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
            addrs=0,
            mutex_ids=[4, 5],
        )
        b_right_db = pl.make_tile_group(
            type=pl.TileType(shape=[128, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
            addrs=0,
            mutex_ids=[6, 7],
        )
        acc_db = pl.make_tile_group(
            type=pl.TileType(shape=[128, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
            addrs=0,
            mutex_ids=[8, 9],
        )

        for i in pl.range(core_id, m // 128, num_cores):
            for j in pl.range(0, n // 128, 1):
                a_l1_tile = a_mat_4_buffer.next()
                pl.load_tile(a_l1_tile, a, [i, 0])
                b_l1_tile = b_mat_4_buffer.next()
                pl.load_tile(b_l1_tile, b, [0, j])

                cur_a_left = a_left_db.next()
                pl.move(cur_a_left, a_l1_tile)
                cur_b_right = b_right_db.next()
                pl.move(cur_b_right, b_l1_tile)

                acc_tile = acc_db.next()
                pl.matmul(acc_tile, cur_a_left, cur_b_right)

                pl.store_tile(out, acc_tile, [i, j])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_matmul_kernel():
    _require_a5()
    device = ST_DEVICE
    torch.manual_seed(0)
    m, k, n = 256, 128, 256

    a = torch.randn(m, k, device=device, dtype=torch.float16)
    b = torch.randn(k, n, device=device, dtype=torch.float16)
    out = torch.zeros(m, n, device=device, dtype=torch.float32)

    matmul_example[None, 32](a, b, out)
    torch.npu.synchronize()

    golden = torch.matmul(a.float(), b.float())
    torch.testing.assert_close(out, golden, rtol=1e-2, atol=1e-2)


TILE_M = 64
TILE_N = 64
VEC_ROWS = 32
K_SIZE = 128
VF_LANES = 64
NEG_INF = -1e30


# =============================================================================
# CV 融合算子快速入门 —— make_tile_group + auto_mutex, fused Matmul+Softmax
#   A[M,128] @ B[128,N] → QK[M,N]，1<=M<=64，N 轴分块
#   Cube 保存 QK，Vector 段按 [M半块, N块] 视图跨全部 N 块完成 Softmax
# =============================================================================
@pl.jit(auto_mutex=True)
def matmul_softmax_kernel(
    a: pl.Tensor[[pl.DYNAMIC, K_SIZE], pl.DT_FP16],
    b: pl.Tensor[[K_SIZE, pl.DYNAMIC], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    workspace: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    m = a.shape[0]
    n = b.shape[1]
    n_tiles = (n + TILE_N - 1) // TILE_N
    valid_m = pl.min(TILE_M, m)

    # ---- Cube Tile：直接计算 QK = A @ B ----
    tt_a_mat = pl.TileType(
        shape=[TILE_M, K_SIZE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1]
    )
    tt_b_mat = pl.TileType(
        shape=[K_SIZE, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1]
    )
    tt_left = pl.TileType(
        shape=[TILE_M, K_SIZE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, valid_shape=[-1, -1], compact=1
    )
    tt_right = pl.TileType(
        shape=[K_SIZE, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right, valid_shape=[-1, -1], compact=1
    )
    tt_acc = pl.TileType(
        shape=[TILE_M, TILE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, valid_shape=[-1, -1], compact=1
    )

    a_l1 = pl.make_tile_group(type=tt_a_mat, addrs=0x0000, mutex_ids=[0])
    b_l1 = pl.make_tile_group(type=tt_b_mat, addrs=0x4000, mutex_ids=[1])
    a_l0a = pl.make_tile_group(type=tt_left, addrs=0x0000, mutex_ids=[2])
    b_l0b = pl.make_tile_group(type=tt_right, addrs=0x0000, mutex_ids=[3])
    qk_l0c = pl.make_tile_group(type=tt_acc, addrs=0x0000, mutex_ids=[4])

    # ---- Vector Tile：每个 AIV 处理最多 32 个 M 行，沿 N 块分三遍完成 Softmax ----
    tt_vec = pl.TileType(
        shape=[VEC_ROWS, TILE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]
    )
    tt_red = pl.TileType(
        shape=[VEC_ROWS, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.DN, valid_shape=[-1, -1]
    )
    # 行归约使用 [M半块, 1] 的 DN 视图，逐元素合并使用同一内存的行主序视图。
    tt_red_rm = pl.TileType(
        shape=[1, VEC_ROWS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]
    )

    qk_vec = pl.make_tile_group(type=tt_vec, addrs=0x0000, mutex_ids=[5])
    tmp_vec = pl.make_tile_group(type=tt_vec, addrs=0x2000, mutex_ids=[6])
    exp_vec = pl.make_tile_group(type=tt_vec, addrs=0x4000, mutex_ids=[7])
    red_vec = pl.make_tile_group(type=tt_red, addrs=0x6000, mutex_ids=[8])
    red_rm = pl.make_tile_group(type=tt_red_rm, addrs=0x6000, mutex_ids=[9])
    global_max = pl.make_tile_group(type=tt_red, addrs=0x6100, mutex_ids=[10])
    global_max_rm = pl.make_tile_group(type=tt_red_rm, addrs=0x6100, mutex_ids=[11])
    global_sum = pl.make_tile_group(type=tt_red, addrs=0x6200, mutex_ids=[12])
    global_sum_rm = pl.make_tile_group(type=tt_red_rm, addrs=0x6200, mutex_ids=[13])

    # ==== Cube: workspace = QK，N 轴分块 ====
    with pl.section_cube():
        cur_a_l1 = a_l1.current()
        cur_b_l1 = b_l1.current()
        cur_a_l0a = a_l0a.current()
        cur_b_l0b = b_l0b.current()
        cur_qk_l0c = qk_l0c.current()
        for nj in pl.range(0, n_tiles, 1):
            n_off = nj * TILE_N
            valid_n = pl.min(TILE_N, n - n_off)
            pl.set_validshape(cur_qk_l0c, [valid_m, valid_n])
            pl.set_validshape(cur_a_l1, [valid_m, K_SIZE])
            pl.set_validshape(cur_b_l1, [K_SIZE, valid_n])
            pl.load(cur_a_l1, a, [0, 0])
            pl.load(cur_b_l1, b, [0, n_off])
            pl.set_validshape(cur_a_l0a, [valid_m, K_SIZE])
            pl.set_validshape(cur_b_l0b, [K_SIZE, valid_n])
            pl.move(cur_a_l0a, cur_a_l1)
            pl.move(cur_b_l0b, cur_b_l1)
            pl.matmul(cur_qk_l0c, cur_a_l0a, cur_b_l0b)
            pl.store(workspace, cur_qk_l0c, [0, n_off])
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0)

    # ==== Vector: 在 QK 上沿 N 方向分块计算 Softmax ====
    with pl.section_vector():
        sub_id = pl.get_subblock_idx()
        pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=0)
        m_off = sub_id * VEC_ROWS
        if m_off < m:
            valid_rows = pl.min(VEC_ROWS, m - m_off)
            cur_qk = qk_vec.current()
            cur_tmp = tmp_vec.current()
            cur_exp = exp_vec.current()
            cur_red = red_vec.current()
            cur_red_rm = red_rm.current()
            cur_global_max = global_max.current()
            cur_global_max_rm = global_max_rm.current()
            cur_global_sum = global_sum.current()
            cur_global_sum_rm = global_sum_rm.current()
            pl.set_validshape(cur_red, [valid_rows, 1])
            pl.set_validshape(cur_red_rm, [1, valid_rows])
            pl.set_validshape(cur_global_max, [valid_rows, 1])
            pl.set_validshape(cur_global_max_rm, [1, valid_rows])
            pl.set_validshape(cur_global_sum, [valid_rows, 1])
            pl.set_validshape(cur_global_sum_rm, [1, valid_rows])

            # 第一遍：合并所有 N 块的行最大值。
            for nj in pl.range(0, n_tiles, 1):
                n_off = nj * TILE_N
                valid_n = pl.min(TILE_N, n - n_off)
                pl.set_validshape(cur_qk, [valid_rows, valid_n])
                pl.set_validshape(cur_tmp, [valid_rows, valid_n])
                pl.load(cur_qk, workspace, [m_off, n_off])
                pl.maximum(cur_red, cur_qk, cur_tmp, dim=0)
                if nj == 0:
                    pl.mul(cur_global_max_rm, cur_red_rm, 1.0)
                else:
                    pl.maximum(cur_global_max_rm, cur_global_max_rm, cur_red_rm)

            # 第二遍：基于全局最大值累加所有 N 块的 exp 和。
            for nj in pl.range(0, n_tiles, 1):
                n_off = nj * TILE_N
                valid_n = pl.min(TILE_N, n - n_off)
                pl.set_validshape(cur_qk, [valid_rows, valid_n])
                pl.set_validshape(cur_tmp, [valid_rows, valid_n])
                pl.set_validshape(cur_exp, [valid_rows, valid_n])
                pl.load(cur_qk, workspace, [m_off, n_off])
                pl.expand_sub(cur_exp, cur_qk, cur_global_max, dim=0)
                pl.exp(cur_exp, cur_exp)
                pl.sum(cur_red, cur_exp, cur_tmp, dim=0)
                if nj == 0:
                    pl.mul(cur_global_sum_rm, cur_red_rm, 1.0)
                else:
                    pl.add(cur_global_sum_rm, cur_global_sum_rm, cur_red_rm)

            # 第三遍：逐块归一化并直接写回 out。
            for nj in pl.range(0, n_tiles, 1):
                n_off = nj * TILE_N
                valid_n = pl.min(TILE_N, n - n_off)
                pl.set_validshape(cur_qk, [valid_rows, valid_n])
                pl.set_validshape(cur_exp, [valid_rows, valid_n])
                pl.load(cur_qk, workspace, [m_off, n_off])
                pl.expand_sub(cur_exp, cur_qk, cur_global_max, dim=0)
                pl.exp(cur_exp, cur_exp)
                pl.expand_div(cur_exp, cur_exp, cur_global_sum, dim=0)
                pl.store(out, cur_exp, [m_off, n_off])


@pytest.mark.parametrize(
    "m,n",
    [
        (32, 64),  # 仅 subblock 0，完整 N 块
        (33, 64),  # subblock 1 仅处理一行
        (64, 64),  # 完整 m/n 块
        (48, 48),  # m/n 均为尾块
        (48, 1000),  # 多个 n 块，末块不对齐
        (48, 1024),  # 文档调用示例
        (16, 4096),  # 4K n 轴
    ],
)
@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_matmul_softmax_kernel(m, n):
    _require_a5()
    device = ST_DEVICE
    torch.manual_seed(0)
    a = torch.randn(m, K_SIZE, device=device, dtype=torch.float16) * 0.1
    b = torch.randn(K_SIZE, n, device=device, dtype=torch.float16) * 0.1
    out = torch.zeros(m, n, device=device, dtype=torch.float32)
    workspace = torch.zeros(m, n, device=device, dtype=torch.float32)

    matmul_softmax_kernel(a, b, out, workspace)
    torch.npu.synchronize()

    matmul_golden = torch.matmul(a.float(), b.float())
    torch.testing.assert_close(workspace, matmul_golden, rtol=1e-2, atol=1e-2)
    golden = torch.softmax(matmul_golden, dim=-1)
    torch.testing.assert_close(out, golden, rtol=1e-2, atol=1e-2)


# =============================================================================
# CV 融合算子快速入门 VF 版本
#   workspace 仍为 [M,N]；Vector 段在 UB 中转为 [N块,64] 后调用 VF
# =============================================================================
@pl.vector_function
def softmax_vf_init(global_max, global_sum, valid_rows: pl.DT_INT64):
    """初始化跨 N 块保存的逐行最大值与指数和。"""
    preg = vf.update_mask(valid_rows, dtype=pl.DT_FP32)
    max_reg = vf.full(NEG_INF, preg, dtype=pl.DT_FP32)
    sum_reg = vf.full(0.0, preg, dtype=pl.DT_FP32)
    vf.store_align(global_max, max_reg, preg)
    vf.store_align(global_sum, sum_reg, preg)


@pl.vector_function
def softmax_vf_update_max(src_dn, global_max, valid_rows: pl.DT_INT64, valid_n: pl.DT_INT64):
    """把当前 N 块合并到逐行全局最大值。"""
    preg = vf.update_mask(valid_rows, dtype=pl.DT_FP32)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
    max_reg = vf.load_align(global_max, 0)
    for ni in pl.range(0, valid_n):
        src_reg = vf.load_align(src_dn, ni * VF_LANES)
        max_reg = vf.max(max_reg, src_reg, preg)
    vf.store_align(global_max, max_reg, preg)


@pl.vector_function
def softmax_vf_update_sum(src_dn, global_max, global_sum, valid_rows: pl.DT_INT64, valid_n: pl.DT_INT64):
    """基于全局最大值累加当前 N 块的指数和。"""
    preg = vf.update_mask(valid_rows, dtype=pl.DT_FP32)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
    max_reg = vf.load_align(global_max, 0)
    sum_reg = vf.load_align(global_sum, 0)
    for ni in pl.range(0, valid_n):
        src_reg = vf.load_align(src_dn, ni * VF_LANES)
        exp_reg = vf.exp_sub(src_reg, max_reg, preg)
        sum_reg = vf.add(sum_reg, exp_reg, preg)
    vf.store_align(global_sum, sum_reg, preg)


@pl.vector_function
def softmax_vf_normalize(src_dn, dst_dn, global_max, global_sum, valid_rows: pl.DT_INT64, valid_n: pl.DT_INT64):
    """使用完整 N 轴的最大值与指数和归一化当前 N 块。"""
    preg = vf.update_mask(valid_rows, dtype=pl.DT_FP32)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
    max_reg = vf.load_align(global_max, 0)
    sum_reg = vf.load_align(global_sum, 0)
    for ni in pl.range(0, valid_n):
        src_reg = vf.load_align(src_dn, ni * VF_LANES)
        exp_reg = vf.exp_sub(src_reg, max_reg, preg)
        out_reg = vf.div(exp_reg, sum_reg, preg)
        vf.store_align(dst_dn + ni * VF_LANES, out_reg, preg)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)


@pl.jit(auto_mutex=True)
def matmul_softmax_vf_kernel(
    a: pl.Tensor[[pl.DYNAMIC, K_SIZE], pl.DT_FP16],
    b: pl.Tensor[[K_SIZE, pl.DYNAMIC], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    workspace: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    m = a.shape[0]
    n = b.shape[1]
    n_tiles = (n + TILE_N - 1) // TILE_N
    valid_m = pl.min(TILE_M, m)

    # ---- Cube Tile：直接计算 QK = A @ B ----
    tt_a_mat = pl.TileType(
        shape=[TILE_M, K_SIZE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1]
    )
    tt_b_mat = pl.TileType(
        shape=[K_SIZE, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1]
    )
    tt_left = pl.TileType(
        shape=[TILE_M, K_SIZE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, valid_shape=[-1, -1], compact=1
    )
    tt_right = pl.TileType(
        shape=[K_SIZE, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right, valid_shape=[-1, -1], compact=1
    )
    tt_acc = pl.TileType(
        shape=[TILE_M, TILE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, valid_shape=[-1, -1], compact=1
    )

    a_l1 = pl.make_tile_group(type=tt_a_mat, addrs=0x0000, mutex_ids=[0])
    b_l1 = pl.make_tile_group(type=tt_b_mat, addrs=0x4000, mutex_ids=[1])
    a_l0a = pl.make_tile_group(type=tt_left, addrs=0x0000, mutex_ids=[2])
    b_l0b = pl.make_tile_group(type=tt_right, addrs=0x0000, mutex_ids=[3])
    qk_l0c = pl.make_tile_group(type=tt_acc, addrs=0x0000, mutex_ids=[4])

    # ---- Vector Tile：TTRANS 前后都使用 64x64 物理 Tile，VF 用 mask 收窄 M 尾块 ----
    tt_vf_block = pl.TileType(
        shape=[VF_LANES, TILE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]
    )
    tt_vf_state = pl.TileType(
        shape=[1, VF_LANES], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]
    )

    qk_nd = pl.make_tile_group(type=tt_vf_block, addrs=0x0000, mutex_ids=[5])
    qk_dn = pl.make_tile_group(type=tt_vf_block, addrs=0x4000, mutex_ids=[6])
    out_dn = pl.make_tile_group(type=tt_vf_block, addrs=0x8000, mutex_ids=[7])
    out_nd = pl.make_tile_group(type=tt_vf_block, addrs=0xC000, mutex_ids=[8])
    global_max = pl.make_tile_group(type=tt_vf_state, addrs=0x10000, mutex_ids=[9])
    global_sum = pl.make_tile_group(type=tt_vf_state, addrs=0x10100, mutex_ids=[10])

    # ==== Cube: workspace = QK，N 轴分块 ====
    with pl.section_cube():
        cur_a_l1 = a_l1.current()
        cur_b_l1 = b_l1.current()
        cur_a_l0a = a_l0a.current()
        cur_b_l0b = b_l0b.current()
        cur_qk_l0c = qk_l0c.current()
        for nj in pl.range(0, n_tiles, 1):
            n_off = nj * TILE_N
            valid_n = pl.min(TILE_N, n - n_off)
            pl.set_validshape(cur_qk_l0c, [valid_m, valid_n])
            pl.set_validshape(cur_a_l1, [valid_m, K_SIZE])
            pl.set_validshape(cur_b_l1, [K_SIZE, valid_n])
            pl.load(cur_a_l1, a, [0, 0])
            pl.load(cur_b_l1, b, [0, n_off])
            pl.set_validshape(cur_a_l0a, [valid_m, K_SIZE])
            pl.set_validshape(cur_b_l0b, [K_SIZE, valid_n])
            pl.move(cur_a_l0a, cur_a_l1)
            pl.move(cur_b_l0b, cur_b_l1)
            pl.matmul(cur_qk_l0c, cur_a_l0a, cur_b_l0b)
            pl.store(workspace, cur_qk_l0c, [0, n_off])
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0)

    # ==== Vector: 在转置后的 [N块,64] UB Tile 上调用 VF ====
    with pl.section_vector():
        sub_id = pl.get_subblock_idx()
        pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=0)
        m_off = sub_id * VEC_ROWS
        if m_off < m:
            valid_rows = pl.min(VEC_ROWS, m - m_off)
            cur_qk_nd = qk_nd.current()
            cur_qk_dn = qk_dn.current()
            cur_out_dn = out_dn.current()
            cur_out_nd = out_nd.current()
            cur_global_max = global_max.current()
            cur_global_sum = global_sum.current()
            pl.set_validshape(cur_global_max, [1, valid_rows])
            pl.set_validshape(cur_global_sum, [1, valid_rows])
            softmax_vf_init(cur_global_max, cur_global_sum, valid_rows)

            # 第一遍：把各 N 块转为 [N块,64]，逐 lane 更新每行最大值。
            for nj in pl.range(0, n_tiles, 1):
                n_off = nj * TILE_N
                valid_n = pl.min(TILE_N, n - n_off)
                pl.set_validshape(cur_qk_nd, [valid_rows, valid_n])
                pl.set_validshape(cur_qk_dn, [valid_n, valid_rows])
                pl.load(cur_qk_nd, workspace, [m_off, n_off])
                pl.transpose(cur_qk_dn, cur_qk_nd)
                softmax_vf_update_max(cur_qk_dn, cur_global_max, valid_rows, valid_n)

            # 第二遍：基于全局最大值逐 lane 累加完整 N 轴的指数和。
            for nj in pl.range(0, n_tiles, 1):
                n_off = nj * TILE_N
                valid_n = pl.min(TILE_N, n - n_off)
                pl.set_validshape(cur_qk_nd, [valid_rows, valid_n])
                pl.set_validshape(cur_qk_dn, [valid_n, valid_rows])
                pl.load(cur_qk_nd, workspace, [m_off, n_off])
                pl.transpose(cur_qk_dn, cur_qk_nd)
                softmax_vf_update_sum(cur_qk_dn, cur_global_max, cur_global_sum, valid_rows, valid_n)

            # 第三遍：VF 归一化后转回 [M半块,N块] 并写回 GM。
            for nj in pl.range(0, n_tiles, 1):
                n_off = nj * TILE_N
                valid_n = pl.min(TILE_N, n - n_off)
                pl.set_validshape(cur_qk_nd, [valid_rows, valid_n])
                pl.set_validshape(cur_qk_dn, [valid_n, valid_rows])
                pl.set_validshape(cur_out_dn, [valid_n, valid_rows])
                pl.set_validshape(cur_out_nd, [valid_rows, valid_n])
                pl.load(cur_qk_nd, workspace, [m_off, n_off])
                pl.transpose(cur_qk_dn, cur_qk_nd)
                softmax_vf_normalize(cur_qk_dn, cur_out_dn, cur_global_max, cur_global_sum, valid_rows, valid_n)
                pl.transpose(cur_out_nd, cur_out_dn)
                pl.store(out, cur_out_nd, [m_off, n_off])


@pytest.mark.parametrize(
    "m,n",
    [
        (32, 64),  # 仅 subblock 0，完整 N 块
        (33, 64),  # subblock 1 仅处理一行
        (64, 64),  # 完整 M/N 块
        (48, 48),  # M/N 均为尾块
        (48, 1000),  # 多个 N 块，末块不对齐
        (48, 1024),  # 文档调用示例
        (16, 4096),  # 4K N 轴
    ],
)
@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_matmul_softmax_vf_kernel(m, n):
    _require_a5()
    device = ST_DEVICE
    torch.manual_seed(0)
    a = torch.randn(m, K_SIZE, device=device, dtype=torch.float16) * 0.1
    b = torch.randn(K_SIZE, n, device=device, dtype=torch.float16) * 0.1
    out = torch.zeros(m, n, device=device, dtype=torch.float32)
    workspace = torch.zeros(m, n, device=device, dtype=torch.float32)

    matmul_softmax_vf_kernel(a, b, out, workspace)
    torch.npu.synchronize()

    matmul_golden = torch.matmul(a.float(), b.float())
    torch.testing.assert_close(workspace, matmul_golden, rtol=1e-2, atol=1e-2)
    golden = torch.softmax(matmul_golden, dim=-1)
    torch.testing.assert_close(out, golden, rtol=1e-2, atol=1e-2)
