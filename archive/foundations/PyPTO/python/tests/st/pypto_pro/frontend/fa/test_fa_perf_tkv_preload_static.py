#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""FlashAttention performance kernel using PyPTO IR manual (non-SSA) mode.

Double-buffered cross-core communication + QK pre-compute pattern.
Reference: fa_performance_kernel.cpp

Features:
  1. Multi-core: each Cube core processes multiple Q tiles via strided loop
  2. Double buffer: L1 ping/pong for K/P/V MAT tiles
  3. FIFO cross-core GM buffers (qk_buf, p_buf) with configurable depth
  4. Cross-core event ID ping/pong (QK_READY_0/1, P_READY_0/1, PV_READY_0/1)
  5. QK pre-compute: Cube runs QK_PRELOAD tiles ahead, then QK[i+preload] + PV[i]
  6. Vector: FIFO exp_corr (by task_id % FIFO_SIZE),
            double-buffered global_max/global_sum (by q_count % 2)

Usage:
    python3 tests/ut/frontend/flash_attention/test_fa_performance.py
"""

import logging
import math
import os

import pypto_pro.language as pl
import pytest
import torch

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"


# ================================================================
#  Configuration -change QK_PRELOAD to tune pre-compute depth
# ================================================================
QK_PRELOAD = 1  # How many KV tiles to pre-compute QK ahead
# (configurable, max ~6; requires skv_tiles >= QK_PRELOAD)
FIFO_SIZE = QK_PRELOAD + 1  # Exp-corr FIFO depth (avoids read/write collision)

# ================================================================
#  Tile dimensions and constants
# ================================================================
TS = 128
TKV = 128
TD = 128
TS_HALF = TS // 2
SCALE = 1.0 / math.sqrt(TD)

# Cube tiles use TS rows
Q_F16 = TS * TD * 2
KT_F16 = TD * TKV * 2
V_F16 = TKV * TD * 2
P_F16 = TS * TKV * 2
QK_HALF_F32 = TS * TKV * 4
PV_HALF_F32 = TS * TD * 4

# ---- MAT (512KB) ----
MA0 = 0
MA0_PONG = MA0 + Q_F16
MA1 = Q_F16 * 2
MA1_PONG = MA1 + KT_F16
MA2 = MA1 + KT_F16 * 2
MA2_PONG = MA2 + P_F16
MA3 = MA2 + P_F16 * 2
MA3_PONG = MA3 + V_F16
LA0 = 0
LA1 = P_F16
RA0 = 0
RA1 = KT_F16
CA0 = 0
CA1 = QK_HALF_F32

# ---- VEC addresses (248KB on a5) ----
VB4_KV = TS_HALF * TKV * 4
VB2_KV = TS_HALF * TKV * 2
VB4 = TS_HALF * TD * 4
VB2 = TS_HALF * TD * 2
VB6 = (TS_HALF + 1) * TD * 2
VB_RED = TS_HALF * 1 * 4  # 256B -[64,1] FP32

VA0 = 0  # qk_vec
VA1 = VA0 + VB4_KV  # tmp_vec
VA2 = VA1 + VB4_KV  # p_f16
VA3 = VA2 + VB2_KV  # reduce_dst
# global_max x 2 (by q_count % 2)
VA_GMAX0 = VA3 + VB_RED
VA_GMAX1 = VA_GMAX0 + VB_RED
# global_sum x 2 (by q_count % 2)
VA_GSUM0 = VA_GMAX1 + VB_RED
VA_GSUM1 = VA_GSUM0 + VB_RED
# exp_corr x FIFO_SIZE (by task_id % FIFO_SIZE)
VA_EXP_BASE = VA_GSUM1 + VB_RED
VA_EXP0 = VA_EXP_BASE
VA_EXP1 = VA_EXP_BASE + VB_RED
VA_AFTER_EXP = VA_EXP_BASE + FIFO_SIZE * VB_RED
VA7 = VA_AFTER_EXP  # running_o
VA8 = VA7 + VB4  # pv_vec
VA9 = VA8 + VB4  # o_f16
VA10 = VA9 + VB2
VA11 = VA10 + VB6  # qk_vec1: after tile_nz
VA12 = VA11 + VB4_KV  # pv_vec1: after qk_vec1 (no overlap)
assert VA12 + VB4 <= 248 * 1024, f"VEC overflow: {VA12 + VB4} > {248 * 1024}"

event_ids_01 = (0, 1)
event_ids_23 = (2, 3)

# Cross-core event IDs (0-15 available on Ascend NPU)
# QK: FIFO_SIZE IDs, P: FIFO_SIZE IDs, PV: FIFO_SIZE IDs
QK_READY_IDS = tuple(range(0, FIFO_SIZE))
P_READY_IDS = tuple(range(FIFO_SIZE, 2 * FIFO_SIZE))
PV_READY_IDS = tuple(range(2 * FIFO_SIZE, 3 * FIFO_SIZE))
assert 3 * FIFO_SIZE <= 16, f"Too many cross-core event IDs: need {3 * FIFO_SIZE}, max 16"

# PV buffer: 2 Q-slots x FIFO_SIZE task-slots per core
PV_CORE_STRIDE = 2 * FIFO_SIZE * TS


# ================================================================
#  Tile allocation helpers
# ================================================================
def alloc_cube_tiles(p_mat_buf):
    q_mat_type = pl.TileType(shape=[TS, TD], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ)
    q_mat_0 = pl.make_tile(q_mat_type, addr=MA0, size=Q_F16)
    q_mat_1 = pl.make_tile(q_mat_type, addr=MA0_PONG, size=Q_F16)
    k_mat_type = pl.TileType(shape=[TD, TKV], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN)
    k_mat_0 = pl.make_tile(k_mat_type, addr=MA1, size=KT_F16)
    k_mat_1 = pl.make_tile(k_mat_type, addr=MA1_PONG, size=KT_F16)
    v_mat_type = pl.TileType(shape=[TKV, TD], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ)
    v_mat_0 = pl.make_tile(v_mat_type, addr=MA3, size=V_F16)
    v_mat_1 = pl.make_tile(v_mat_type, addr=MA3_PONG, size=V_F16)
    left_0 = pl.make_tile(
        pl.TileType(shape=[TS, TD], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addr=LA0,
        size=Q_F16,
    )
    left_1 = pl.make_tile(
        pl.TileType(shape=[TS, TD], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addr=LA1,
        size=Q_F16,
    )
    right_0 = pl.make_tile(
        pl.TileType(shape=[TKV, TD], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addr=RA0,
        size=KT_F16,
    )
    right_1 = pl.make_tile(
        pl.TileType(shape=[TKV, TD], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addr=RA1,
        size=KT_F16,
    )
    acc_0 = pl.make_tile(
        pl.TileType(shape=[TS, TKV], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024),
        addr=CA0,
        size=QK_HALF_F32,
    )
    acc_1 = pl.make_tile(
        pl.TileType(shape=[TS, TKV], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024),
        addr=CA1,
        size=PV_HALF_F32,
    )
    return pl.make_tuple(
        q_mat_buf=(q_mat_0, q_mat_1),
        k_mat_buf=(k_mat_0, k_mat_1),
        p_mat_buf=p_mat_buf,
        v_mat_buf=(v_mat_0, v_mat_1),
        left_buf=(left_0, left_1),
        right_buf=(right_0, right_1),
        acc_buf=(acc_0, acc_1),
    )


# ================================================================
# Cube section - compute_qk, compute_pv
# ================================================================
def compute_qk(
    task_id: pl.DT_INT64,
    ki: pl.DT_INT64,
    q_count: pl.DT_INT64,
    sq_off: pl.DT_INT64,
    skv_tiles: pl.DT_INT64,
    buf_idx: pl.DT_INT64,
    l0ab_idx: pl.DT_INT64,
    l0c_idx: pl.DT_INT64,
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    cube,
    qk_vec_buf,
) -> None:
    """QK = Q * K^T -> ACC->Vec via dual_mode_split_m. Caller toggles l0ab/l0c."""
    q_mat_buf = cube.q_mat_buf
    k_mat_buf = cube.k_mat_buf
    left_buf = cube.left_buf
    right_buf = cube.right_buf
    acc_buf = cube.acc_buf
    qk_fifo_slot = task_id % FIFO_SIZE
    skv_off = ki * TKV

    pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=event_ids_01[buf_idx])
    if ki == 0:
        pl.load(q_mat_buf[q_count % 2], q, [sq_off, 0])
    pl.load(k_mat_buf[buf_idx], k, [skv_off, 0], order=[1, 0])
    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)

    pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=event_ids_01[l0ab_idx])
    pl.move(left_buf[l0ab_idx], q_mat_buf[q_count % 2])
    pl.move(right_buf[l0ab_idx], k_mat_buf[buf_idx])
    pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
    pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
    pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=event_ids_01[buf_idx])

    pl.system.sync_dst(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=event_ids_01[l0c_idx])
    pl.matmul(acc_buf[l0c_idx], left_buf[l0ab_idx], right_buf[l0ab_idx])
    pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
    pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
    pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=event_ids_01[l0ab_idx])

    if qk_fifo_slot == 0:
        if l0c_idx == 0:
            pl.move(qk_vec_buf[0], acc_buf[0], acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
        else:
            pl.move(qk_vec_buf[0], acc_buf[1], acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
    else:
        if l0c_idx == 0:
            pl.move(qk_vec_buf[1], acc_buf[0], acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
        else:
            pl.move(qk_vec_buf[1], acc_buf[1], acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)

    pl.system.sync_src(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=event_ids_01[l0c_idx])
    pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=QK_READY_IDS[qk_fifo_slot])
    return


def compute_pv(
    task_id: pl.DT_INT64,
    ki: pl.DT_INT64,
    q_count: pl.DT_INT64,
    buf_idx: pl.DT_INT64,
    l0ab_idx: pl.DT_INT64,
    l0c_idx: pl.DT_INT64,
    v: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    cube,
    pv_vec_buf,
) -> None:
    """PV = P * V -> ACC->Vec via dual_mode_split_m. Caller toggles l0ab/l0c."""
    v_mat_buf = cube.v_mat_buf
    p_mat_buf = cube.p_mat_buf
    left_buf = cube.left_buf
    right_buf = cube.right_buf
    acc_buf = cube.acc_buf
    pv_task_slot = task_id % FIFO_SIZE
    sv_off = ki * TKV
    pv_fifo_slot = task_id % FIFO_SIZE

    pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=event_ids_23[buf_idx])
    pl.load(v_mat_buf[buf_idx], v, [sv_off, 0])
    pl.system.wait_cross_core(pipe=pl.PipeType.MTE1, event_id=P_READY_IDS[pv_fifo_slot])

    pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=event_ids_01[l0ab_idx])
    pl.move(left_buf[l0ab_idx], p_mat_buf[buf_idx])
    pl.move(right_buf[l0ab_idx], v_mat_buf[buf_idx])
    pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=event_ids_23[buf_idx])
    pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
    pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)

    pl.system.sync_dst(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=event_ids_01[l0c_idx])
    pl.matmul(acc_buf[l0c_idx], left_buf[l0ab_idx], right_buf[l0ab_idx])
    pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
    pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
    pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=event_ids_01[l0ab_idx])

    if pv_task_slot == 0:
        if l0c_idx == 0:
            pl.move(pv_vec_buf[0], acc_buf[0], acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
        else:
            pl.move(pv_vec_buf[0], acc_buf[1], acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
    else:
        if l0c_idx == 0:
            pl.move(pv_vec_buf[1], acc_buf[0], acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
        else:
            pl.move(pv_vec_buf[1], acc_buf[1], acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)

    pl.system.sync_src(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=event_ids_01[l0c_idx])
    pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=PV_READY_IDS[pv_task_slot])
    return


# ================================================================
# Vector section - softmax_body, compute_p, compute_gu
# ================================================================
def softmax_body(
    task_id: pl.DT_INT64,
    ki: pl.DT_INT64,
    q_count: pl.DT_INT64,
    buf_idx: pl.DT_INT64,
    stiles,
) -> None:
    """Softmax body (no cross-core sync, no store). Uses static independent vars.
    On a5 with 2 sub-blocks, each sub-block handles half the 64-row tile.
    Both sub-blocks share row_off, so they jointly cover rows [row_off, row_off+64).
    """
    qk_vec_buf = stiles.qk_vec_buf
    tmp_vec = stiles.tmp_vec
    p_f16 = stiles.p_f16
    reduce_dst = stiles.reduce_dst
    reduce_dst_rm = stiles.reduce_dst_rm
    global_max_rm_buf = stiles.global_max_rm_buf
    global_sum_rm_buf = stiles.global_sum_rm_buf
    exp_corr_rm_fifo = stiles.exp_corr_rm_fifo
    tile_nz = stiles.tile_nz
    p_mat_buf = stiles.p_mat_buf
    p_fifo_slot = task_id % FIFO_SIZE
    q_idx = q_count % 2
    global_max_rm_cur = global_max_rm_buf[q_idx]
    global_sum_rm_cur = global_sum_rm_buf[q_idx]
    sub_id = pl.get_subblock_idx()

    if ki == 0:
        pl.maximum(reduce_dst, qk_vec_buf[p_fifo_slot], tmp_vec, dim=0)
        pl.expand_sub(tmp_vec, qk_vec_buf[p_fifo_slot], reduce_dst)
        pl.mul(global_max_rm_cur, reduce_dst_rm, 1.0)
        pl.mul(tmp_vec, tmp_vec, SCALE)
        pl.exp(qk_vec_buf[p_fifo_slot], tmp_vec)
        pl.sum(reduce_dst, qk_vec_buf[p_fifo_slot], tmp_vec)
        pl.mul(global_sum_rm_cur, reduce_dst_rm, 1.0)
        pl.cast(p_f16, qk_vec_buf[p_fifo_slot], mode=pl.RoundMode.CAST_ROUND)
    if ki > 0:
        pl.maximum(reduce_dst, qk_vec_buf[p_fifo_slot], tmp_vec, dim=0)
        pl.maximum(reduce_dst_rm, reduce_dst_rm, global_max_rm_cur)
        pl.sub(exp_corr_rm_fifo[p_fifo_slot], global_max_rm_cur, reduce_dst_rm)
        pl.mul(global_max_rm_cur, reduce_dst_rm, 1.0)
        pl.expand_sub(tmp_vec, qk_vec_buf[p_fifo_slot], reduce_dst)
        pl.mul(exp_corr_rm_fifo[p_fifo_slot], exp_corr_rm_fifo[p_fifo_slot], SCALE)
        pl.mul(tmp_vec, tmp_vec, SCALE)
        pl.exp(exp_corr_rm_fifo[p_fifo_slot], exp_corr_rm_fifo[p_fifo_slot])
        pl.exp(qk_vec_buf[p_fifo_slot], tmp_vec)
        pl.cast(p_f16, qk_vec_buf[p_fifo_slot], mode=pl.RoundMode.CAST_ROUND)
        pl.mul(global_sum_rm_cur, global_sum_rm_cur, exp_corr_rm_fifo[p_fifo_slot])
        pl.sum(reduce_dst, qk_vec_buf[p_fifo_slot], tmp_vec)
        pl.add(global_sum_rm_cur, global_sum_rm_cur, reduce_dst_rm)
    pl.move(tile_nz, p_f16)
    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
    pl.insert(p_mat_buf[buf_idx], tile_nz, [64 * sub_id, 0])
    return


def compute_p(
    task_id: pl.DT_INT64,
    ki: pl.DT_INT64,
    q_count: pl.DT_INT64,
    buf_idx: pl.DT_INT64,
    stiles,
) -> None:
    """Softmax on QK tile -> P. Includes cross-core sync."""
    p_fifo_slot = task_id % FIFO_SIZE
    pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=QK_READY_IDS[p_fifo_slot])
    softmax_body(task_id, ki, q_count, buf_idx, stiles)
    pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=P_READY_IDS[p_fifo_slot])
    return


def compute_gu(
    task_id: pl.DT_INT64,
    ki: pl.DT_INT64,
    skv_tiles: pl.DT_INT64,
    q_count: pl.DT_INT64,
    sq_off: pl.DT_INT64,
    row_off: pl.DT_INT64,
    o: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    gtiles,
) -> None:
    """GU: running output update. Includes cross-core sync."""
    pv_vec_buf = gtiles.pv_vec_buf
    running_o = gtiles.running_o
    exp_corr_fifo = gtiles.exp_corr_fifo
    global_sum_buf = gtiles.global_sum_buf
    o_f16 = gtiles.o_f16
    pv_slot = task_id % FIFO_SIZE
    pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=PV_READY_IDS[pv_slot])
    if ki == 0:
        pl.move(running_o, pv_vec_buf[pv_slot])
    if ki > 0:
        pl.expand_mul(running_o, running_o, exp_corr_fifo[pv_slot])
        pl.add(running_o, running_o, pv_vec_buf[pv_slot])
    if ki == skv_tiles - 1:
        pl.expand_div(running_o, running_o, global_sum_buf[q_count % 2])
        pl.cast(o_f16, running_o, mode=pl.RoundMode.CAST_ROUND)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
        pl.store(o, o_f16, [sq_off + row_off, 0])
    return


# ================================================================
#  Kernel
# ================================================================
@pl.jit()
def fa_perf_tkv_preload_kernel(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    v: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    o: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    qk_buf: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],  # FIFO_SIZE x Sq rows
    p_buf: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],  # FIFO_SIZE x Sq rows
    pv_buf: pl.Tensor[[48 * PV_CORE_STRIDE, pl.DYNAMIC], pl.DT_FP32],  # double-buffered per core
):

    sq_dim = q.shape[0]
    skv_dim = k.shape[0]
    sq_tiles = (sq_dim + (TS - 1)) // TS
    skv_tiles = (skv_dim + (TKV - 1)) // TKV
    num_cores = pl.get_block_num()
    core_id = pl.get_block_idx() // pl.get_subblock_num()

    # Shared between cube and vector sections: p_mat double buffers
    p_mat_type = pl.TileType(shape=[TS, TKV], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ)
    p_mat_0 = pl.make_tile(p_mat_type, addr=MA2, size=P_F16)
    p_mat_1 = pl.make_tile(p_mat_type, addr=MA2_PONG, size=P_F16)
    p_mat_buf = (p_mat_0, p_mat_1)

    # Cross-section VEC tiles: written by Cube (acc→vec TMOV), read by Vector
    vec_type_kv = pl.TileType(shape=[TS_HALF, TKV], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    vec_type_d = pl.TileType(shape=[TS_HALF, TD], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    qk_vec = pl.make_tile(vec_type_kv, addr=VA0, size=VB4_KV)
    qk_vec1 = pl.make_tile(vec_type_kv, addr=VA11, size=VB4_KV)
    qk_vec_buf = (qk_vec, qk_vec1)
    pv_vec = pl.make_tile(vec_type_d, addr=VA8, size=VB4)
    pv_vec1 = pl.make_tile(vec_type_d, addr=VA12, size=VB4)
    pv_vec_buf = (pv_vec, pv_vec1)

    # =================== CUBE SECTION ===================
    with pl.section_cube():
        cube_tiles = alloc_cube_tiles(p_mat_buf)

        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=0)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=1)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=2)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=3)
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=1)
        pl.system.sync_src(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=0)
        pl.system.sync_src(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=1)

        task_id = 0
        q_count = 0
        l0ab_idx = 0
        l0c_idx = 0
        ctx_arr = pl.struct_array(3, "CubeCtx", sq_off=0, task_id=0, qi=0, ki=0, skv_tiles=0, q_count=0)
        for qi in pl.range(core_id, sq_tiles, num_cores):
            sq_off = qi * TS
            # ---- Main loop: QK[ki+preload] ahead + PV[ki] current ----
            for ki in pl.range(0, skv_tiles):
                ctx_curr = ctx_arr[task_id % 3]
                ctx_curr.sq_off = sq_off
                ctx_curr.task_id = task_id
                ctx_curr.qi = qi
                ctx_curr.ki = ki
                ctx_curr.skv_tiles = skv_tiles
                ctx_curr.q_count = q_count
                buf_idx = (q_count * skv_tiles + ki) % 2
                compute_qk(
                    task_id,
                    ki,
                    q_count,
                    sq_off,
                    skv_tiles,
                    buf_idx,
                    l0ab_idx,
                    l0c_idx,
                    q,
                    k,
                    cube_tiles,
                    qk_vec_buf,
                )
                l0ab_idx = 1 - l0ab_idx
                l0c_idx = 1 - l0c_idx

                ctx_pre = ctx_arr[(task_id + 2) % 3]
                if task_id > 0:
                    pre_buf_idx = (ctx_pre.q_count * ctx_pre.skv_tiles + ctx_pre.ki) % 2
                    compute_pv(
                        ctx_pre.task_id,
                        ctx_pre.ki,
                        ctx_pre.q_count,
                        pre_buf_idx,
                        l0ab_idx,
                        l0c_idx,
                        v,
                        cube_tiles,
                        pv_vec_buf,
                    )
                    l0ab_idx = 1 - l0ab_idx
                    l0c_idx = 1 - l0c_idx
                task_id = task_id + 1
            q_count = q_count + 1

        ctx_last = ctx_arr[(task_id + 2) % 3]
        last_buf_idx = (ctx_last.q_count * ctx_last.skv_tiles + ctx_last.ki) % 2
        compute_pv(
            ctx_last.task_id,
            ctx_last.ki,
            ctx_last.q_count,
            last_buf_idx,
            l0ab_idx,
            l0c_idx,
            v,
            cube_tiles,
            pv_vec_buf,
        )
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=2)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=3)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=1)

    # =================== VECTOR SECTION ===================
    with pl.section_vector():
        running_o = pl.make_tile(
            pl.TileType(shape=[TS_HALF, TD], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addr=VA7,
            size=VB4,
        )
        tmp_vec = pl.make_tile(
            pl.TileType(shape=[TS_HALF, TKV], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addr=VA1,
            size=VB4_KV,
        )
        p_f16 = pl.make_tile(
            pl.TileType(shape=[TS_HALF, TKV], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
            addr=VA2,
            size=VB2_KV,
        )
        reduce_dst = pl.make_tile(
            pl.TileType(shape=[TS_HALF, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.DN),
            addr=VA3,
            size=VB_RED,
        )
        reduce_dst_rm = pl.make_tile(
            pl.TileType(shape=[1, TS_HALF], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addr=VA3,
            size=VB_RED,
        )

        red_type = pl.TileType(shape=[TS_HALF, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.DN)
        red_rm_type = pl.TileType(shape=[1, TS_HALF], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)

        # Double-buffered global_max / global_sum (by q_count % 2)
        gmax_rm_0 = pl.make_tile(red_rm_type, addr=VA_GMAX0, size=VB_RED)
        gmax_rm_1 = pl.make_tile(red_rm_type, addr=VA_GMAX1, size=VB_RED)

        gsum_0 = pl.make_tile(red_type, addr=VA_GSUM0, size=VB_RED)
        gsum_1 = pl.make_tile(red_type, addr=VA_GSUM1, size=VB_RED)
        gsum_rm_0 = pl.make_tile(red_rm_type, addr=VA_GSUM0, size=VB_RED)
        gsum_rm_1 = pl.make_tile(red_rm_type, addr=VA_GSUM1, size=VB_RED)

        # FIFO exp_corr (by task_id % FIFO_SIZE)
        exp_corr_type = pl.TileType(
            shape=[TS_HALF, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.DN
        )
        exp_corr_rm_type = pl.TileType(shape=[1, TS_HALF], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        exp_corr_0 = pl.make_tile(exp_corr_type, addr=VA_EXP0, size=VB_RED)
        exp_corr_rm_0 = pl.make_tile(exp_corr_rm_type, addr=VA_EXP0, size=VB_RED)
        exp_corr_1 = pl.make_tile(exp_corr_type, addr=VA_EXP1, size=VB_RED)
        exp_corr_rm_1 = pl.make_tile(exp_corr_rm_type, addr=VA_EXP1, size=VB_RED)

        o_f16 = pl.make_tile(
            pl.TileType(shape=[TS_HALF, TD], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
            addr=VA9,
            size=VB2,
        )
        tile_type_nz = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec, layout=pl.NZ)
        tile_nz = pl.make_tile(tile_type_nz, addr=VA10, size=VB6)

        task_id = 0
        q_count = 0
        sub_id = pl.get_subblock_idx()
        ctx_arr = pl.struct_array(3, "VecCtx", sq_off=0, task_id=0, qi=0, ki=0, skv_tiles=0, q_count=0, core_id=0)

        row_off = sub_id * TS_HALF
        # Build tile groups OUTSIDE the loop to avoid TupleType return vars in for-loop phi nodes
        softmax_tiles = pl.make_tuple(
            qk_vec_buf=(qk_vec, qk_vec1),
            tmp_vec=tmp_vec,
            p_f16=p_f16,
            reduce_dst=reduce_dst,
            reduce_dst_rm=reduce_dst_rm,
            global_max_rm_buf=(gmax_rm_0, gmax_rm_1),
            global_sum_rm_buf=(gsum_rm_0, gsum_rm_1),
            exp_corr_rm_fifo=(exp_corr_rm_0, exp_corr_rm_1),
            tile_nz=tile_nz,
            p_mat_buf=p_mat_buf,
        )
        gu_tiles = pl.make_tuple(
            pv_vec_buf=(pv_vec, pv_vec1),
            running_o=running_o,
            exp_corr_fifo=(exp_corr_0, exp_corr_1),
            global_sum_buf=(gsum_0, gsum_1),
            o_f16=o_f16,
        )
        for qi in pl.range(core_id, sq_tiles, num_cores):
            sq_off = qi * TS

            # ---- Main loop: P[ki+preload] ahead + GU[ki] current ----
            for ki in pl.range(0, skv_tiles):
                ctx_curr = ctx_arr[task_id % 3]
                ctx_curr.sq_off = sq_off
                ctx_curr.task_id = task_id
                ctx_curr.qi = qi
                ctx_curr.ki = ki
                ctx_curr.skv_tiles = skv_tiles
                ctx_curr.q_count = q_count
                ctx_curr.core_id = core_id
                if task_id > 0:
                    ctx_p = ctx_arr[(task_id + 2) % 3]
                    p_buf_idx = (ctx_p.q_count * ctx_p.skv_tiles + ctx_p.ki) % 2
                    compute_p(ctx_p.task_id, ctx_p.ki, ctx_p.q_count, p_buf_idx, softmax_tiles)
                if task_id > 1:
                    ctx_g = ctx_arr[(task_id + 1) % 3]
                    compute_gu(
                        ctx_g.task_id,
                        ctx_g.ki,
                        ctx_g.skv_tiles,
                        ctx_g.q_count,
                        ctx_g.sq_off,
                        row_off,
                        o,
                        gu_tiles,
                    )
                task_id = task_id + 1
            q_count = q_count + 1

        # Drain pipeline: last P
        ctx_p2 = ctx_arr[(task_id + 2) % 3]
        p2_buf_idx = (ctx_p2.q_count * ctx_p2.skv_tiles + ctx_p2.ki) % 2
        compute_p(ctx_p2.task_id, ctx_p2.ki, ctx_p2.q_count, p2_buf_idx, softmax_tiles)
        if task_id > 1:
            ctx_g2 = ctx_arr[(task_id + 1) % 3]
            compute_gu(
                ctx_g2.task_id,
                ctx_g2.ki,
                ctx_g2.skv_tiles,
                ctx_g2.q_count,
                ctx_g2.sq_off,
                row_off,
                o,
                gu_tiles,
            )
        task_id = task_id + 1
        # Drain pipeline: last GU
        ctx_g3 = ctx_arr[(task_id + 1) % 3]
        compute_gu(
            ctx_g3.task_id,
            ctx_g3.ki,
            ctx_g3.skv_tiles,
            ctx_g3.q_count,
            ctx_g3.sq_off,
            row_off,
            o,
            gu_tiles,
        )


# ================================================================
#  Reference + Tests
# ================================================================
def flash_attention_ref(q, k, v, d):
    scale_val = 1.0 / math.sqrt(d)
    qk = torch.matmul(q.float(), k.float().T)
    scale = qk * scale_val
    max_val = torch.max(scale, dim=-1, keepdim=True).values
    x_sub = scale - max_val
    x_exp = torch.exp(x_sub)
    attn = x_exp / torch.sum(x_exp, dim=-1, keepdim=True)

    return qk, x_exp, torch.matmul(attn, v.float()).half()


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_fa_perf():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(42)
    for sq, skv, d, num_cores in [
        (8192, 8192, TD, 28),
    ]:
        logging.info("\nFA-Perf (%s,%s,%s) cores=%s  QK_PRELOAD=%s", sq, skv, d, num_cores, QK_PRELOAD)
        q_t = torch.rand((sq, d), device=device, dtype=torch.float16)
        k_t = torch.rand((skv, d), device=device, dtype=torch.float16)
        v_t = torch.rand((skv, d), device=device, dtype=torch.float16)
        o_t = torch.zeros((sq, d), device=device, dtype=torch.float16)
        qk_t = torch.zeros((sq * FIFO_SIZE, skv), device=device, dtype=torch.float32)
        p_t = torch.zeros((sq * FIFO_SIZE, skv), device=device, dtype=torch.float16)
        pv_t = torch.zeros((48 * PV_CORE_STRIDE, d), device=device, dtype=torch.float32)
        fa_perf_tkv_preload_kernel[None, num_cores](q_t, k_t, v_t, o_t, qk_t, p_t, pv_t)
        torch.npu.synchronize()
        qk_ref, x_exp_ref, o_ref = flash_attention_ref(q_t, k_t, v_t, d)
        diff = (o_t - o_ref).abs().max().item()
        logging.info("  max|diff|=%.4f", diff)
        torch.testing.assert_close(o_t, o_ref, rtol=5e-3, atol=5e-3)
        logging.info("  PASS")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.info("FA perf: double-buffer + QK pre-compute (QK_PRELOAD=%s, FIFO=%s)", QK_PRELOAD, FIFO_SIZE)
    logging.info("%s", '=' * 60)
    test_fa_perf()
    logging.info("\nAll FlashAttention tests passed!")
