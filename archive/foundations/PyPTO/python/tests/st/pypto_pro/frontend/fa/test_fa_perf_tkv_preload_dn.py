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

"""FlashAttention performance kernel using PyPTO IR manual (non-SSA) mode -- DN mode.

DN (DecN) mode differences vs standard ND mode:
  - compute_qk: K x Q^T (left/right swapped vs Q x K^T),
                Q loaded with layout="dn" -> L1 shape [TD, TS] layout=pl.ZN,
                K loaded normally -> L1 shape [TKV, TD] layout=pl.NZ,
                matmul Left=K, Right=Q^T, acc output shape [TKV, TS],
                acc_to_vec_mode=pl.AccToVecMode.DualModeSplitN (split along N=TS axis)
  - softmax:    column-direction ops on qk_vec[TKV, TS_HALF]
                (col_max / col_expand_sub / col_sum replacing row_* equivalents)
  - compute_pv: P[TS, TKV] (layout=pl.ZN) x V[TKV, TD] -> acc[TS, TD],
                TINSERT uses offset[1]=TS_HALF*sub_id (column offset)

Reference: fa_performance_dn_kernel.cpp
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
#  Configuration -- change QK_PRELOAD to tune pre-compute depth
# ================================================================
QK_PRELOAD = 1  # How many KV tiles to pre-compute QK ahead
FIFO_SIZE = QK_PRELOAD + 1  # Exp-corr FIFO depth (avoids read/write collision)

# ================================================================
#  Tile dimensions and constants
# ================================================================
TS = 128
TKV = 128
TD = 128
TS_HALF = TS // 2
SCALE = 1.0 / math.sqrt(TD)

# Cube tile byte sizes
Q_F16 = TS * TD * 2
KT_F16 = TKV * TD * 2
V_F16 = TKV * TD * 2
P_F16 = TS * TKV * 2
QK_HALF_F32 = TKV * TS * 4
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
LA1 = KT_F16
RA0 = 0
RA1 = Q_F16
CA0 = 0
CA1 = QK_HALF_F32

# ---- VEC addresses (248KB on a5) ----
VB4_KV = TS_HALF * TKV * 4
VB2_KV = TS_HALF * TKV * 2
VB4 = TS_HALF * TD * 4
VB2 = TS_HALF * TD * 2
VB6_DN = (TKV + 1) * TS_HALF * 2
VB_RED = TS_HALF * 1 * 4

VA0 = 0
VA1 = VA0 + VB4_KV
VA2 = VA1 + VB4_KV
VA3 = VA2 + VB2_KV
VA_GMAX0 = VA3 + VB_RED
VA_GMAX1 = VA_GMAX0 + VB_RED
VA_GSUM0 = VA_GMAX1 + VB_RED
VA_GSUM1 = VA_GSUM0 + VB_RED
VA_EXP_BASE = VA_GSUM1 + VB_RED
VA_EXP0 = VA_EXP_BASE
VA_EXP1 = VA_EXP_BASE + VB_RED
VA_AFTER_EXP = VA_EXP_BASE + FIFO_SIZE * VB_RED
VA7 = VA_AFTER_EXP
VA8 = VA7 + VB4
VA9 = VA8 + VB4
VA10 = VA9 + VB2
VA11 = VA10 + VB6_DN
VA12 = VA11 + VB4_KV
assert VA12 + VB4 <= 248 * 1024, f"VEC overflow: {VA12 + VB4} > {248 * 1024}"

event_ids_01 = (0, 1)
event_ids_23 = (2, 3)

# Cross-core event IDs
QK_READY_IDS = tuple(range(0, FIFO_SIZE))
P_READY_IDS = tuple(range(FIFO_SIZE, 2 * FIFO_SIZE))
PV_READY_IDS = tuple(range(2 * FIFO_SIZE, 3 * FIFO_SIZE))
assert 3 * FIFO_SIZE <= 16, f"Too many cross-core event IDs: need {3 * FIFO_SIZE}, max 16"

PV_CORE_STRIDE = 2 * FIFO_SIZE * TS


# ================================================================
#  Tile allocation helpers
# ================================================================
def alloc_cube_tiles():
    """Allocate cube-section tiles for DN mode."""
    q_mat_type = pl.TileType(shape=[TD, TS], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN)
    q_mat_0 = pl.make_tile(q_mat_type, addr=MA0, size=Q_F16)
    q_mat_1 = pl.make_tile(q_mat_type, addr=MA0_PONG, size=Q_F16)

    k_mat_type = pl.TileType(shape=[TKV, TD], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ)
    k_mat_0 = pl.make_tile(k_mat_type, addr=MA1, size=KT_F16)
    k_mat_1 = pl.make_tile(k_mat_type, addr=MA1_PONG, size=KT_F16)

    v_mat_type = pl.TileType(shape=[TKV, TD], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ)
    v_mat_0 = pl.make_tile(v_mat_type, addr=MA3, size=V_F16)
    v_mat_1 = pl.make_tile(v_mat_type, addr=MA3_PONG, size=V_F16)

    left_0 = pl.make_tile(
        pl.TileType(shape=[TKV, TD], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addr=LA0,
        size=KT_F16,
    )
    left_1 = pl.make_tile(
        pl.TileType(shape=[TKV, TD], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addr=LA1,
        size=KT_F16,
    )
    right_0 = pl.make_tile(
        pl.TileType(shape=[TD, TS], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addr=RA0,
        size=Q_F16,
    )
    right_1 = pl.make_tile(
        pl.TileType(shape=[TD, TS], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addr=RA1,
        size=Q_F16,
    )

    acc_buf1 = pl.make_tile(
        pl.TileType(shape=[TKV, TS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
        addr=CA0,
        size=QK_HALF_F32,
    )
    acc_buf2 = pl.make_tile(
        pl.TileType(shape=[TS, TD], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
        addr=CA1,
        size=PV_HALF_F32,
    )

    return pl.make_tuple(
        q_mat_buf=(q_mat_0, q_mat_1),
        k_mat_buf=(k_mat_0, k_mat_1),
        v_mat_buf=(v_mat_0, v_mat_1),
        left_buf=(left_0, left_1),
        right_buf=(right_0, right_1),
        acc_buf1=acc_buf1,
        acc_buf2=acc_buf2,
    )


def alloc_exp_corr_fifo():
    exp_corr_type = pl.TileType(shape=[TS_HALF, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.DN)
    exp_corr_rm_type = pl.TileType(shape=[1, TS_HALF], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    ec0 = pl.make_tile(exp_corr_type, addr=VA_EXP0, size=VB_RED)
    ec0_rm = pl.make_tile(exp_corr_rm_type, addr=VA_EXP0, size=VB_RED)
    ec1 = pl.make_tile(exp_corr_type, addr=VA_EXP1, size=VB_RED)
    ec1_rm = pl.make_tile(exp_corr_rm_type, addr=VA_EXP1, size=VB_RED)
    return (ec0, ec1), (ec0_rm, ec1_rm)


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
    qk_vec,
    qk_vec1,
) -> None:
    q_mat_buf = cube.q_mat_buf
    k_mat_buf = cube.k_mat_buf
    left_buf = cube.left_buf
    right_buf = cube.right_buf
    acc_buf1 = cube.acc_buf1
    acc_buf2 = cube.acc_buf2
    qk_fifo_slot = task_id % FIFO_SIZE
    skv_off = ki * TKV

    pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=event_ids_01[buf_idx])
    if ki == 0:
        pl.load(q_mat_buf[q_count % 2], q, [sq_off, 0], order=[1, 0])
    pl.load(k_mat_buf[buf_idx], k, [skv_off, 0])
    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
    pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=event_ids_01[l0ab_idx])
    pl.move(left_buf[l0ab_idx], k_mat_buf[buf_idx])
    pl.move(right_buf[l0ab_idx], q_mat_buf[q_count % 2])
    pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
    pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
    pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=event_ids_01[buf_idx])
    pl.system.sync_dst(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=event_ids_01[l0c_idx])
    if l0c_idx == 0:
        pl.matmul(acc_buf1, left_buf[l0ab_idx], right_buf[l0ab_idx])
    else:
        pl.matmul(acc_buf2, left_buf[l0ab_idx], right_buf[l0ab_idx])
    pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
    pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
    pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=event_ids_01[l0ab_idx])
    if qk_fifo_slot == 0:
        if l0c_idx == 0:
            pl.move(qk_vec, acc_buf1, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitN)
        else:
            pl.move(qk_vec, acc_buf2, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitN)
    else:
        if l0c_idx == 0:
            pl.move(qk_vec1, acc_buf1, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitN)
        else:
            pl.move(qk_vec1, acc_buf2, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitN)
    pl.system.sync_src(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=event_ids_01[l0c_idx])
    pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=QK_READY_IDS[qk_fifo_slot])
    return


def compute_pv(
    task_id: pl.DT_INT64,
    ki: pl.DT_INT64,
    q_count: pl.DT_INT64,
    sq_off: pl.DT_INT64,
    skv_tiles: pl.DT_INT64,
    buf_idx: pl.DT_INT64,
    l0ab_idx: pl.DT_INT64,
    l0c_idx: pl.DT_INT64,
    v: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    cube,
    p_mat_buf1,
    p_mat_buf2,
    pv_vec,
    pv_vec1,
) -> None:
    v_mat_buf = cube.v_mat_buf
    left_buf = cube.left_buf
    right_buf = cube.right_buf
    acc_buf1 = cube.acc_buf1
    acc_buf2 = cube.acc_buf2
    pv_task_slot = task_id % FIFO_SIZE
    sv_off = ki * TKV
    pv_fifo_slot = task_id % FIFO_SIZE

    pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=event_ids_23[buf_idx])
    pl.load(v_mat_buf[buf_idx], v, [sv_off, 0])
    pl.system.wait_cross_core(pipe=pl.PipeType.MTE1, event_id=P_READY_IDS[pv_fifo_slot])
    pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=event_ids_01[l0ab_idx])
    if buf_idx == 0:
        pl.move(left_buf[l0ab_idx], p_mat_buf1)
    else:
        pl.move(left_buf[l0ab_idx], p_mat_buf2)
    pl.move(right_buf[l0ab_idx], v_mat_buf[buf_idx])
    pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=event_ids_23[buf_idx])
    pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
    pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
    left_0 = left_buf[0]
    left_1 = left_buf[1]
    right_0 = right_buf[0]
    right_1 = right_buf[1]
    pl.system.sync_dst(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=event_ids_01[l0c_idx])
    if l0ab_idx == 0:
        if l0c_idx == 0:
            pl.matmul(acc_buf1, left_0, right_0)
        else:
            pl.matmul(acc_buf2, left_0, right_0)
    else:
        if l0c_idx == 0:
            pl.matmul(acc_buf1, left_1, right_1)
        else:
            pl.matmul(acc_buf2, left_1, right_1)
    pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
    pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
    pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=event_ids_01[l0ab_idx])
    if pv_task_slot == 0:
        if l0c_idx == 0:
            pl.move(pv_vec, acc_buf1, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
        else:
            pl.move(pv_vec, acc_buf2, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
    else:
        if l0c_idx == 0:
            pl.move(pv_vec1, acc_buf1, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
        else:
            pl.move(pv_vec1, acc_buf2, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
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
    skv_tiles: pl.DT_INT64,
    sq_off: pl.DT_INT64,
    sq_dim: pl.DT_INT64,
    row_off: pl.DT_INT64,
    buf_idx: pl.DT_INT64,
    stiles,
    p_mat_buf1,
    p_mat_buf2,
) -> None:
    """DN softmax: column-direction ops on qk_vec[TKV, TS_HALF]."""
    qk_vec = stiles.qk_vec
    qk_vec1 = stiles.qk_vec1
    tmp_vec = stiles.tmp_vec
    p_f16 = stiles.p_f16
    reduce_dst_rm = stiles.reduce_dst_rm
    global_max_rm_buf = stiles.global_max_rm_buf
    global_sum_rm_buf = stiles.global_sum_rm_buf
    exp_corr_rm_fifo = stiles.exp_corr_rm_fifo
    tile_nz = stiles.tile_nz

    q_idx = q_count % 2
    global_max_rm_cur = global_max_rm_buf[q_idx]
    global_sum_rm_cur = global_sum_rm_buf[q_idx]
    p_fifo_slot = task_id % FIFO_SIZE
    sub_id = pl.get_subblock_idx()
    if p_fifo_slot == 0:
        if ki == 0:
            pl.maximum(reduce_dst_rm, qk_vec, tmp_vec, dim=1)
            pl.expand_sub(tmp_vec, qk_vec, reduce_dst_rm, dim=1)
            pl.mul(global_max_rm_cur, reduce_dst_rm, 1.0)
            pl.mul(tmp_vec, tmp_vec, SCALE)
            pl.exp(qk_vec, tmp_vec)
            pl.sum(reduce_dst_rm, qk_vec, tmp_vec, dim=1)
            pl.mul(global_sum_rm_cur, reduce_dst_rm, 1.0)
            pl.cast(p_f16, qk_vec, mode=pl.RoundMode.CAST_ROUND)
        if ki > 0:
            pl.maximum(reduce_dst_rm, qk_vec, tmp_vec, dim=1)
            pl.maximum(reduce_dst_rm, reduce_dst_rm, global_max_rm_cur)
            pl.sub(exp_corr_rm_fifo[p_fifo_slot], global_max_rm_cur, reduce_dst_rm)
            pl.mul(global_max_rm_cur, reduce_dst_rm, 1.0)
            pl.expand_sub(tmp_vec, qk_vec, reduce_dst_rm, dim=1)
            pl.mul(exp_corr_rm_fifo[p_fifo_slot], exp_corr_rm_fifo[p_fifo_slot], SCALE)
            pl.mul(tmp_vec, tmp_vec, SCALE)
            pl.exp(exp_corr_rm_fifo[p_fifo_slot], exp_corr_rm_fifo[p_fifo_slot])
            pl.exp(qk_vec, tmp_vec)
            pl.cast(p_f16, qk_vec, mode=pl.RoundMode.CAST_ROUND)
            pl.mul(global_sum_rm_cur, global_sum_rm_cur, exp_corr_rm_fifo[p_fifo_slot])
            pl.sum(reduce_dst_rm, qk_vec, tmp_vec, dim=1)
            pl.add(global_sum_rm_cur, global_sum_rm_cur, reduce_dst_rm)
    elif p_fifo_slot == 1:
        if ki == 0:
            pl.maximum(reduce_dst_rm, qk_vec1, tmp_vec, dim=1)
            pl.expand_sub(tmp_vec, qk_vec1, reduce_dst_rm, dim=1)
            pl.mul(global_max_rm_cur, reduce_dst_rm, 1.0)
            pl.mul(tmp_vec, tmp_vec, SCALE)
            pl.exp(qk_vec1, tmp_vec)
            pl.sum(reduce_dst_rm, qk_vec1, tmp_vec, dim=1)
            pl.mul(global_sum_rm_cur, reduce_dst_rm, 1.0)
            pl.cast(p_f16, qk_vec1, mode=pl.RoundMode.CAST_ROUND)
        if ki > 0:
            pl.maximum(reduce_dst_rm, qk_vec1, tmp_vec, dim=1)
            pl.maximum(reduce_dst_rm, reduce_dst_rm, global_max_rm_cur)
            pl.sub(exp_corr_rm_fifo[p_fifo_slot], global_max_rm_cur, reduce_dst_rm)
            pl.mul(global_max_rm_cur, reduce_dst_rm, 1.0)
            pl.expand_sub(tmp_vec, qk_vec1, reduce_dst_rm, dim=1)
            pl.mul(exp_corr_rm_fifo[p_fifo_slot], exp_corr_rm_fifo[p_fifo_slot], SCALE)
            pl.mul(tmp_vec, tmp_vec, SCALE)
            pl.exp(exp_corr_rm_fifo[p_fifo_slot], exp_corr_rm_fifo[p_fifo_slot])
            pl.exp(qk_vec1, tmp_vec)
            pl.cast(p_f16, qk_vec1, mode=pl.RoundMode.CAST_ROUND)
            pl.mul(global_sum_rm_cur, global_sum_rm_cur, exp_corr_rm_fifo[p_fifo_slot])
            pl.sum(reduce_dst_rm, qk_vec1, tmp_vec, dim=1)
            pl.add(global_sum_rm_cur, global_sum_rm_cur, reduce_dst_rm)
    pl.move(tile_nz, p_f16)
    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
    if buf_idx == 0:
        pl.insert(p_mat_buf1, tile_nz, [0, TS_HALF * sub_id])
    else:
        pl.insert(p_mat_buf2, tile_nz, [0, TS_HALF * sub_id])
    return


def compute_p(
    task_id: pl.DT_INT64,
    ki: pl.DT_INT64,
    q_count: pl.DT_INT64,
    skv_tiles: pl.DT_INT64,
    sq_off: pl.DT_INT64,
    sq_dim: pl.DT_INT64,
    row_off: pl.DT_INT64,
    buf_idx: pl.DT_INT64,
    stiles,
    p_mat_buf1,
    p_mat_buf2,
) -> None:
    """Softmax on KQ tile -> P. Includes cross-core sync."""
    p_fifo_slot = task_id % FIFO_SIZE
    pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=QK_READY_IDS[p_fifo_slot])
    softmax_body(
        task_id,
        ki,
        q_count,
        skv_tiles,
        sq_off,
        sq_dim,
        row_off,
        buf_idx,
        stiles,
        p_mat_buf1,
        p_mat_buf2,
    )
    pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=P_READY_IDS[p_fifo_slot])
    return


def compute_gu(
    task_id: pl.DT_INT64,
    ki: pl.DT_INT64,
    skv_tiles: pl.DT_INT64,
    sq_off: pl.DT_INT64,
    row_off: pl.DT_INT64,
    q_count: pl.DT_INT64,
    o: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    gtiles,
) -> None:
    """GU: running output update."""
    pv_vec = gtiles.pv_vec
    pv_vec1 = gtiles.pv_vec1
    running_o = gtiles.running_o
    exp_corr_fifo = gtiles.exp_corr_fifo
    global_sum_buf = gtiles.global_sum_buf
    o_f16 = gtiles.o_f16
    global_sum_cur = global_sum_buf[q_count % 2]
    pv_slot = task_id % FIFO_SIZE
    pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=PV_READY_IDS[pv_slot])
    if pv_slot == 0:
        if ki == 0:
            pl.move(running_o, pv_vec)
        if ki > 0:
            pl.expand_mul(running_o, running_o, exp_corr_fifo[pv_slot])
            pl.add(running_o, running_o, pv_vec)
    else:
        if ki == 0:
            pl.move(running_o, pv_vec1)
        if ki > 0:
            pl.expand_mul(running_o, running_o, exp_corr_fifo[pv_slot])
            pl.add(running_o, running_o, pv_vec1)
    if ki == skv_tiles - 1:
        pl.expand_div(running_o, running_o, global_sum_cur)
        pl.cast(o_f16, running_o, mode=pl.RoundMode.CAST_ROUND)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
        pl.store(o, o_f16, [sq_off + row_off, 0])
    return


# ================================================================
#  Kernel
# ================================================================
@pl.jit()
def fa_perf_tkv_preload_dn_kernel(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    v: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    o: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    qk_buf: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    p_buf: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    pv_buf: pl.Tensor[[48 * PV_CORE_STRIDE, pl.DYNAMIC], pl.DT_FP32],
):

    sq_dim = q.shape[0]
    skv_dim = k.shape[0]
    sq_tiles = (sq_dim + (TS - 1)) // TS
    skv_tiles = (skv_dim + (TKV - 1)) // TKV
    num_cores = pl.get_block_num()
    core_id = pl.get_block_idx() // pl.get_subblock_num()

    # DN: qk_vec shape [TKV, TS_HALF]
    qk_vec = pl.make_tile(
        pl.TileType(shape=[TKV, TS_HALF], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addr=VA0,
        size=VB4_KV,
    )
    qk_vec1 = pl.make_tile(
        pl.TileType(shape=[TKV, TS_HALF], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addr=VA11,
        size=VB4_KV,
    )
    pv_vec = pl.make_tile(
        pl.TileType(shape=[TS_HALF, TD], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addr=VA8,
        size=VB4,
    )
    pv_vec1 = pl.make_tile(
        pl.TileType(shape=[TS_HALF, TD], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addr=VA12,
        size=VB4,
    )
    running_o = pl.make_tile(
        pl.TileType(shape=[TS_HALF, TD], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addr=VA7,
        size=VB4,
    )

    p_mat_type = pl.TileType(shape=[TS, TKV], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN)
    p_mat_buf1 = pl.make_tile(p_mat_type, addr=MA2, size=P_F16)
    p_mat_buf2 = pl.make_tile(p_mat_type, addr=MA2_PONG, size=P_F16)

    # =================== CUBE SECTION ===================
    with pl.section_cube():
        cube_tiles = alloc_cube_tiles()

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
        # Track previous task parameters for pipelined PV
        prev_sq_off = 0
        prev_ki = 0
        prev_q_count = 0
        prev_skv_tiles = 0
        prev_buf_idx = 0
        for qi in pl.range(core_id, sq_tiles, num_cores):
            sq_off = qi * TS
            for ki in pl.range(0, skv_tiles):
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
                    qk_vec,
                    qk_vec1,
                )
                l0ab_idx = 1 - l0ab_idx
                l0c_idx = 1 - l0c_idx

                if task_id > 0:
                    compute_pv(
                        task_id - 1,
                        prev_ki,
                        prev_q_count,
                        prev_sq_off,
                        prev_skv_tiles,
                        prev_buf_idx,
                        l0ab_idx,
                        l0c_idx,
                        v,
                        cube_tiles,
                        p_mat_buf1,
                        p_mat_buf2,
                        pv_vec,
                        pv_vec1,
                    )
                    l0ab_idx = 1 - l0ab_idx
                    l0c_idx = 1 - l0c_idx
                prev_sq_off = sq_off
                prev_ki = ki
                prev_q_count = q_count
                prev_skv_tiles = skv_tiles
                prev_buf_idx = buf_idx
                task_id = task_id + 1
            q_count = q_count + 1

        # Final PV for the last QK task
        compute_pv(
            task_id - 1,
            prev_ki,
            prev_q_count,
            prev_sq_off,
            prev_skv_tiles,
            prev_buf_idx,
            l0ab_idx,
            l0c_idx,
            v,
            cube_tiles,
            p_mat_buf1,
            p_mat_buf2,
            pv_vec,
            pv_vec1,
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
        tmp_vec = pl.make_tile(
            pl.TileType(shape=[TKV, TS_HALF], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addr=VA1,
            size=VB4_KV,
        )
        p_f16 = pl.make_tile(
            pl.TileType(shape=[TKV, TS_HALF], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
            addr=VA2,
            size=VB2_KV,
        )
        reduce_dst_rm = pl.make_tile(
            pl.TileType(shape=[1, TS_HALF], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addr=VA3,
            size=VB_RED,
        )

        red_rm_type = pl.TileType(shape=[1, TS_HALF], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        red_type = pl.TileType(shape=[TS_HALF, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.DN)

        gmax_rm_0 = pl.make_tile(red_rm_type, addr=VA_GMAX0, size=VB_RED)
        gmax_rm_1 = pl.make_tile(red_rm_type, addr=VA_GMAX1, size=VB_RED)
        global_max_rm_buf = (gmax_rm_0, gmax_rm_1)

        gsum_0 = pl.make_tile(red_type, addr=VA_GSUM0, size=VB_RED)
        gsum_1 = pl.make_tile(red_type, addr=VA_GSUM1, size=VB_RED)
        gsum_rm_0 = pl.make_tile(red_rm_type, addr=VA_GSUM0, size=VB_RED)
        gsum_rm_1 = pl.make_tile(red_rm_type, addr=VA_GSUM1, size=VB_RED)
        global_sum_buf = (gsum_0, gsum_1)
        global_sum_rm_buf = (gsum_rm_0, gsum_rm_1)

        exp_corr_fifo, exp_corr_rm_fifo = alloc_exp_corr_fifo()

        o_f16 = pl.make_tile(
            pl.TileType(shape=[TS_HALF, TD], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
            addr=VA9,
            size=VB2,
        )

        tile_type_nz = pl.TileType(
            shape=[TKV, TS_HALF],
            dtype=pl.DT_FP16,
            target_memory=pl.MemorySpace.Vec,
            layout=pl.NZ,
        )
        tile_nz = pl.make_tile(tile_type_nz, addr=VA10, size=VB6_DN)

        task_id = 0
        q_count = 0
        sub_id = pl.get_subblock_idx()
        row_off = sub_id * TS_HALF

        # Track previous task params for pipelined P/GU
        prev_sq_off = 0
        prev_ki = 0
        prev_q_count = 0
        prev_skv_tiles = 0
        prev_buf_idx = 0
        prev2_sq_off = 0
        prev2_ki = 0
        prev2_q_count = 0
        prev2_skv_tiles = 0
        for qi in pl.range(core_id, sq_tiles, num_cores):
            sq_off = qi * TS
            softmax_tiles = pl.make_tuple(
                qk_vec=qk_vec,
                qk_vec1=qk_vec1,
                tmp_vec=tmp_vec,
                p_f16=p_f16,
                reduce_dst_rm=reduce_dst_rm,
                global_max_rm_buf=global_max_rm_buf,
                global_sum_rm_buf=global_sum_rm_buf,
                exp_corr_rm_fifo=exp_corr_rm_fifo,
                tile_nz=tile_nz,
            )
            gu_tiles = pl.make_tuple(
                pv_vec=pv_vec,
                pv_vec1=pv_vec1,
                running_o=running_o,
                exp_corr_fifo=exp_corr_fifo,
                global_sum_buf=global_sum_buf,
                o_f16=o_f16,
            )

            for ki in pl.range(0, skv_tiles):
                buf_idx = (q_count * skv_tiles + ki) % 2
                if task_id > 0:
                    compute_p(
                        task_id - 1,
                        prev_ki,
                        prev_q_count,
                        prev_skv_tiles,
                        prev_sq_off,
                        sq_dim,
                        row_off,
                        prev_buf_idx,
                        softmax_tiles,
                        p_mat_buf1,
                        p_mat_buf2,
                    )
                if task_id > 1:
                    compute_gu(
                        task_id - 2,
                        prev2_ki,
                        prev2_skv_tiles,
                        prev2_sq_off,
                        row_off,
                        prev2_q_count,
                        o,
                        gu_tiles,
                    )
                prev2_sq_off = prev_sq_off
                prev2_ki = prev_ki
                prev2_q_count = prev_q_count
                prev2_skv_tiles = prev_skv_tiles
                prev_sq_off = sq_off
                prev_ki = ki
                prev_q_count = q_count
                prev_skv_tiles = skv_tiles
                prev_buf_idx = buf_idx
                task_id = task_id + 1
            q_count = q_count + 1

        # Drain: compute_p for last task (softmax_tiles and gu_tiles still valid)
        compute_p(
            task_id - 1,
            prev_ki,
            prev_q_count,
            prev_skv_tiles,
            prev_sq_off,
            sq_dim,
            row_off,
            prev_buf_idx,
            softmax_tiles,
            p_mat_buf1,
            p_mat_buf2,
        )
        if task_id > 1:
            compute_gu(
                task_id - 2,
                prev2_ki,
                prev2_skv_tiles,
                prev2_sq_off,
                row_off,
                prev2_q_count,
                o,
                gu_tiles,
            )
        task_id = task_id + 1
        compute_gu(
            task_id - 2,
            prev_ki,
            prev_skv_tiles,
            prev_sq_off,
            row_off,
            prev_q_count,
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
    num_cores = 28

    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(42)
    for sq, skv, d in [
        (8192, 8192, TD),
    ]:
        logging.info("\nFA-Perf DN (%s,%s,%s) cores=%s  QK_PRELOAD=%s", sq, skv, d, num_cores, QK_PRELOAD)
        q_t = torch.rand((sq, d), device=device, dtype=torch.float16)
        k_t = torch.rand((skv, d), device=device, dtype=torch.float16)
        v_t = torch.rand((skv, d), device=device, dtype=torch.float16)
        o_t = torch.zeros((sq, d), device=device, dtype=torch.float16)
        qk_t = torch.zeros((sq * FIFO_SIZE, skv), device=device, dtype=torch.float32)
        p_t = torch.zeros((sq * FIFO_SIZE, skv), device=device, dtype=torch.float16)
        pv_t = torch.zeros((48 * PV_CORE_STRIDE, d), device=device, dtype=torch.float32)
        fa_perf_tkv_preload_dn_kernel[None, num_cores](q_t, k_t, v_t, o_t, qk_t, p_t, pv_t)
        torch.npu.synchronize()
        qk_ref, x_exp_ref, o_ref = flash_attention_ref(q_t, k_t, v_t, d)
        diff = (o_t - o_ref).abs().max().item()
        logging.info("  max|diff|=%.4f", diff)
        torch.testing.assert_close(o_t, o_ref, rtol=5e-3, atol=5e-3)
        logging.info("  PASS")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.info("FA perf DN: double-buffer + QK pre-compute (QK_PRELOAD=%s, FIFO=%s)", QK_PRELOAD, FIFO_SIZE)
    logging.info("%s", '=' * 60)
    test_fa_perf()
    logging.info("\nAll FlashAttention DN tests passed!")
