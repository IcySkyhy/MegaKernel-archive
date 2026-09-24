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
                Q loaded with is_transpose=True -> L1 shape [TD, TS] layout=pl.ZN,
                K loaded normally -> L1 shape [TKV, TD] layout=pl.NZ,
                matmul Left=K, Right=Q^T, acc output shape [TKV, TS]
  - softmax:    column-direction ops on qk_vec[TKV, actual_sq_half]
                (col_max / col_expand_sub / col_sum replacing row_* equivalents)
  - compute_pv: P[TS, TKV] (layout=pl.ZN) x V[TKV, TD] -> acc[TS, TD],
                TINSERT uses dynamic index_col based on actual_sq_half and sub_id
Unaligned tail handling (this file):
  - Pad actual_sq up to next multiple of 32 -> actual_sq_pad. Aligning to 32 (not
    16) is REQUIRED so each dual_split half (= pad/2) is 16-element aligned --
    fixpipe N (output cols) must be 16-element aligned or it raises an address
    misalign exception. (Row alignment is NOT required: actual_skv may stay e.g.
    62 unchanged.)
  - Always use dual_mode_split_n / dual_mode_split_m (no single_vec0 fallback) so both
    sub_blocks stay active even on tail tiles.
  - Each sub_id processes actual_sq_pad // 2 columns/rows in compute; columns
    beyond actual_sq contain stale acc data but stay isolated within their own
    column (all softmax reductions are col-wise), so they do not pollute real
    columns. compute_gu stores only the real-row count per sub_id, so junk rows
    are computed but never written back to o.
"""

import logging
import math
import os

import pypto_pro.language as pl
import pytest
import torch

import pypto

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
Q_F16 = TS * TD * 2  # [TS,  TD]  FP16 = 32KB  (DN: stored as [TD, TS])
KT_F16 = TKV * TD * 2  # [TKV, TD]  FP16 = 32KB  (DN: K normal layout)
V_F16 = TKV * TD * 2  # [TKV, TD]  FP16 = 32KB
P_F16 = TS * TKV * 2  # [TS,  TKV] FP16 = 32KB
QK_HALF_F32 = TKV * TS * 4  # [TKV, TS]  FP32 = 64KB  (DN acc shape)
PV_HALF_F32 = TS * TD * 4  # [TS,  TD]  FP32 = 64KB

# ---- MAT (512KB) ----
MA0 = 0
MA0_PONG = MA0 + Q_F16
MA1 = Q_F16 * 2
MA1_PONG = MA1 + KT_F16
MA2 = MA1 + KT_F16 * 2
MA2_PONG = MA2 + P_F16
MA3 = MA2 + P_F16 * 2
MA3_PONG = MA3 + V_F16
# DN: Left holds K [TKV,TD], Right holds Q^T [TD,TS]
# All sizes = 32KB so address offsets are unchanged
LA0 = 0
LA1 = KT_F16
RA0 = 0
RA1 = Q_F16
CA0 = 0
CA1 = QK_HALF_F32

# ---- VEC addresses (248KB on a5) ----
VB4_KV = TS_HALF * TKV * 4  # [TS_HALF, TKV] or [TKV, TS_HALF] FP32 = 32KB (same bytes)
VB2_KV = TS_HALF * TKV * 2  # [TS_HALF, TKV] or [TKV, TS_HALF] FP16 = 16KB
VB4 = TS_HALF * TD * 4  # [TS_HALF, TD]  FP32 = 32KB
VB2 = TS_HALF * TD * 2  # [TS_HALF, TD]  FP16 = 16KB
VB6_DN = (TKV + 1) * TS_HALF * 2  # [TKV+1, TS_HALF] FP16 = 129 * 64 * 2 = 16512 B
VB_RED = TS_HALF * 1 * 4  # [TS_HALF, 1] FP32 = 256 B

VA0 = 0  # qk_vec  [TKV, TS_HALF] FP32 (DN shape)
VA1 = VA0 + VB4_KV  # tmp_vec [TKV, TS_HALF] FP32
VA2 = VA1 + VB4_KV  # p_f16   [TKV, TS_HALF] FP16
VA3 = VA2 + VB2_KV  # reduce_dst_rm
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
VA7 = VA_EXP_BASE + FIFO_SIZE * VB_RED  # running_o [TS_HALF, TD] FP32
VA8 = VA7 + VB4  # pv_vec    [TS_HALF, TD] FP32
VA9 = VA8 + VB4  # o_f16     [TS_HALF, TD] FP16
VA10 = VA9 + VB2  # tile_nz   [TKV, TS_HALF] FP16 (DN shape)
VA11 = VA10 + VB6_DN  # qk_vec1   [TKV, TS_HALF] FP32 (slot 1)
VA12 = VA11 + VB4_KV  # pv_vec1   [TS_HALF, TD] FP32 (slot 1)
assert VA12 + VB4 <= 248 * 1024, f"VEC overflow: {VA12 + VB4} > {248 * 1024}"

event_ids_01 = (0, 1)
event_ids_23 = (2, 3)

# Cross-core event IDs
QK_READY_IDS = tuple(range(0, FIFO_SIZE))
P_READY_IDS = tuple(range(FIFO_SIZE, 2 * FIFO_SIZE))
PV_READY_IDS = tuple(range(2 * FIFO_SIZE, 3 * FIFO_SIZE))
assert 3 * FIFO_SIZE <= 16, f"Too many cross-core event IDs: need {3 * FIFO_SIZE}, max 16"

# PV buffer: 2 Q-slots x FIFO_SIZE task-slots per core
PV_CORE_STRIDE = 2 * FIFO_SIZE * TS


def align32byte_fp32(n):
    """Align n up to next multiple of 32 (so each dual_split half is 16-aligned)."""
    return ((n + 31) // 32) * 32


# ================================================================
#  Tile allocation helpers
# ================================================================
def alloc_cube_tiles():
    """Allocate cube-section tiles for DN mode."""
    q_mat_type = pl.TileType(
        shape=[TD, TS],
        dtype=pl.DT_FP16,
        target_memory=pl.MemorySpace.Mat,
        layout=pl.ZN,
        valid_shape=[-1, -1],
        compact=1,
    )
    q_mat_0 = pl.make_tile(q_mat_type, addr=MA0, size=Q_F16)
    q_mat_1 = pl.make_tile(q_mat_type, addr=MA0_PONG, size=Q_F16)

    k_mat_type = pl.TileType(
        shape=[TKV, TD],
        dtype=pl.DT_FP16,
        target_memory=pl.MemorySpace.Mat,
        layout=pl.NZ,
        valid_shape=[-1, -1],
        compact=1,
    )
    k_mat_0 = pl.make_tile(k_mat_type, addr=MA1, size=KT_F16)
    k_mat_1 = pl.make_tile(k_mat_type, addr=MA1_PONG, size=KT_F16)

    v_mat_type = pl.TileType(
        shape=[TKV, TD],
        dtype=pl.DT_FP16,
        target_memory=pl.MemorySpace.Mat,
        layout=pl.NZ,
        valid_shape=[-1, -1],
        compact=1,
    )
    v_mat_0 = pl.make_tile(v_mat_type, addr=MA3, size=V_F16)
    v_mat_1 = pl.make_tile(v_mat_type, addr=MA3_PONG, size=V_F16)

    left_0 = pl.make_tile(
        pl.TileType(
            shape=[TKV, TD],
            dtype=pl.DT_FP16,
            target_memory=pl.MemorySpace.Left,
            layout=pl.NZ,
            valid_shape=[-1, -1],
            compact=1,
        ),
        addr=LA0,
        size=KT_F16,
    )
    left_1 = pl.make_tile(
        pl.TileType(
            shape=[TKV, TD],
            dtype=pl.DT_FP16,
            target_memory=pl.MemorySpace.Left,
            layout=pl.NZ,
            valid_shape=[-1, -1],
            compact=1,
        ),
        addr=LA1,
        size=KT_F16,
    )

    right_0 = pl.make_tile(
        pl.TileType(
            shape=[TD, TS],
            dtype=pl.DT_FP16,
            target_memory=pl.MemorySpace.Right,
            valid_shape=[-1, -1],
            compact=1,
        ),
        addr=RA0,
        size=Q_F16,
    )
    right_1 = pl.make_tile(
        pl.TileType(
            shape=[TD, TS],
            dtype=pl.DT_FP16,
            target_memory=pl.MemorySpace.Right,
            valid_shape=[-1, -1],
            compact=1,
        ),
        addr=RA1,
        size=Q_F16,
    )

    acc_0 = pl.make_tile(
        pl.TileType(
            shape=[TKV, TS],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Acc,
            valid_shape=[-1, -1],
            compact=1,
        ),
        addr=CA0,
        size=QK_HALF_F32,
    )
    acc_1 = pl.make_tile(
        pl.TileType(
            shape=[TS, TD],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Acc,
            valid_shape=[-1, -1],
            compact=1,
        ),
        addr=CA1,
        size=PV_HALF_F32,
    )

    return pl.make_tuple(
        q_mat_buf=(q_mat_0, q_mat_1),
        k_mat_buf=(k_mat_0, k_mat_1),
        v_mat_buf=(v_mat_0, v_mat_1),
        left_buf=(left_0, left_1),
        right_buf=(right_0, right_1),
        acc_buf1=acc_0,
        acc_buf2=acc_1,
    )


def alloc_exp_corr_fifo():
    exp_corr_type = pl.TileType(
        shape=[TS_HALF, 1],
        dtype=pl.DT_FP32,
        target_memory=pl.MemorySpace.Vec,
        layout=pl.DN,
        valid_shape=[-1, -1],
        compact=1,
    )
    exp_corr_rm_type = pl.TileType(
        shape=[1, TS_HALF], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1], compact=1
    )
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
    skv_off: pl.DT_INT64,
    buf_idx: pl.DT_INT64,
    actual_skv: pl.DT_INT64,
    actual_sq: pl.DT_INT64,
    actual_sq_pad: pl.DT_INT64,
    actual_sq_half: pl.DT_INT64,
    l0ab_idx: pl.DT_INT64,
    l0c_idx: pl.DT_INT64,
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    cube,
    qk_vec,
    qk_vec1,
) -> None:
    """DN: KQ^T = K * Q^T. Both full and tail tiles use dual_mode_split_n.

    Each sub_id moves actual_sq_half = actual_sq_pad // 2 columns.
    """
    q_mat_buf = cube.q_mat_buf
    k_mat_buf = cube.k_mat_buf
    left_buf = cube.left_buf
    right_buf = cube.right_buf
    acc_buf1 = cube.acc_buf1
    acc_buf2 = cube.acc_buf2
    qk_fifo_slot = task_id % FIFO_SIZE

    pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=event_ids_01[buf_idx])
    # DN: Q loaded with is_transpose=True -> stored in L1 as [TD, TS] (transposed)
    if ki == 0:
        pl.set_validshape(q_mat_buf[q_count % 2], [TD, actual_sq])
        pl.load(q_mat_buf[q_count % 2], q, [sq_off, 0], order=[1, 0])
    pl.set_validshape(k_mat_buf[buf_idx], [actual_skv, TD])
    pl.load(k_mat_buf[buf_idx], k, [skv_off, 0])
    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
    pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=event_ids_01[l0ab_idx])
    pl.set_validshape(left_buf[l0ab_idx], [actual_skv, TD])
    pl.move(left_buf[l0ab_idx], k_mat_buf[buf_idx])
    pl.set_validshape(right_buf[l0ab_idx], [TD, actual_sq])
    pl.move(right_buf[l0ab_idx], q_mat_buf[q_count % 2])
    pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
    pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
    pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=event_ids_01[buf_idx])
    pl.system.sync_dst(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=event_ids_01[l0c_idx])
    # matmul fills only actual_sq columns of acc
    if l0c_idx == 0:
        pl.set_validshape(acc_buf1, [actual_skv, actual_sq])
        pl.matmul(acc_buf1, left_buf[l0ab_idx], right_buf[l0ab_idx])
    else:
        pl.set_validshape(acc_buf2, [actual_skv, actual_sq])
        pl.matmul(acc_buf2, left_buf[l0ab_idx], right_buf[l0ab_idx])
    pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
    pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
    pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=event_ids_01[l0ab_idx])
    # Always dual_mode_split_n: extend acc valid_shape to actual_sq_pad (multiple of 32)
    # so FIX splits at actual_sq_half = actual_sq_pad // 2, which is 16-element
    # aligned (required by fixpipe N alignment). Columns in [actual_sq, actual_sq_pad)
    # hold stale acc data but stay column-isolated through the softmax reductions.
    if qk_fifo_slot == 0:
        if l0c_idx == 0:
            pl.set_validshape(acc_buf1, [actual_skv, actual_sq_pad])
            pl.set_validshape(qk_vec, [actual_skv, actual_sq_half])
            pl.move(qk_vec, acc_buf1, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitN)
        else:
            pl.set_validshape(acc_buf2, [actual_skv, actual_sq_pad])
            pl.set_validshape(qk_vec, [actual_skv, actual_sq_half])
            pl.move(qk_vec, acc_buf2, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitN)
    else:
        if l0c_idx == 0:
            pl.set_validshape(acc_buf1, [actual_skv, actual_sq_pad])
            pl.set_validshape(qk_vec1, [actual_skv, actual_sq_half])
            pl.move(qk_vec1, acc_buf1, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitN)
        else:
            pl.set_validshape(acc_buf2, [actual_skv, actual_sq_pad])
            pl.set_validshape(qk_vec1, [actual_skv, actual_sq_half])
            pl.move(qk_vec1, acc_buf2, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitN)
    pl.system.sync_src(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=event_ids_01[l0c_idx])
    pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=QK_READY_IDS[qk_fifo_slot])
    return


def compute_pv(
    task_id: pl.DT_INT64,
    ki: pl.DT_INT64,
    q_count: pl.DT_INT64,
    skv_off: pl.DT_INT64,
    buf_idx: pl.DT_INT64,
    actual_skv: pl.DT_INT64,
    actual_sq_pad: pl.DT_INT64,
    actual_sq_half: pl.DT_INT64,
    l0ab_idx: pl.DT_INT64,
    l0c_idx: pl.DT_INT64,
    v: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    cube,
    p_mat_buf1,
    p_mat_buf2,
    pv_vec,
    pv_vec1,
) -> None:
    """DN: PV = P * V. P[TS,TKV] x V[TKV,TD] -> acc[TS,TD].

    Always dual_mode_split_m: each sub_id receives actual_sq_half rows of acc.
    """
    v_mat_buf = cube.v_mat_buf
    left_buf = cube.left_buf
    right_buf = cube.right_buf
    acc_buf1 = cube.acc_buf1
    acc_buf2 = cube.acc_buf2
    pv_fifo_slot = task_id % FIFO_SIZE

    pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=event_ids_23[buf_idx])
    pl.set_validshape(v_mat_buf[buf_idx], [actual_skv, TD])
    pl.load(v_mat_buf[buf_idx], v, [skv_off, 0])
    pl.system.wait_cross_core(pipe=pl.PipeType.MTE1, event_id=P_READY_IDS[pv_fifo_slot])
    pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=event_ids_01[l0ab_idx])
    # P columns 0..actual_sq_pad-1 of p_mat are filled (both sub_ids ran TINSERT in
    # softmax); cols [actual_sq, actual_sq_pad) are junk. Drive PV with the padded
    # extent so dual_mode_split_m can split acc rows evenly between the two sub_blocks.
    pl.set_validshape(left_buf[l0ab_idx], [actual_sq_pad, actual_skv])
    if buf_idx == 0:
        pl.set_validshape(p_mat_buf1, [actual_skv, actual_sq_pad])
        pl.move(left_buf[l0ab_idx], p_mat_buf1)
    else:
        pl.set_validshape(p_mat_buf2, [actual_skv, actual_sq_pad])
        pl.move(left_buf[l0ab_idx], p_mat_buf2)
    pl.set_validshape(right_buf[l0ab_idx], [actual_skv, TD])
    pl.move(right_buf[l0ab_idx], v_mat_buf[buf_idx])
    pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=event_ids_23[buf_idx])
    pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
    pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
    pl.system.sync_dst(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=event_ids_01[l0c_idx])
    if l0c_idx == 0:
        pl.set_validshape(acc_buf1, [actual_sq_pad, TD])
        pl.matmul(acc_buf1, left_buf[l0ab_idx], right_buf[l0ab_idx])
    else:
        pl.set_validshape(acc_buf2, [actual_sq_pad, TD])
        pl.matmul(acc_buf2, left_buf[l0ab_idx], right_buf[l0ab_idx])
    pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
    pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
    pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=event_ids_01[l0ab_idx])
    if pv_fifo_slot == 0:
        if l0c_idx == 0:
            pl.set_validshape(pv_vec, [actual_sq_half, TD])
            pl.move(pv_vec, acc_buf1, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
        else:
            pl.set_validshape(pv_vec, [actual_sq_half, TD])
            pl.move(pv_vec, acc_buf2, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
    else:
        if l0c_idx == 0:
            pl.set_validshape(pv_vec1, [actual_sq_half, TD])
            pl.move(pv_vec1, acc_buf1, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
        else:
            pl.set_validshape(pv_vec1, [actual_sq_half, TD])
            pl.move(pv_vec1, acc_buf2, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
    pl.system.sync_src(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=event_ids_01[l0c_idx])
    pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=PV_READY_IDS[pv_fifo_slot])
    return


# ================================================================
# Vector section - softmax_body, compute_p, compute_gu
# ================================================================
def softmax_body(
    task_id: pl.DT_INT64,
    ki: pl.DT_INT64,
    q_count: pl.DT_INT64,
    buf_idx: pl.DT_INT64,
    actual_skv: pl.DT_INT64,
    actual_sq_half: pl.DT_INT64,
    stiles,
    p_mat_buf1,
    p_mat_buf2,
) -> None:
    """DN softmax: column-direction ops. Always dual_mode_split_n -- both sub_ids
    process actual_sq_half columns. fillpad handles the row-padding [actual_skv,
    TKV) so col_max/col_sum stay correct.
    """
    qk_vec = stiles.qk_vec
    qk_vec1 = stiles.qk_vec1
    tmp_vec = stiles.tmp_vec
    p_f16 = stiles.p_f16
    reduce_dst_rm = stiles.reduce_dst_rm
    global_max_rm_buf = stiles.global_max_rm_buf
    global_sum_rm_buf = stiles.global_sum_rm_buf
    exp_corr_rm_fifo = stiles.exp_corr_rm_fifo
    tile_nz = stiles.tile_nz

    p_fifo_slot = task_id % FIFO_SIZE
    q_idx = q_count % 2
    sub_id = pl.get_subblock_idx()
    global_max_rm_cur = global_max_rm_buf[q_idx]
    global_sum_rm_cur = global_sum_rm_buf[q_idx]

    pl.set_validshape(tmp_vec, [actual_skv, actual_sq_half])
    pl.set_validshape(p_f16, [actual_skv, actual_sq_half])
    pl.set_validshape(reduce_dst_rm, [1, actual_sq_half])
    pl.set_validshape(global_max_rm_cur, [1, actual_sq_half])
    pl.set_validshape(global_sum_rm_cur, [1, actual_sq_half])
    pl.set_validshape(exp_corr_rm_fifo[p_fifo_slot], [1, actual_sq_half])

    if p_fifo_slot == 0:
        pl.set_validshape(qk_vec, [actual_skv, actual_sq_half])
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
        pl.set_validshape(qk_vec1, [actual_skv, actual_sq_half])
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
    pl.set_validshape(tile_nz, [actual_skv, actual_sq_half])
    pl.move(tile_nz, p_f16)
    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
    if buf_idx == 0:
        pl.insert(p_mat_buf1, tile_nz, [0, actual_sq_half * sub_id])
    else:
        pl.insert(p_mat_buf2, tile_nz, [0, actual_sq_half * sub_id])
    return


def compute_p(
    task_id: pl.DT_INT64,
    ki: pl.DT_INT64,
    q_count: pl.DT_INT64,
    buf_idx: pl.DT_INT64,
    actual_skv: pl.DT_INT64,
    actual_sq_half: pl.DT_INT64,
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
        buf_idx,
        actual_skv,
        actual_sq_half,
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
    actual_sq_half: pl.DT_INT64,
    real_rows: pl.DT_INT64,
    o: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    gtiles,
) -> None:
    """GU: running output update. Both sub_blocks accumulate actual_sq_half rows
    of pv_vec into running_o; at the final ki, only the real-row count for this
    sub_id is written back to o (junk rows are computed but discarded).
    """
    pv_vec = gtiles.pv_vec
    pv_vec1 = gtiles.pv_vec1
    running_o = gtiles.running_o
    exp_corr_fifo = gtiles.exp_corr_fifo
    global_sum_buf = gtiles.global_sum_buf
    o_f16 = gtiles.o_f16
    pv_slot = task_id % FIFO_SIZE
    pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=PV_READY_IDS[pv_slot])
    pl.set_validshape(running_o, [actual_sq_half, TD])
    if pv_slot == 0:
        if ki == 0:
            pl.move(running_o, pv_vec)
        if ki > 0:
            pl.set_validshape(exp_corr_fifo[pv_slot], [actual_sq_half, 1])
            pl.expand_mul(running_o, running_o, exp_corr_fifo[pv_slot])
            pl.add(running_o, running_o, pv_vec)
    else:
        if ki == 0:
            pl.move(running_o, pv_vec1)
        if ki > 0:
            pl.set_validshape(exp_corr_fifo[pv_slot], [actual_sq_half, 1])
            pl.expand_mul(running_o, running_o, exp_corr_fifo[pv_slot])
            pl.add(running_o, running_o, pv_vec1)
    if ki == skv_tiles - 1:
        pl.expand_div(running_o, running_o, global_sum_buf[q_count % 2])
        pl.cast(o_f16, running_o, mode=pl.RoundMode.CAST_ROUND)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
        # Only real rows are stored. For sub_id with real_rows==0 (tiny tails),
        # this becomes a 0-row store and writes nothing to o.
        pl.set_validshape(o_f16, [real_rows, TD])
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
    qk_buf: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],  # FIFO_SIZE x Sq rows (DN: shape same)
    p_buf: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],  # FIFO_SIZE x Sq rows
    pv_buf: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):

    sq_dim = q.shape[0]
    skv_dim = k.shape[0]
    sq_tiles = (sq_dim + (TS - 1)) // TS
    skv_tiles = (skv_dim + (TKV - 1)) // TKV
    num_cores = pl.get_block_num()
    core_id = pl.get_block_idx() // pl.get_subblock_num()

    # qk_vec: pad=min so fillpad fills row-padding [actual_skv, TKV) with FP32 min,
    # making col_max ignore those rows and col_sum see exp(min*SCALE)~=0.
    qk_vec = pl.make_tile(
        pl.TileType(
            shape=[TKV, TS_HALF],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Vec,
            valid_shape=[-1, -1],
            compact=1,
            pad=pl.TilePad.min,
        ),
        addr=VA0,
        size=VB4_KV,
    )
    qk_vec1 = pl.make_tile(
        pl.TileType(
            shape=[TKV, TS_HALF],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Vec,
            valid_shape=[-1, -1],
            compact=1,
            pad=pl.TilePad.min,
        ),
        addr=VA11,
        size=VB4_KV,
    )
    pv_vec = pl.make_tile(
        pl.TileType(
            shape=[TS_HALF, TD],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Vec,
            valid_shape=[-1, -1],
            compact=1,
        ),
        addr=VA8,
        size=VB4,
    )
    pv_vec1 = pl.make_tile(
        pl.TileType(
            shape=[TS_HALF, TD],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Vec,
            valid_shape=[-1, -1],
            compact=1,
        ),
        addr=VA12,
        size=VB4,
    )
    running_o = pl.make_tile(
        pl.TileType(
            shape=[TS_HALF, TD],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Vec,
            valid_shape=[-1, -1],
            compact=1,
        ),
        addr=VA7,
        size=VB4,
    )

    # DN: p_mat with layout=pl.ZN -- Left format for P x V matmul
    p_mat_type = pl.TileType(
        shape=[TS, TKV],
        dtype=pl.DT_FP16,
        target_memory=pl.MemorySpace.Mat,
        layout=pl.ZN,
        valid_shape=[-1, -1],
        compact=1,
    )
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
        ctx_arr = pl.struct_array(
            3,
            "CubeCtx",
            sq_off=0,
            task_id=0,
            ki=0,
            q_count=0,
            skv_off=0,
            buf_idx=0,
            actual_skv=0,
            actual_sq=0,
            actual_sq_pad=0,
            actual_sq_half=0,
        )
        for qi in pl.range(core_id, sq_tiles, num_cores):
            sq_off = qi * TS
            actual_sq = pl.min(sq_dim - sq_off, TS)
            # Pad to multiple of 32 -> each dual_split half (= pad/2) is 16-element
            # aligned, satisfying fixpipe's N alignment requirement (otherwise FIX
            # raises an address misalign exception).
            actual_sq_pad = align32byte_fp32(actual_sq)
            actual_sq_half = actual_sq_pad // 2
            for ki in pl.range(0, skv_tiles):
                skv_off = ki * TKV
                buf_idx = (q_count * skv_tiles + ki) % 2
                actual_skv = pl.min(skv_dim - skv_off, TKV)
                ctx_curr = ctx_arr[task_id % 3]
                ctx_curr.sq_off = sq_off
                ctx_curr.task_id = task_id
                ctx_curr.ki = ki
                ctx_curr.q_count = q_count
                ctx_curr.skv_off = skv_off
                ctx_curr.buf_idx = buf_idx
                ctx_curr.actual_skv = actual_skv
                ctx_curr.actual_sq = actual_sq
                ctx_curr.actual_sq_pad = actual_sq_pad
                ctx_curr.actual_sq_half = actual_sq_half
                compute_qk(
                    task_id,
                    ki,
                    q_count,
                    sq_off,
                    skv_off,
                    buf_idx,
                    actual_skv,
                    actual_sq,
                    actual_sq_pad,
                    actual_sq_half,
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

                ctx_pre = ctx_arr[(task_id + 2) % 3]
                if task_id > 0:
                    compute_pv(
                        ctx_pre.task_id,
                        ctx_pre.ki,
                        ctx_pre.q_count,
                        ctx_pre.skv_off,
                        ctx_pre.buf_idx,
                        ctx_pre.actual_skv,
                        ctx_pre.actual_sq_pad,
                        ctx_pre.actual_sq_half,
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
                task_id = task_id + 1
            q_count = q_count + 1

        ctx_last = ctx_arr[(task_id + 2) % 3]
        compute_pv(
            ctx_last.task_id,
            ctx_last.ki,
            ctx_last.q_count,
            ctx_last.skv_off,
            ctx_last.buf_idx,
            ctx_last.actual_skv,
            ctx_last.actual_sq_pad,
            ctx_last.actual_sq_half,
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
            pl.TileType(
                shape=[TKV, TS_HALF],
                dtype=pl.DT_FP32,
                target_memory=pl.MemorySpace.Vec,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addr=VA1,
            size=VB4_KV,
        )
        p_f16 = pl.make_tile(
            pl.TileType(
                shape=[TKV, TS_HALF],
                dtype=pl.DT_FP16,
                target_memory=pl.MemorySpace.Vec,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addr=VA2,
            size=VB2_KV,
        )
        reduce_dst_rm = pl.make_tile(
            pl.TileType(
                shape=[1, TS_HALF],
                dtype=pl.DT_FP32,
                target_memory=pl.MemorySpace.Vec,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addr=VA3,
            size=VB_RED,
        )

        red_type = pl.TileType(
            shape=[TS_HALF, 1],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Vec,
            layout=pl.DN,
            valid_shape=[-1, -1],
            compact=1,
        )
        red_rm_type = pl.TileType(
            shape=[1, TS_HALF], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1], compact=1
        )

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
            pl.TileType(
                shape=[TS_HALF, TD],
                dtype=pl.DT_FP16,
                target_memory=pl.MemorySpace.Vec,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addr=VA9,
            size=VB2,
        )

        tile_type_nz = pl.TileType(
            shape=[TKV, TS_HALF],
            dtype=pl.DT_FP16,
            target_memory=pl.MemorySpace.Vec,
            valid_shape=[-1, -1],
            layout=pl.NZ,
            compact=1,
        )
        tile_nz = pl.make_tile(tile_type_nz, addr=VA10, size=VB6_DN)

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

        task_id = 0
        q_count = 0
        sub_id = pl.get_subblock_idx()
        ctx_arr = pl.struct_array(
            3,
            "VecCtx",
            sq_off=0,
            task_id=0,
            ki=0,
            skv_tiles=0,
            q_count=0,
            row_off=0,
            buf_idx=0,
            actual_skv=0,
            actual_sq_half=0,
            real_rows=0,
        )

        for qi in pl.range(core_id, sq_tiles, num_cores):
            sq_off = qi * TS
            actual_sq = pl.min(sq_dim - sq_off, TS)
            # Pad to multiple of 32 -> each dual_split half (pad/2) is 16-element
            # aligned, satisfying fixpipe's N alignment requirement.
            actual_sq_pad = align32byte_fp32(actual_sq)
            actual_sq_half = actual_sq_pad // 2
            # Real rows per sub_id:
            sub0_real_rows = pl.min(actual_sq, actual_sq_half)
            sub1_real_rows = actual_sq - sub0_real_rows
            real_rows = (1 - sub_id) * sub0_real_rows + sub_id * sub1_real_rows
            row_off = sub_id * actual_sq_half

            for ki in pl.range(0, skv_tiles):
                skv_off = ki * TKV
                buf_idx = (q_count * skv_tiles + ki) % 2
                actual_skv = pl.min(skv_dim - skv_off, TKV)
                ctx_curr = ctx_arr[task_id % 3]
                ctx_curr.sq_off = sq_off
                ctx_curr.task_id = task_id
                ctx_curr.ki = ki
                ctx_curr.skv_tiles = skv_tiles
                ctx_curr.q_count = q_count
                ctx_curr.row_off = row_off
                ctx_curr.buf_idx = buf_idx
                ctx_curr.actual_skv = actual_skv
                ctx_curr.actual_sq_half = actual_sq_half
                ctx_curr.real_rows = real_rows
                if task_id > 0:
                    ctx_p = ctx_arr[(task_id + 2) % 3]
                    compute_p(
                        ctx_p.task_id,
                        ctx_p.ki,
                        ctx_p.q_count,
                        ctx_p.buf_idx,
                        ctx_p.actual_skv,
                        ctx_p.actual_sq_half,
                        softmax_tiles,
                        p_mat_buf1,
                        p_mat_buf2,
                    )
                if task_id > 1:
                    ctx_gu = ctx_arr[(task_id + 1) % 3]
                    compute_gu(
                        ctx_gu.task_id,
                        ctx_gu.ki,
                        ctx_gu.skv_tiles,
                        ctx_gu.sq_off,
                        ctx_gu.row_off,
                        ctx_gu.q_count,
                        ctx_gu.actual_sq_half,
                        ctx_gu.real_rows,
                        o,
                        gu_tiles,
                    )
                task_id = task_id + 1
            q_count = q_count + 1

        ctx_p = ctx_arr[(task_id + 2) % 3]
        compute_p(
            ctx_p.task_id,
            ctx_p.ki,
            ctx_p.q_count,
            ctx_p.buf_idx,
            ctx_p.actual_skv,
            ctx_p.actual_sq_half,
            softmax_tiles,
            p_mat_buf1,
            p_mat_buf2,
        )
        if task_id > 1:
            ctx_gu = ctx_arr[(task_id + 1) % 3]
            compute_gu(
                ctx_gu.task_id,
                ctx_gu.ki,
                ctx_gu.skv_tiles,
                ctx_gu.sq_off,
                ctx_gu.row_off,
                ctx_gu.q_count,
                ctx_gu.actual_sq_half,
                ctx_gu.real_rows,
                o,
                gu_tiles,
            )
        task_id = task_id + 1
        ctx_gu = ctx_arr[(task_id + 1) % 3]
        compute_gu(
            ctx_gu.task_id,
            ctx_gu.ki,
            ctx_gu.skv_tiles,
            ctx_gu.sq_off,
            ctx_gu.row_off,
            ctx_gu.q_count,
            ctx_gu.actual_sq_half,
            ctx_gu.real_rows,
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
    pv = torch.matmul(x_exp, v.float())
    attn = (pv / torch.sum(x_exp, dim=-1, keepdim=True)).half()
    return qk, x_exp, pv, attn


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_fa_perf():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    device = f"npu:{device_id}"
    torch.manual_seed(42)
    for sq, skv, d, num_cores in [
        (34, 62, TD, 1),
        (162, 190, TD, 1),
        (8192, 8192, TD, 28),
        (8128, 8128, TD, 28),
        (8111, 7777, TD, 28),
    ]:
        logging.info("\nFA-Perf DN (%s,%s,%s) cores=%s  QK_PRELOAD=%s", sq, skv, d, num_cores, QK_PRELOAD)
        q_t = torch.rand((sq, d), device=device, dtype=torch.float16)
        k_t = torch.rand((skv, d), device=device, dtype=torch.float16)
        v_t = torch.rand((skv, d), device=device, dtype=torch.float16)
        o_t = torch.zeros((sq, d), device=device, dtype=torch.float16)
        qk_t = torch.zeros((sq * FIFO_SIZE, skv), device=device, dtype=torch.float32)
        p_t = torch.zeros((sq * FIFO_SIZE, skv), device=device, dtype=torch.float16)
        pv_t = torch.zeros((sq * FIFO_SIZE, d), device=device, dtype=torch.float32)
        for _ in range(20):
            fa_perf_tkv_preload_dn_kernel[None, num_cores](q_t, k_t, v_t, o_t, qk_t, p_t, pv_t)
            torch.npu.synchronize()
            qk_ref, x_exp_ref, pv_ref, o_ref = flash_attention_ref(q_t, k_t, v_t, d)
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
