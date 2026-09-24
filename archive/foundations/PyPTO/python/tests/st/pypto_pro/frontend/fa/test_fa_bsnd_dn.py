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
#  Tile dimensions and constants
# ================================================================
TS = 128
TKV = 128
TD = 128
TS_HALF = TS // 2
SCALE = 1.0 / math.sqrt(TD)
FIFO_SIZE = 2

# Cube tile byte sizes
Q_F16 = TS * TD * 2  # [TS,  TD]  FP16 = 32KB (DN: stored as [TD, TS])
KT_F16 = TKV * TD * 2  # [TKV, TD]  FP16 = 32KB
V_F16 = TKV * TD * 2  # [TKV, TD]  FP16 = 32KB
P_F16 = TS * TKV * 2  # [TS,  TKV] FP16 = 32KB
QK_F32 = TKV * TS * 4  # [TKV, TS]  FP32 = 64KB (DN acc shape)
PV_F32 = TS * TD * 4  # [TS,  TD]  FP32 = 64KB
PV_CORE_STRIDE = 2 * FIFO_SIZE * TS

# ---- MAT (512KB) ----
MA0 = 0
MA0_PONG = MA0 + Q_F16
MA1 = Q_F16 * 2
MA1_PONG = MA1 + KT_F16
MA2 = MA1 + KT_F16 * 2
MA2_PONG = MA2 + P_F16
MA3 = MA2 + P_F16 * 2
MA3_PONG = MA3 + V_F16

# DN: Left holds K [TKV, TD], Right holds Q^T [TD, TS].
LA0 = 0
LA1 = KT_F16
RA0 = 0
RA1 = Q_F16
CA0 = 0
CA1 = QK_F32

# ---- VEC addresses (248KB on a5) ----
VB4_KV = TKV * TS_HALF * 4  # [TKV, TS_HALF] FP32 = 32KB
VB2_KV = TKV * TS_HALF * 2  # [TKV, TS_HALF] FP16 = 16KB
VB6_DN = (TKV + 1) * TS_HALF * 2  # [TKV+1, TS_HALF] FP16 = 16512B
VB1_KV = TKV * TS_HALF  # [TKV, TS_HALF] UINT8 = 8KB
VB4 = TS_HALF * TD * 4  # [TS_HALF, TD] FP32 = 32KB
VB2 = TS_HALF * TD * 2  # [TS_HALF, TD] FP16 = 16KB
VB_RED = TS_HALF * 4  # [TS_HALF, 1] FP32 = 256B


def _align_up(value, align=1024):
    return ((value + align - 1) // align) * align


VA0 = 0  # qk_vec  [TKV, TS_HALF] FP32
VA1 = _align_up(VA0 + VB4_KV)  # tmp_vec [TKV, TS_HALF] FP32
VA2 = _align_up(VA1 + VB4_KV)  # p_f16   [TKV, TS_HALF] FP16
VA3 = _align_up(VA2 + VB2_KV)  # reduce_dst / reduce_dst_rm
# Keep the whole VEC layout on 1KB boundaries; A5 CCE is stricter about
# VEC address alignment than the older A3 path.
VA_GMAX0 = _align_up(VA3 + VB_RED)  # global_max slot 0
VA_GMAX1 = _align_up(VA_GMAX0 + VB_RED)  # global_max slot 1
VA_GSUM0 = _align_up(VA_GMAX1 + VB_RED)  # global_sum slot 0
VA_GSUM1 = _align_up(VA_GSUM0 + VB_RED)  # global_sum slot 1
VA_EXP0 = _align_up(VA_GSUM1 + VB_RED)  # exp_corr slot 0
VA_EXP1 = _align_up(VA_EXP0 + VB_RED)  # exp_corr slot 1
VA7 = _align_up(VA_EXP1 + VB_RED)  # running_o [TS_HALF, TD] FP32
VA8 = _align_up(VA7 + VB4)  # pv_vec    [TS_HALF, TD] FP32
VA9 = _align_up(VA8 + VB4)  # o_f16     [TS_HALF, TD] FP16
VA10 = _align_up(VA9 + VB2)  # tile_nz   [TKV+1, TS_HALF] FP16
VA11 = _align_up(VA10 + VB6_DN)  # qk_vec1   [TKV, TS_HALF] FP32
VA12 = _align_up(VA11 + VB4_KV)  # pv_vec1   [TS_HALF, TD] FP32
assert VA12 + VB4 <= 248 * 1024, f"VEC overflow: {VA12 + VB4} > {248 * 1024}"

EVENT_IDS_01 = (0, 1)
EVENT_IDS_23 = (2, 3)
QK_READY_IDS = (0, 1)
P_READY_IDS = (2, 3)
PV_READY_IDS = (4, 5)


def compute_qk(
    b_idx: pl.DT_INT64,
    n_idx: pl.DT_INT64,
    qi: pl.DT_INT64,
    ki: pl.DT_INT64,
    skv_tiles: pl.DT_INT64,
    q_count: pl.DT_INT64,
    task_id: pl.DT_INT64,
    l0ab_idx: pl.DT_INT64,
    l0c_idx: pl.DT_INT64,
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    cube,
) -> None:
    q_mat_buf = cube.q_mat_buf
    k_mat_buf = cube.k_mat_buf
    left_buf = cube.left_buf
    right_buf = cube.right_buf
    acc_buf = (cube.acc_buf1, cube.acc_buf2)
    qk_vec = cube.qk_vec
    qk_vec1 = cube.qk_vec1
    qk_slot = task_id % FIFO_SIZE
    buf_idx = (q_count * skv_tiles + ki) % 2
    pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=EVENT_IDS_01[buf_idx])
    if ki == 0:
        pl.load_tile(q_mat_buf[q_count % 2], q, [b_idx, qi, n_idx, 0], order=[3, 1])
    pl.load_tile(k_mat_buf[buf_idx], k, [b_idx, ki, n_idx, 0], order=[1, 3])
    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)

    pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=EVENT_IDS_01[l0ab_idx])
    pl.move(left_buf[l0ab_idx], k_mat_buf[buf_idx])
    pl.move(right_buf[l0ab_idx], q_mat_buf[q_count % 2])
    pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
    pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
    pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=EVENT_IDS_01[buf_idx])

    pl.system.sync_dst(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=EVENT_IDS_01[l0c_idx])
    pl.matmul(acc_buf[l0c_idx], left_buf[l0ab_idx], right_buf[l0ab_idx])
    pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
    pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
    pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=EVENT_IDS_01[l0ab_idx])

    if qk_slot == 0:
        pl.move(qk_vec, acc_buf[l0c_idx], acc_to_vec_mode=pl.AccToVecMode.DualModeSplitN)
    else:
        pl.move(qk_vec1, acc_buf[l0c_idx], acc_to_vec_mode=pl.AccToVecMode.DualModeSplitN)
    pl.system.sync_src(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=EVENT_IDS_01[l0c_idx])
    pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=QK_READY_IDS[qk_slot])
    return


def compute_pv(
    b_idx: pl.DT_INT64,
    n_idx: pl.DT_INT64,
    ki: pl.DT_INT64,
    skv_tiles: pl.DT_INT64,
    q_count: pl.DT_INT64,
    task_id: pl.DT_INT64,
    l0ab_idx: pl.DT_INT64,
    l0c_idx: pl.DT_INT64,
    v: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    cube,
) -> None:
    pv_vec = cube.pv_vec
    pv_vec1 = cube.pv_vec1
    p_mat_buf = (cube.p_mat_buf1, cube.p_mat_buf2)
    v_mat_buf = cube.v_mat_buf
    left_buf = cube.left_buf
    right_buf = cube.right_buf
    acc_buf = (cube.acc_buf1, cube.acc_buf2)
    pv_slot = task_id % FIFO_SIZE
    buf_idx = (q_count * skv_tiles + ki) % 2
    pl.system.wait_cross_core(pipe=pl.PipeType.MTE1, event_id=P_READY_IDS[pv_slot])
    pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=EVENT_IDS_23[buf_idx])
    pl.load_tile(v_mat_buf[buf_idx], v, [b_idx, ki, n_idx, 0], order=[1, 3])
    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)

    pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=EVENT_IDS_01[l0ab_idx])
    pl.move(left_buf[l0ab_idx], p_mat_buf[buf_idx])
    pl.move(right_buf[l0ab_idx], v_mat_buf[buf_idx])
    pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=EVENT_IDS_23[buf_idx])
    pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
    pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)

    pl.system.sync_dst(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=EVENT_IDS_01[l0c_idx])
    pl.matmul(acc_buf[l0c_idx], left_buf[l0ab_idx], right_buf[l0ab_idx])
    pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
    pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
    pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=EVENT_IDS_01[l0ab_idx])

    if pv_slot == 0:
        pl.move(pv_vec, acc_buf[l0c_idx], acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
    else:
        pl.move(pv_vec1, acc_buf[l0c_idx], acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
    pl.system.sync_src(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=EVENT_IDS_01[l0c_idx])
    pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=PV_READY_IDS[pv_slot])
    return


def softmax_body(
    b_idx: pl.DT_INT64,
    n_idx: pl.DT_INT64,
    qi: pl.DT_INT64,
    ki: pl.DT_INT64,
    skv_tiles: pl.DT_INT64,
    q_count: pl.DT_INT64,
    sub_id: pl.DT_INT64,
    task_id: pl.DT_INT64,
    sq_dim: pl.DT_INT64,
    skv_dim: pl.DT_INT64,
    stiles,
    p_buf: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
) -> None:
    qk_vec = stiles.qk_vec
    qk_vec1 = stiles.qk_vec1
    tmp_vec = stiles.tmp_vec
    p_f16 = stiles.p_f16
    reduce_dst_rm = stiles.reduce_dst_rm
    global_max_rm_buf = stiles.global_max_rm_buf
    global_sum_rm_buf = stiles.global_sum_rm_buf
    exp_corr_rm = stiles.exp_corr_rm
    exp_corr_rm1 = stiles.exp_corr_rm1
    tile_nz = stiles.tile_nz
    p_mat_buf1 = stiles.p_mat_buf1
    p_mat_buf2 = stiles.p_mat_buf2
    p_f16_store = stiles.p_f16_store
    p_slot = task_id % FIFO_SIZE
    q_idx = q_count % 2
    global_max_rm = global_max_rm_buf[q_idx]
    global_sum_rm = global_sum_rm_buf[q_idx]
    buf_idx = (q_count * skv_tiles + ki) % 2
    if p_slot == 0:
        if ki == 0:
            pl.maximum(reduce_dst_rm, qk_vec, tmp_vec, dim=1)
            pl.expand_sub(tmp_vec, qk_vec, reduce_dst_rm, dim=1)
            pl.mul(global_max_rm, reduce_dst_rm, 1.0)
            pl.mul(tmp_vec, tmp_vec, SCALE)
            pl.exp(qk_vec, tmp_vec)
            pl.sum(reduce_dst_rm, qk_vec, tmp_vec, dim=1)
            pl.mul(global_sum_rm, reduce_dst_rm, 1.0)
            pl.cast(p_f16, qk_vec, mode=pl.RoundMode.CAST_ROUND)
        else:
            pl.maximum(reduce_dst_rm, qk_vec, tmp_vec, dim=1)
            pl.maximum(reduce_dst_rm, reduce_dst_rm, global_max_rm)
            pl.sub(exp_corr_rm, global_max_rm, reduce_dst_rm)
            pl.mul(global_max_rm, reduce_dst_rm, 1.0)
            pl.expand_sub(tmp_vec, qk_vec, reduce_dst_rm, dim=1)
            pl.mul(exp_corr_rm, exp_corr_rm, SCALE)
            pl.mul(tmp_vec, tmp_vec, SCALE)
            pl.exp(exp_corr_rm, exp_corr_rm)
            pl.exp(qk_vec, tmp_vec)
            pl.cast(p_f16, qk_vec, mode=pl.RoundMode.CAST_ROUND)
            pl.mul(global_sum_rm, global_sum_rm, exp_corr_rm)
            pl.sum(reduce_dst_rm, qk_vec, tmp_vec, dim=1)
            pl.add(global_sum_rm, global_sum_rm, reduce_dst_rm)
    else:
        if ki == 0:
            pl.maximum(reduce_dst_rm, qk_vec1, tmp_vec, dim=1)
            pl.expand_sub(tmp_vec, qk_vec1, reduce_dst_rm, dim=1)
            pl.mul(global_max_rm, reduce_dst_rm, 1.0)
            pl.mul(tmp_vec, tmp_vec, SCALE)
            pl.exp(qk_vec1, tmp_vec)
            pl.sum(reduce_dst_rm, qk_vec1, tmp_vec, dim=1)
            pl.mul(global_sum_rm, reduce_dst_rm, 1.0)
            pl.cast(p_f16, qk_vec1, mode=pl.RoundMode.CAST_ROUND)
        else:
            pl.maximum(reduce_dst_rm, qk_vec1, tmp_vec, dim=1)
            pl.maximum(reduce_dst_rm, reduce_dst_rm, global_max_rm)
            pl.sub(exp_corr_rm1, global_max_rm, reduce_dst_rm)
            pl.mul(global_max_rm, reduce_dst_rm, 1.0)
            pl.expand_sub(tmp_vec, qk_vec1, reduce_dst_rm, dim=1)
            pl.mul(exp_corr_rm1, exp_corr_rm1, SCALE)
            pl.mul(tmp_vec, tmp_vec, SCALE)
            pl.exp(exp_corr_rm1, exp_corr_rm1)
            pl.exp(qk_vec1, tmp_vec)
            pl.cast(p_f16, qk_vec1, mode=pl.RoundMode.CAST_ROUND)
            pl.mul(global_sum_rm, global_sum_rm, exp_corr_rm1)
            pl.sum(reduce_dst_rm, qk_vec1, tmp_vec, dim=1)
            pl.add(global_sum_rm, global_sum_rm, reduce_dst_rm)

    pl.move(tile_nz, p_f16)
    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
    if buf_idx == 0:
        pl.insert(p_mat_buf1, tile_nz, [0, TS_HALF * sub_id])
    else:
        pl.insert(p_mat_buf2, tile_nz, [0, TS_HALF * sub_id])
    pl.store_tile(p_buf, p_f16_store, [b_idx, qi * 2 + sub_id, n_idx, ki], order=[1, 3])


def compute_p(
    b_idx: pl.DT_INT64,
    n_idx: pl.DT_INT64,
    qi: pl.DT_INT64,
    ki: pl.DT_INT64,
    skv_tiles: pl.DT_INT64,
    q_count: pl.DT_INT64,
    sub_id: pl.DT_INT64,
    task_id: pl.DT_INT64,
    sq_dim: pl.DT_INT64,
    skv_dim: pl.DT_INT64,
    stiles,
    p_buf: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
) -> None:
    p_slot = task_id % FIFO_SIZE
    pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=QK_READY_IDS[p_slot])
    softmax_body(
        b_idx,
        n_idx,
        qi,
        ki,
        skv_tiles,
        q_count,
        sub_id,
        task_id,
        sq_dim,
        skv_dim,
        stiles,
        p_buf,
    )
    # Both vector subblocks contribute one half of the shared P MAT tile. Make
    # sure both halves have finished TINSERT before cube starts PV.
    pl.system.bar_all()
    pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=P_READY_IDS[p_slot])


def compute_gu(
    b_idx: pl.DT_INT64,
    n_idx: pl.DT_INT64,
    qi: pl.DT_INT64,
    ki: pl.DT_INT64,
    skv_tiles: pl.DT_INT64,
    q_count: pl.DT_INT64,
    sub_id: pl.DT_INT64,
    task_id: pl.DT_INT64,
    o: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    gtiles,
) -> None:
    pv_vec = gtiles.pv_vec
    pv_vec1 = gtiles.pv_vec1
    running_o = gtiles.running_o
    exp_corr = gtiles.exp_corr
    exp_corr1 = gtiles.exp_corr1
    global_sum_buf = gtiles.global_sum_buf
    o_f16 = gtiles.o_f16
    pv_slot = task_id % FIFO_SIZE
    pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=PV_READY_IDS[pv_slot])
    if pv_slot == 0:
        if ki == 0:
            pl.move(running_o, pv_vec)
        else:
            pl.expand_mul(running_o, running_o, exp_corr)
            pl.add(running_o, running_o, pv_vec)
    else:
        if ki == 0:
            pl.move(running_o, pv_vec1)
        else:
            pl.expand_mul(running_o, running_o, exp_corr1)
            pl.add(running_o, running_o, pv_vec1)
    if ki == skv_tiles - 1:
        pl.expand_div(running_o, running_o, global_sum_buf[q_count % 2])
        pl.cast(o_f16, running_o, mode=pl.RoundMode.CAST_ROUND)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
        pl.store_tile(o, o_f16, [b_idx, qi * 2 + sub_id, n_idx, 0], order=[1, 3])


@pl.jit()
def fa_bsnd_dn_kernel(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    v: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    o: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    p_buf: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    pv_buf: pl.Tensor[[48 * PV_CORE_STRIDE, pl.DYNAMIC], pl.DT_FP32],
    work_ranges: pl.Tensor[[pl.DYNAMIC, 2], pl.DT_INT32],
):
    sq_dim = q.shape[1]
    skv_dim = k.shape[1]
    sq_tiles = (sq_dim + TS - 1) // TS
    skv_tiles = (skv_dim + TKV - 1) // TKV
    core_id = pl.get_block_idx() // pl.get_subblock_num()
    n_dim = q.shape[2]

    qk_vec = pl.make_tile(
        pl.TileType(shape=[TKV, TS_HALF], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addr=VA0, size=VB4_KV
    )
    tmp_vec = pl.make_tile(
        pl.TileType(shape=[TKV, TS_HALF], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addr=VA1, size=VB4_KV
    )
    # Keep the source descriptor identical to the proven DN perf kernel.
    # TMOV(tile_nz, p_f16) is sensitive to the Vec tile descriptor here.
    p_f16 = pl.make_tile(
        pl.TileType(shape=[TKV, TS_HALF], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec), addr=VA2, size=VB2_KV
    )
    p_f16_store = pl.make_tile(
        pl.TileType(shape=[TS_HALF, TKV], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec), addr=VA2, size=VB2_KV
    )
    reduce_dst_rm = pl.make_tile(
        pl.TileType(shape=[1, TS_HALF], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addr=VA3, size=VB_RED
    )
    red_rm_type = pl.TileType(shape=[1, TS_HALF], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    red_type = pl.TileType(shape=[TS_HALF, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.DN)
    global_max_rm_0 = pl.make_tile(red_rm_type, addr=VA_GMAX0, size=VB_RED)
    global_max_rm_1 = pl.make_tile(red_rm_type, addr=VA_GMAX1, size=VB_RED)
    global_max_rm_buf = (global_max_rm_0, global_max_rm_1)  # noqa: F841
    global_sum_0 = pl.make_tile(red_type, addr=VA_GSUM0, size=VB_RED)
    global_sum_1 = pl.make_tile(red_type, addr=VA_GSUM1, size=VB_RED)
    global_sum_buf = (global_sum_0, global_sum_1)  # noqa: F841
    global_sum_rm_0 = pl.make_tile(red_rm_type, addr=VA_GSUM0, size=VB_RED)
    global_sum_rm_1 = pl.make_tile(red_rm_type, addr=VA_GSUM1, size=VB_RED)
    global_sum_rm_buf = (global_sum_rm_0, global_sum_rm_1)  # noqa: F841
    exp_corr = pl.make_tile(red_type, addr=VA_EXP0, size=VB_RED)
    exp_corr_rm = pl.make_tile(red_rm_type, addr=VA_EXP0, size=VB_RED)
    exp_corr1 = pl.make_tile(red_type, addr=VA_EXP1, size=VB_RED)
    exp_corr_rm1 = pl.make_tile(red_rm_type, addr=VA_EXP1, size=VB_RED)
    running_o = pl.make_tile(
        pl.TileType(shape=[TS_HALF, TD], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addr=VA7, size=VB4
    )
    pv_vec = pl.make_tile(
        pl.TileType(shape=[TS_HALF, TD], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addr=VA8, size=VB4
    )
    o_f16 = pl.make_tile(
        pl.TileType(shape=[TS_HALF, TD], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec), addr=VA9, size=VB2
    )
    tile_nz = pl.make_tile(
        pl.TileType(shape=[TKV, TS_HALF], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec, layout=pl.NZ),
        addr=VA10,
        size=VB6_DN,
    )
    qk_vec1 = pl.make_tile(
        pl.TileType(shape=[TKV, TS_HALF], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addr=VA11, size=VB4_KV
    )
    pv_vec1 = pl.make_tile(
        pl.TileType(shape=[TS_HALF, TD], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addr=VA12, size=VB4
    )
    p_mat_type = pl.TileType(shape=[TS, TKV], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN)
    p_mat_buf1 = pl.make_tile(p_mat_type, addr=MA2, size=P_F16)
    p_mat_buf2 = pl.make_tile(p_mat_type, addr=MA2_PONG, size=P_F16)

    with pl.section_cube():
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
            pl.TileType(shape=[TD, TS], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right), addr=RA0, size=Q_F16
        )
        right_1 = pl.make_tile(
            pl.TileType(shape=[TD, TS], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right), addr=RA1, size=Q_F16
        )
        acc_buf1 = pl.make_tile(
            pl.TileType(shape=[TKV, TS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc), addr=CA0, size=QK_F32
        )
        acc_buf2 = pl.make_tile(
            pl.TileType(shape=[TS, TD], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc), addr=CA1, size=PV_F32
        )
        cube_tiles = pl.make_tuple(
            q_mat_buf=(q_mat_0, q_mat_1),
            k_mat_buf=(k_mat_0, k_mat_1),
            v_mat_buf=(v_mat_0, v_mat_1),
            left_buf=(left_0, left_1),
            right_buf=(right_0, right_1),
            acc_buf1=acc_buf1,
            acc_buf2=acc_buf2,
            qk_vec=qk_vec,
            qk_vec1=qk_vec1,
            pv_vec=pv_vec,
            pv_vec1=pv_vec1,
            p_mat_buf1=p_mat_buf1,
            p_mat_buf2=p_mat_buf2,
        )
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=0)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=1)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=2)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=3)
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=1)
        pl.system.sync_src(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=0)
        pl.system.sync_src(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=1)

        work_start = work_ranges[core_id, 0]
        work_end = work_ranges[core_id, 1]
        for work_id in pl.range(work_start, work_end):
            task_id = 0
            q_count = 0
            l0ab_idx = 0
            l0c_idx = 0
            b_idx = work_id // n_dim
            n_idx = work_id % n_dim
            ctx_arr = pl.struct_array(3, "CubeCtx", b_idx=0, n_idx=0, qi=0, ki=0, skv_tiles=0, q_count=0, task_id=0)
            for qi in pl.range(0, sq_tiles):
                for ki in pl.range(0, skv_tiles):
                    ctx_curr = ctx_arr[task_id % 3]
                    ctx_curr.b_idx = b_idx
                    ctx_curr.n_idx = n_idx
                    ctx_curr.qi = qi
                    ctx_curr.ki = ki
                    ctx_curr.skv_tiles = skv_tiles
                    ctx_curr.q_count = q_count
                    ctx_curr.task_id = task_id
                    compute_qk(
                        b_idx,
                        n_idx,
                        qi,
                        ki,
                        skv_tiles,
                        q_count,
                        task_id,
                        l0ab_idx,
                        l0c_idx,
                        q,
                        k,
                        cube_tiles,
                    )
                    l0ab_idx = 1 - l0ab_idx
                    l0c_idx = 1 - l0c_idx
                    if task_id > 0:
                        ctx_pre = ctx_arr[(task_id + 2) % 3]
                        compute_pv(
                            b_idx,
                            n_idx,
                            ctx_pre.ki,
                            ctx_pre.skv_tiles,
                            ctx_pre.q_count,
                            ctx_pre.task_id,
                            l0ab_idx,
                            l0c_idx,
                            v,
                            cube_tiles,
                        )
                        l0ab_idx = 1 - l0ab_idx
                        l0c_idx = 1 - l0c_idx
                    task_id = task_id + 1
                q_count = q_count + 1
            if task_id > 0:
                ctx_pre = ctx_arr[(task_id + 2) % 3]
                compute_pv(
                    b_idx,
                    n_idx,
                    ctx_pre.ki,
                    ctx_pre.skv_tiles,
                    ctx_pre.q_count,
                    ctx_pre.task_id,
                    l0ab_idx,
                    l0c_idx,
                    v,
                    cube_tiles,
                )
                l0ab_idx = 1 - l0ab_idx
                l0c_idx = 1 - l0c_idx

    with pl.section_vector():
        work_start = work_ranges[core_id, 0]
        work_end = work_ranges[core_id, 1]
        sub_id = pl.get_subblock_idx()
        softmax_tiles = pl.make_tuple(
            qk_vec=qk_vec,
            qk_vec1=qk_vec1,
            tmp_vec=tmp_vec,
            p_f16=p_f16,
            reduce_dst_rm=reduce_dst_rm,
            global_max_rm_buf=(global_max_rm_0, global_max_rm_1),
            global_sum_rm_buf=(global_sum_rm_0, global_sum_rm_1),
            exp_corr_rm=exp_corr_rm,
            exp_corr_rm1=exp_corr_rm1,
            tile_nz=tile_nz,
            p_mat_buf1=p_mat_buf1,
            p_mat_buf2=p_mat_buf2,
            p_f16_store=p_f16_store,
        )
        gu_tiles = pl.make_tuple(
            pv_vec=pv_vec,
            pv_vec1=pv_vec1,
            running_o=running_o,
            exp_corr=exp_corr,
            exp_corr1=exp_corr1,
            global_sum_buf=(global_sum_0, global_sum_1),
            o_f16=o_f16,
        )
        for work_id in pl.range(work_start, work_end):
            task_id = 0
            q_count = 0
            b_idx = work_id // n_dim
            n_idx = work_id % n_dim
            ctx_arr = pl.struct_array(
                3, "VecCtx", b_idx=0, n_idx=0, qi=0, ki=0, skv_tiles=0, q_count=0, sub_id=0, task_id=0
            )
            for qi in pl.range(0, sq_tiles):
                for ki in pl.range(0, skv_tiles):
                    ctx_curr = ctx_arr[task_id % 3]
                    ctx_curr.b_idx = b_idx
                    ctx_curr.n_idx = n_idx
                    ctx_curr.qi = qi
                    ctx_curr.ki = ki
                    ctx_curr.skv_tiles = skv_tiles
                    ctx_curr.q_count = q_count
                    ctx_curr.sub_id = sub_id
                    ctx_curr.task_id = task_id
                    if task_id > 0:
                        ctx_p = ctx_arr[(task_id + 2) % 3]
                        compute_p(
                            b_idx,
                            n_idx,
                            ctx_p.qi,
                            ctx_p.ki,
                            ctx_p.skv_tiles,
                            ctx_p.q_count,
                            sub_id,
                            ctx_p.task_id,
                            sq_dim,
                            skv_dim,
                            softmax_tiles,
                            p_buf,
                        )
                    if task_id > 1:
                        ctx_g = ctx_arr[(task_id + 1) % 3]
                        compute_gu(
                            b_idx,
                            n_idx,
                            ctx_g.qi,
                            ctx_g.ki,
                            ctx_g.skv_tiles,
                            ctx_g.q_count,
                            sub_id,
                            ctx_g.task_id,
                            o,
                            gu_tiles,
                        )
                    task_id = task_id + 1
                q_count = q_count + 1
            if task_id > 0:
                ctx_p = ctx_arr[(task_id + 2) % 3]
                compute_p(
                    b_idx,
                    n_idx,
                    ctx_p.qi,
                    ctx_p.ki,
                    ctx_p.skv_tiles,
                    ctx_p.q_count,
                    sub_id,
                    ctx_p.task_id,
                    sq_dim,
                    skv_dim,
                    softmax_tiles,
                    p_buf,
                )
                if task_id > 1:
                    ctx_g = ctx_arr[(task_id + 1) % 3]
                    compute_gu(
                        b_idx,
                        n_idx,
                        ctx_g.qi,
                        ctx_g.ki,
                        ctx_g.skv_tiles,
                        ctx_g.q_count,
                        sub_id,
                        ctx_g.task_id,
                        o,
                        gu_tiles,
                    )
                task_id = task_id + 1
                ctx_g2 = ctx_arr[(task_id + 1) % 3]
                compute_gu(
                    b_idx,
                    n_idx,
                    ctx_g2.qi,
                    ctx_g2.ki,
                    ctx_g2.skv_tiles,
                    ctx_g2.q_count,
                    sub_id,
                    ctx_g2.task_id,
                    o,
                    gu_tiles,
                )


def flash_attention_ref_bs(q, k, v, d):
    scale_val = 1.0 / math.sqrt(d)
    b, sq, n, _ = q.shape
    _, skv, _, _ = k.shape
    o_ref = torch.zeros_like(q)
    for bi in range(b):
        for ni in range(n):
            qk = torch.matmul(q[bi, :, ni, :].float(), k[bi, :, ni, :].float().T) * scale_val
            attn = torch.softmax(qk, dim=-1)
            o_ref[bi, :, ni, :] = torch.matmul(attn, v[bi, :, ni, :].float()).half()
    return o_ref


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_fa_bs_a5():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(42)
    for b, sq, n, skv, d, num_cores in [
        (1, 128, 1, 128, TD, 1),
        (1, 128, 2, 128, TD, 2),
        (2, 1024, 3, 1024, TD, 24),
        (1, 8192, 2, 8192, TD, 24),
        (8, 256, 2, 256, TD, 6),
        (2, 512, 1, 256, TD, 4),
    ]:
        logging.info("\nFA-BSND-DN-A5 (b=%s, sq=%s, n=%s, skv=%s, d=%s) cores=%s", b, sq, n, skv, d, num_cores)
        q = torch.rand((b, sq, n, d), device=device, dtype=torch.float16)
        k = torch.rand((b, skv, n, d), device=device, dtype=torch.float16)
        v = torch.rand((b, skv, n, d), device=device, dtype=torch.float16)
        o = torch.zeros((b, sq, n, d), device=device, dtype=torch.float16)
        p_buf = torch.zeros((b, sq, n, skv), device=device, dtype=torch.float16)
        pv_buf = torch.zeros((48 * PV_CORE_STRIDE, d), device=device, dtype=torch.float32)
        total_work = b * n
        work_ranges = torch.zeros((num_cores, 2), device=device, dtype=torch.int32)
        work_per_core = (total_work + num_cores - 1) // num_cores
        for core in range(num_cores):
            work_ranges[core, 0] = core * work_per_core
            work_ranges[core, 1] = min((core + 1) * work_per_core, total_work)
        actual_num_cores = min(num_cores, total_work)
        fa_bsnd_dn_kernel[None, actual_num_cores](q, k, v, o, p_buf, pv_buf, work_ranges)
        torch.npu.synchronize()
        o_ref = flash_attention_ref_bs(q, k, v, d)
        diff = (o - o_ref).abs().max().item()
        logging.info("  max|diff|=%.4f", diff)
        torch.testing.assert_close(o, o_ref, rtol=5e-3, atol=5e-3)
        logging.info("  PASS")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.info("FA BSND DN on A5 CCE")
    logging.info("%s", '=' * 60)
    test_fa_bs_a5()
    logging.info("\nAll FlashAttention DN tests passed!")
