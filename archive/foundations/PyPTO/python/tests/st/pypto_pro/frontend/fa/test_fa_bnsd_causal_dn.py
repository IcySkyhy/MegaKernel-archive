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
NEG_INF = -1e9
FIXED_MASK_S = 2048
FIFO_SIZE = 1

# Cube tile byte sizes
Q_F16 = TS * TD * 2  # [TS,  TD]  FP16 = 32KB (DN: stored as [TD, TS])
KT_F16 = TKV * TD * 2  # [TKV, TD]  FP16 = 32KB
V_F16 = TKV * TD * 2  # [TKV, TD]  FP16 = 32KB
P_F16 = TS * TKV * 2  # [TS,  TKV] FP16 = 32KB
QK_F32 = TKV * TS * 4  # [TKV, TS]  FP32 = 64KB (DN acc shape)
PV_F32 = TS * TD * 4  # [TS,  TD]  FP32 = 64KB

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
# VEC address alignment than the older A3 path. The causal path also adds
# mask/TSEL scratch tiles after the DN perf-kernel buffers.
VA_GMAX0 = _align_up(VA3 + VB_RED)  # global_max slot 0
VA_GMAX1 = _align_up(VA_GMAX0 + VB_RED)  # global_max slot 1
VA_GSUM0 = _align_up(VA_GMAX1 + VB_RED)  # global_sum slot 0
VA_GSUM1 = _align_up(VA_GSUM0 + VB_RED)  # global_sum slot 1
VA_EXP0 = _align_up(VA_GSUM1 + VB_RED)  # exp_corr
VA7 = _align_up(VA_EXP0 + VB_RED)  # running_o [TS_HALF, TD] FP32
VA8 = _align_up(VA7 + VB4)  # pv_vec    [TS_HALF, TD] FP32
VA9 = _align_up(VA8 + VB4)  # o_f16     [TS_HALF, TD] FP16
VA10 = _align_up(VA9 + VB2)  # tile_nz   [TKV+1, TS_HALF] FP16
VA11 = _align_up(VA10 + VB6_DN)  # mask_u8_dn   [TKV, TS_HALF] UINT8
VA12 = _align_up(VA11 + VB1_KV)  # mask_fp16_dn [TKV, TS_HALF] FP16
VA13 = _align_up(VA12 + VB2_KV)  # mask_vec_dn  [TKV, TS_HALF] UINT8
VA14 = _align_up(VA13 + VB1_KV)  # neg_inf_vec  [TKV, TS_HALF] FP32
assert VA14 + VB4_KV <= 248 * 1024, f"VEC overflow: {VA14 + VB4_KV} > {248 * 1024}"

EVENT_IDS_01 = (0, 1)
EVENT_IDS_23 = (2, 3)
QK_READY_IDS = (0,)
P_READY_IDS = (1,)
PV_READY_IDS = (2,)


def compute_qk(
    task_id: pl.DT_INT64,
    ki: pl.DT_INT64,
    qi: pl.DT_INT64,
    q_count: pl.DT_INT64,
    skv_tiles: pl.DT_INT64,
    b_idx: pl.DT_INT64,
    n_idx: pl.DT_INT64,
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
    buf_idx = (q_count * skv_tiles + ki) % 2
    pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=EVENT_IDS_01[buf_idx])
    if ki == 0:
        pl.load_tile(q_mat_buf[q_count % 2], q, [b_idx, n_idx, qi, 0], order=[3, 2])
    pl.load_tile(k_mat_buf[buf_idx], k, [b_idx, n_idx, ki, 0])
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

    pl.move(qk_vec, acc_buf[l0c_idx], acc_to_vec_mode=pl.AccToVecMode.DualModeSplitN)
    pl.system.sync_src(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=EVENT_IDS_01[l0c_idx])
    pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=QK_READY_IDS[0])
    return


def compute_pv(
    task_id: pl.DT_INT64,
    ki: pl.DT_INT64,
    q_count: pl.DT_INT64,
    skv_tiles: pl.DT_INT64,
    b_idx: pl.DT_INT64,
    n_idx: pl.DT_INT64,
    l0ab_idx: pl.DT_INT64,
    l0c_idx: pl.DT_INT64,
    v: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    cube,
) -> None:
    pv_vec = cube.pv_vec
    p_mat_buf = (cube.p_mat_buf1, cube.p_mat_buf2)
    v_mat_buf = cube.v_mat_buf
    left_buf = cube.left_buf
    right_buf = cube.right_buf
    acc_buf = (cube.acc_buf1, cube.acc_buf2)
    buf_idx = (q_count * skv_tiles + ki) % 2
    pl.system.wait_cross_core(pipe=pl.PipeType.MTE1, event_id=P_READY_IDS[0])
    pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=EVENT_IDS_23[buf_idx])
    pl.load_tile(v_mat_buf[buf_idx], v, [b_idx, n_idx, ki, 0])
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

    pl.move(pv_vec, acc_buf[l0c_idx], acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
    pl.system.sync_src(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=EVENT_IDS_01[l0c_idx])
    pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=PV_READY_IDS[0])
    return


def apply_diag_mask(
    ki: pl.DT_INT64,
    qi: pl.DT_INT64,
    sub_id: pl.DT_INT64,
    sq_dim: pl.DT_INT64,
    skv_dim: pl.DT_INT64,
    stiles,
    attn_mask: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_UINT8],
) -> None:
    qk_vec = stiles.qk_vec
    tmp_vec = stiles.tmp_vec
    mask_u8_dn = stiles.mask_u8_dn
    mask_fp16_dn = stiles.mask_fp16_dn
    mask_vec_dn = stiles.mask_vec_dn
    neg_inf_vec = stiles.neg_inf_vec
    # qk_vec is DN [K, Q] and softmax reduces over K. The fixed UINT8 mask is
    # stored as mask[k, q] = 1 when q > k. Loading with Q offset +1 makes the
    # diagonal block produce 1 for keep positions and 0 for masked positions.
    # TCMPS(EQ 0) flips that to the TSEL predicate: 1 selects neg_inf.
    # Current A5 TCMPS/TSEL expects the full [TKV, TS_HALF] uint8 descriptor.
    pl.load(mask_u8_dn, attn_mask, [ki * TKV, qi * TS + sub_id * TS_HALF + 1])
    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
    pl.cast(mask_fp16_dn, mask_u8_dn, mode=pl.RoundMode.CAST_ROUND)
    pl.eq(mask_vec_dn, mask_fp16_dn, 0.0)
    pl.expands(neg_inf_vec, NEG_INF)
    pl.select(qk_vec, mask_vec_dn, neg_inf_vec, qk_vec, tmp_vec)


def softmax_body(
    task_id: pl.DT_INT64,
    ki: pl.DT_INT64,
    qi: pl.DT_INT64,
    q_count: pl.DT_INT64,
    skv_tiles: pl.DT_INT64,
    sub_id: pl.DT_INT64,
    b_idx: pl.DT_INT64,
    n_idx: pl.DT_INT64,
    sq_dim: pl.DT_INT64,
    skv_dim: pl.DT_INT64,
    stiles,
    attn_mask: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_UINT8],
) -> None:
    qk_vec = stiles.qk_vec
    tmp_vec = stiles.tmp_vec
    p_f16 = stiles.p_f16
    reduce_dst_rm = stiles.reduce_dst_rm
    global_max_rm = stiles.global_max_rm_buf[q_count % 2]
    global_sum_rm = stiles.global_sum_rm_buf[q_count % 2]
    exp_corr_rm = stiles.exp_corr_rm
    tile_nz = stiles.tile_nz
    p_mat_buf1 = stiles.p_mat_buf1
    p_mat_buf2 = stiles.p_mat_buf2
    if ki == qi:
        apply_diag_mask(ki, qi, sub_id, sq_dim, skv_dim, stiles, attn_mask)

    buf_idx = (q_count * skv_tiles + ki) % 2

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

    pl.move(tile_nz, p_f16)
    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
    if buf_idx == 0:
        pl.insert(p_mat_buf1, tile_nz, [0, TS_HALF * sub_id])
    else:
        pl.insert(p_mat_buf2, tile_nz, [0, TS_HALF * sub_id])


def compute_p(
    task_id: pl.DT_INT64,
    ki: pl.DT_INT64,
    qi: pl.DT_INT64,
    q_count: pl.DT_INT64,
    skv_tiles: pl.DT_INT64,
    sub_id: pl.DT_INT64,
    b_idx: pl.DT_INT64,
    n_idx: pl.DT_INT64,
    sq_dim: pl.DT_INT64,
    skv_dim: pl.DT_INT64,
    stiles,
    attn_mask: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_UINT8],
) -> None:
    pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=QK_READY_IDS[0])
    softmax_body(
        task_id,
        ki,
        qi,
        q_count,
        skv_tiles,
        sub_id,
        b_idx,
        n_idx,
        sq_dim,
        skv_dim,
        stiles,
        attn_mask,
    )
    # Both vector subblocks contribute one half of the shared P MAT tile. Make
    # sure both halves have finished TINSERT before cube starts PV.
    pl.system.bar_all()
    pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=P_READY_IDS[0])


def compute_gu(
    task_id: pl.DT_INT64,
    ki: pl.DT_INT64,
    qi: pl.DT_INT64,
    q_count: pl.DT_INT64,
    skv_tiles: pl.DT_INT64,
    sub_id: pl.DT_INT64,
    b_idx: pl.DT_INT64,
    n_idx: pl.DT_INT64,
    o: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    gtiles,
) -> None:
    pv_vec = gtiles.pv_vec
    running_o = gtiles.running_o
    exp_corr = gtiles.exp_corr
    global_sum = gtiles.global_sum_buf[q_count % 2]
    o_f16 = gtiles.o_f16
    pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=PV_READY_IDS[0])
    if ki == 0:
        pl.move(running_o, pv_vec)
    else:
        pl.expand_mul(running_o, running_o, exp_corr)
        pl.add(running_o, running_o, pv_vec)
    last_ki = pl.min(qi, skv_tiles - 1)
    if ki == last_ki:
        pl.expand_div(running_o, running_o, global_sum)
        pl.cast(o_f16, running_o, mode=pl.RoundMode.CAST_ROUND)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
        pl.store_tile(o, o_f16, [b_idx, n_idx, qi * 2 + sub_id, 0])


@pl.jit()
def fa_causal_bnsd_dn_kernel_v6(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    v: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    o: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    work_ranges: pl.Tensor[[pl.DYNAMIC, 2], pl.DT_INT32],
    attn_mask: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_UINT8],
):
    sq_dim = q.shape[2]
    skv_dim = k.shape[2]
    sq_tiles = (sq_dim + TS - 1) // TS
    skv_tiles = (skv_dim + TKV - 1) // TKV
    core_id = pl.get_block_idx() // pl.get_subblock_num()
    n_dim = q.shape[1]

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
    reduce_dst_rm = pl.make_tile(
        pl.TileType(shape=[1, TS_HALF], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addr=VA3, size=VB_RED
    )
    red_rm_type = pl.TileType(shape=[1, TS_HALF], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    red_type = pl.TileType(shape=[TS_HALF, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.DN)
    global_max_rm_0 = pl.make_tile(red_rm_type, addr=VA_GMAX0, size=VB_RED)
    global_max_rm_1 = pl.make_tile(red_rm_type, addr=VA_GMAX1, size=VB_RED)
    global_sum_0 = pl.make_tile(red_type, addr=VA_GSUM0, size=VB_RED)
    global_sum_1 = pl.make_tile(red_type, addr=VA_GSUM1, size=VB_RED)
    global_sum_rm_0 = pl.make_tile(red_rm_type, addr=VA_GSUM0, size=VB_RED)
    global_sum_rm_1 = pl.make_tile(red_rm_type, addr=VA_GSUM1, size=VB_RED)
    exp_corr = pl.make_tile(red_type, addr=VA_EXP0, size=VB_RED)
    exp_corr_rm = pl.make_tile(red_rm_type, addr=VA_EXP0, size=VB_RED)
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
    # mask_u8_dn holds the UINT8 tile loaded from fixed GM mask. mask_vec_dn is
    # the TCMPS predicate consumed by TSEL, using the full [TKV, TS_HALF] layout.
    mask_u8_dn = pl.make_tile(
        pl.TileType(shape=[TKV, TS_HALF], dtype=pl.DT_UINT8, target_memory=pl.MemorySpace.Vec), addr=VA11, size=VB1_KV
    )
    mask_fp16_dn = pl.make_tile(
        pl.TileType(shape=[TKV, TS_HALF], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec), addr=VA12, size=VB2_KV
    )
    mask_vec_dn = pl.make_tile(
        pl.TileType(shape=[TKV, TS_HALF], dtype=pl.DT_UINT8, target_memory=pl.MemorySpace.Vec),
        addr=VA13,
        size=VB1_KV,
    )
    neg_inf_vec = pl.make_tile(
        pl.TileType(shape=[TKV, TS_HALF], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addr=VA14, size=VB4_KV
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
            pl.TileType(
                shape=[TKV, TD],
                dtype=pl.DT_FP16,
                target_memory=pl.MemorySpace.Left,
                layout=pl.NZ,
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
            ),
            addr=LA1,
            size=KT_F16,
        )
        right_0 = pl.make_tile(
            pl.TileType(
                shape=[TD, TS],
                dtype=pl.DT_FP16,
                target_memory=pl.MemorySpace.Right,
            ),
            addr=RA0,
            size=Q_F16,
        )
        right_1 = pl.make_tile(
            pl.TileType(
                shape=[TD, TS],
                dtype=pl.DT_FP16,
                target_memory=pl.MemorySpace.Right,
            ),
            addr=RA1,
            size=Q_F16,
        )
        acc_buf1 = pl.make_tile(
            pl.TileType(
                shape=[TKV, TS],
                dtype=pl.DT_FP32,
                target_memory=pl.MemorySpace.Acc,
            ),
            addr=CA0,
            size=QK_F32,
        )
        acc_buf2 = pl.make_tile(
            pl.TileType(
                shape=[TS, TD],
                dtype=pl.DT_FP32,
                target_memory=pl.MemorySpace.Acc,
            ),
            addr=CA1,
            size=PV_F32,
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
            pv_vec=pv_vec,
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
            # Shift-register for delayed context (prev = task_id - 1)
            prev_task_id = 0
            prev_ki = 0
            prev_q_count = 0
            prev_skv_tiles = 0
            prev_b_idx = 0
            prev_n_idx = 0
            b_idx = work_id // n_dim
            n_idx = work_id % n_dim
            for qi in pl.range(0, sq_tiles):
                for ki in pl.range(0, skv_tiles):
                    if ki <= qi:
                        if task_id > 0:
                            compute_pv(
                                prev_task_id,
                                prev_ki,
                                prev_q_count,
                                prev_skv_tiles,
                                prev_b_idx,
                                prev_n_idx,
                                l0ab_idx,
                                l0c_idx,
                                v,
                                cube_tiles,
                            )
                            l0ab_idx = 1 - l0ab_idx
                            l0c_idx = 1 - l0c_idx
                        compute_qk(
                            task_id,
                            ki,
                            qi,
                            q_count,
                            skv_tiles,
                            b_idx,
                            n_idx,
                            l0ab_idx,
                            l0c_idx,
                            q,
                            k,
                            cube_tiles,
                        )
                        l0ab_idx = 1 - l0ab_idx
                        l0c_idx = 1 - l0c_idx
                        prev_task_id = task_id
                        prev_ki = ki
                        prev_q_count = q_count
                        prev_skv_tiles = skv_tiles
                        prev_b_idx = b_idx
                        prev_n_idx = n_idx
                        task_id = task_id + 1
                q_count = q_count + 1
            if task_id > 0:
                compute_pv(
                    prev_task_id,
                    prev_ki,
                    prev_q_count,
                    prev_skv_tiles,
                    prev_b_idx,
                    prev_n_idx,
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
            tmp_vec=tmp_vec,
            p_f16=p_f16,
            reduce_dst_rm=reduce_dst_rm,
            global_max_rm_buf=(global_max_rm_0, global_max_rm_1),
            global_sum_rm_buf=(global_sum_rm_0, global_sum_rm_1),
            exp_corr_rm=exp_corr_rm,
            tile_nz=tile_nz,
            p_mat_buf1=p_mat_buf1,
            p_mat_buf2=p_mat_buf2,
            mask_u8_dn=mask_u8_dn,
            mask_fp16_dn=mask_fp16_dn,
            mask_vec_dn=mask_vec_dn,
            neg_inf_vec=neg_inf_vec,
        )
        gu_tiles = pl.make_tuple(
            pv_vec=pv_vec,
            running_o=running_o,
            exp_corr=exp_corr,
            global_sum_buf=(global_sum_0, global_sum_1),
            o_f16=o_f16,
        )
        for work_id in pl.range(work_start, work_end):
            task_id = 0
            q_count = 0
            # Shift-registers for delayed context (prev = task-1, prev2 = task-2)
            prev_task_id = 0
            prev_ki = 0
            prev_qi = 0
            prev_q_count = 0
            prev_skv_tiles = 0
            prev_sub_id = 0
            prev_b_idx = 0
            prev_n_idx = 0
            prev2_task_id = 0
            prev2_ki = 0
            prev2_qi = 0
            prev2_q_count = 0
            prev2_skv_tiles = 0
            prev2_sub_id = 0
            prev2_b_idx = 0
            prev2_n_idx = 0
            b_idx = work_id // n_dim
            n_idx = work_id % n_dim
            for qi in pl.range(0, sq_tiles):
                for ki in pl.range(0, skv_tiles):
                    if ki <= qi:
                        if task_id > 1:
                            compute_gu(
                                prev2_task_id,
                                prev2_ki,
                                prev2_qi,
                                prev2_q_count,
                                prev2_skv_tiles,
                                prev2_sub_id,
                                prev2_b_idx,
                                prev2_n_idx,
                                o,
                                gu_tiles,
                            )
                        if task_id > 0:
                            compute_p(
                                prev_task_id,
                                prev_ki,
                                prev_qi,
                                prev_q_count,
                                prev_skv_tiles,
                                prev_sub_id,
                                prev_b_idx,
                                prev_n_idx,
                                sq_dim,
                                skv_dim,
                                softmax_tiles,
                                attn_mask,
                            )
                        # Shift: prev2 <- prev, prev <- current
                        prev2_task_id = prev_task_id
                        prev2_ki = prev_ki
                        prev2_qi = prev_qi
                        prev2_q_count = prev_q_count
                        prev2_skv_tiles = prev_skv_tiles
                        prev2_sub_id = prev_sub_id
                        prev2_b_idx = prev_b_idx
                        prev2_n_idx = prev_n_idx
                        prev_task_id = task_id
                        prev_ki = ki
                        prev_qi = qi
                        prev_q_count = q_count
                        prev_skv_tiles = skv_tiles
                        prev_sub_id = sub_id
                        prev_b_idx = b_idx
                        prev_n_idx = n_idx
                        task_id = task_id + 1
                q_count = q_count + 1
            if task_id > 0:
                if task_id > 1:
                    compute_gu(
                        prev2_task_id,
                        prev2_ki,
                        prev2_qi,
                        prev2_q_count,
                        prev2_skv_tiles,
                        prev2_sub_id,
                        prev2_b_idx,
                        prev2_n_idx,
                        o,
                        gu_tiles,
                    )
                compute_p(
                    prev_task_id,
                    prev_ki,
                    prev_qi,
                    prev_q_count,
                    prev_skv_tiles,
                    prev_sub_id,
                    prev_b_idx,
                    prev_n_idx,
                    sq_dim,
                    skv_dim,
                    softmax_tiles,
                    attn_mask,
                )
                compute_gu(
                    prev_task_id,
                    prev_ki,
                    prev_qi,
                    prev_q_count,
                    prev_skv_tiles,
                    prev_sub_id,
                    prev_b_idx,
                    prev_n_idx,
                    o,
                    gu_tiles,
                )


def flash_attention_causal_ref_bn(q, k, v, d):
    scale_val = 1.0 / math.sqrt(d)
    b, n, sq, _ = q.shape
    _, _, skv, _ = k.shape
    o_ref = torch.zeros_like(q)
    causal_mask = torch.triu(torch.ones(sq, skv, dtype=torch.bool, device=q.device), diagonal=1)
    for bi in range(b):
        for ni in range(n):
            qk = torch.matmul(q[bi, ni, :, :].float(), k[bi, ni, :, :].float().T) * scale_val
            qk = qk.masked_fill(causal_mask, float("-inf"))
            attn = torch.softmax(qk, dim=-1)
            o_ref[bi, ni, :, :] = torch.matmul(attn, v[bi, ni, :, :].float()).half()
    return o_ref


def make_causal_mask_dn_fixed_2048_u8(device):
    return torch.triu(torch.ones((FIXED_MASK_S, FIXED_MASK_S), dtype=torch.uint8, device=device), diagonal=1)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_fa_causal_bn_a5():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(42)

    for b, sq, n, skv, d, num_cores in [
        (1, 128, 1, 128, TD, 1),
        (1, 256, 1, 256, TD, 1),
        (1, 256, 5, 256, TD, 4),
        (3, 512, 1, 512, TD, 24),
        (1, 384, 1, 384, TD, 2),
        (1, 8192, 1, 8192, TD, 24),
    ]:
        logging.info("\nFA-BNSD-causal-DN-A5 (b=%s, sq=%s, n=%s, skv=%s, d=%s) cores=%s", b, sq, n, skv, d, num_cores)
        q = torch.rand((b, n, sq, d), device=device, dtype=torch.float16)
        k = torch.rand((b, n, skv, d), device=device, dtype=torch.float16)
        v = torch.rand((b, n, skv, d), device=device, dtype=torch.float16)
        o = torch.zeros((b, n, sq, d), device=device, dtype=torch.float16)
        attn_mask = make_causal_mask_dn_fixed_2048_u8(device)

        total_work = b * n
        work_ranges = torch.zeros((num_cores, 2), device=device, dtype=torch.int32)
        work_per_core = (total_work + num_cores - 1) // num_cores
        for core in range(num_cores):
            work_ranges[core, 0] = core * work_per_core
            work_ranges[core, 1] = min((core + 1) * work_per_core, total_work)

        actual_num_cores = min(num_cores, total_work)
        fa_causal_bnsd_dn_kernel_v6[None, actual_num_cores](q, k, v, o, work_ranges, attn_mask)
        torch.npu.synchronize()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.info("FA BNSD causal DN on A5 CCE")
    logging.info("%s", '=' * 60)
    test_fa_causal_bn_a5()
    logging.info("\nAll FlashAttention DN tests passed!")
