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

"""FlashAttention performance kernel using NBuffer + auto_mutex.

Refactored from test_fa_perf_tkv_preload.py to use:
  - NBuffer with current() auto-rotate cursor (no manual buf_idx)
  - auto_mutex=True for automatic Mutex synchronization
  - Cross-core event_id synchronization preserved (QK_READY, P_READY, PV_READY)

Features:
  1. Multi-core: each Cube core processes multiple Q tiles via strided loop
  2. Double buffer: NBuffer with current() auto-rotate for Q/K/V/P/L0A/L0B/L0C
  3. FIFO cross-core GM buffers with configurable depth
  4. Cross-core event ID ping/pong
  5. QK pre-compute: Cube runs QK_PRELOAD tiles ahead
  6. Vector: FIFO exp_corr, double-buffered global_max/global_sum

Usage:
    python3 python/tests/st/pypto_pro/frontend/fa/test_fa_perf_tkv_preload_nbuf.py
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
#  Configuration
# ================================================================
QK_PRELOAD = 1
FIFO_SIZE = QK_PRELOAD + 1

# ================================================================
#  Tile dimensions and constants
# ================================================================
TS = 128
TKV = 128
TD = 128
TS_HALF = TS // 2
SCALE = 1.0 / math.sqrt(TD)

# Buffer sizes (bytes)
Q_F16 = TS * TD * 2
KT_F16 = TD * TKV * 2
V_F16 = TKV * TD * 2
P_F16 = TS * TKV * 2
QK_HALF_F32 = TS * TKV * 4
PV_HALF_F32 = TS * TD * 4

# VEC buffer sizes
VB4_KV = TS_HALF * TKV * 4
VB2_KV = TS_HALF * TKV * 2
VB4 = TS_HALF * TD * 4
VB2 = TS_HALF * TD * 2
VB6 = (TS_HALF + 1) * TD * 2
VB_RED = TS_HALF * 1 * 4

# ================================================================
#  Buffer addresses
# ================================================================
# MAT (512KB) - L1 buffers
MA0_Q = 0
MA1_K = Q_F16 * 2
MA2_P = MA1_K + KT_F16 * 2
MA3_V = MA2_P + P_F16 * 2

# L0A/L0B/L0C addresses
LA0 = 0
LA1 = P_F16
RA0 = 0
RA1 = KT_F16
CA0 = 0
CA1 = QK_HALF_F32

# VEC (192KB) addresses
VA0 = 0
VA1 = VA0 + VB4_KV * 2
VA2 = VA1 + VB4_KV
VA3 = VA2 + VB2_KV
VA_GMAX0 = VA3 + VB_RED
VA_GMAX1 = VA_GMAX0 + VB_RED
VA_GSUM0 = VA_GMAX1 + VB_RED
VA_GSUM1 = VA_GSUM0 + VB_RED
VA_EXP_BASE = VA_GSUM1 + VB_RED
VA_AFTER_EXP = VA_EXP_BASE + FIFO_SIZE * VB_RED
VA7 = VA_AFTER_EXP
VA8 = VA7 + VB4
VA9 = VA8 + VB4 * 2
VA10 = VA9 + VB2
VA11 = VA10 + VB6
assert VA11 <= 248 * 1024

# ================================================================
#  Event IDs (cross-core)
# ================================================================
QK_READY_IDS = tuple(range(0, FIFO_SIZE))
P_READY_IDS = tuple(range(FIFO_SIZE, 2 * FIFO_SIZE))
PV_READY_IDS = tuple(range(2 * FIFO_SIZE, 3 * FIFO_SIZE))
assert 3 * FIFO_SIZE <= 16

# ================================================================
#  Mutex IDs - Cube and Vector use independent buf_id spaces
# ================================================================
# Cube-only (inside section_cube): 0-11
#   Q L1: (0, 1), K L1: (2, 3), V L1: (4, 5)
#   L0A: (6, 7), L0B: (8, 9), L0C: (10, 11)
#
# Vector-only (inside section_vector): 0-11
#   tmp_vec: 0, p_f16: 1, reduce_dst: 2
#   gmax_rm: (3, 4), gsum: (5, 6)
#   exp_corr: (7, 8), running_o: 9, o_f16: 10, tile_nz: 11
#
# Cross-core shared (outside sections): 12-17
#   P MAT: (12, 13)
#   qk_vec UB: (14, 15)
#   pv_vec UB: (16, 17)

# Cross-core shared buffer IDs
P_MUTEX_IDS = (12, 13)
QK_VEC_BUF_IDS = (14, 15)
PV_VEC_BUF_IDS = (16, 17)

PV_CORE_STRIDE = 2 * FIFO_SIZE * TS


# ================================================================
#  Kernel with NBuffer + auto_mutex
# ================================================================
@pl.jit(auto_mutex=True)
def fa_perf_tkv_preload_nbuf_kernel(
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

    # ========== Cross-core shared buffers (tile groups for double-buffer) ==========
    # P MAT - Vector insert, Cube PV read
    p_mat_db = pl.make_tile_group(
        type=pl.TileType(shape=[TS, TKV], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=MA2_P,
        mutex_ids=P_MUTEX_IDS,
    )

    # qk_vec UB - Cube store from ACC, Vector softmax (double-buffer for FIFO)
    qk_vec_db = pl.make_tile_group(
        type=pl.TileType(shape=[TS_HALF, TKV], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=VA0,
        mutex_ids=QK_VEC_BUF_IDS,
    )

    # pv_vec UB - Cube store from ACC, Vector GU (double-buffer for FIFO)
    pv_vec_db = pl.make_tile_group(
        type=pl.TileType(shape=[TS_HALF, TD], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=VA8,
        mutex_ids=PV_VEC_BUF_IDS,
    )

    # =================== CUBE SECTION ===================
    with pl.section_cube():
        # Cube-only buffers (independent buf_id space: 0-11)
        q_l1_db = pl.make_tile_group(
            type=pl.TileType(shape=[TS, TD], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
            addrs=MA0_Q,
            mutex_ids=[0, 1],
        )
        k_l1_db = pl.make_tile_group(
            type=pl.TileType(shape=[TD, TKV], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
            addrs=MA1_K,
            mutex_ids=[2, 3],
        )
        v_l1_db = pl.make_tile_group(
            type=pl.TileType(shape=[TKV, TD], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
            addrs=MA3_V,
            mutex_ids=[4, 5],
        )

        left_db = pl.make_tile_group(
            type=pl.TileType(shape=[TS, TD], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
            addrs=LA0,
            mutex_ids=[6, 7],
        )
        right_db = pl.make_tile_group(
            type=pl.TileType(shape=[TKV, TD], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
            addrs=RA0,
            mutex_ids=[8, 9],
        )
        acc_db = pl.make_tile_group(
            type=pl.TileType(shape=[TS, TKV], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
            addrs=CA0,
            mutex_ids=[10, 11],
        )

        task_id = 0
        ctx_arr = pl.struct_array(3, "CubeCtx", task_id=0, ki=0)

        for qi in pl.range(core_id, sq_tiles, num_cores):
            sq_off = qi * TS

            cur_q = q_l1_db.next()
            for ki in pl.range(0, skv_tiles):
                # Save current context
                ctx_curr = ctx_arr[task_id % 3]
                ctx_curr.task_id = task_id
                ctx_curr.ki = ki

                # ========== compute_qk (current step) ==========
                skv_off = ki * TKV
                cur_k = k_l1_db.next()
                qk_left = left_db.next()
                qk_right = right_db.next()
                qk_acc = acc_db.next()

                pl.load(cur_k, k, [skv_off, 0], order=[1, 0])
                if ki == 0:
                    pl.load(cur_q, q, [sq_off, 0])

                pl.move(qk_left, cur_q)
                pl.move(qk_right, cur_k)
                pl.matmul(qk_acc, qk_left, qk_right)

                qk_t = qk_vec_db.next()
                pl.move(qk_t, qk_acc, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)

                qk_eid = task_id % FIFO_SIZE
                pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=QK_READY_IDS[qk_eid])

                # ========== compute_pv (delayed 1 step: uses ctx from task_id-1) ==========
                if task_id > 0:
                    ctx_pre = ctx_arr[(task_id + 2) % 3]
                    pv_eid = ctx_pre.task_id % FIFO_SIZE
                    sv_off = ctx_pre.ki * TKV

                    pl.system.wait_cross_core(pipe=pl.PipeType.MTE1, event_id=P_READY_IDS[pv_eid])

                    cur_v = v_l1_db.next()
                    # next() advances p_mat_db cursor to stay in sync with Vec.
                    cur_p = p_mat_db.next()
                    pv_left = left_db.next()
                    pv_right = right_db.next()
                    pv_acc = acc_db.next()

                    pl.load(cur_v, v, [sv_off, 0])
                    pl.move(pv_left, cur_p)
                    pl.move(pv_right, cur_v)
                    pl.matmul(pv_acc, pv_left, pv_right)

                    pv_t = pv_vec_db.next()
                    pl.move(pv_t, pv_acc, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
                    pl.system.set_cross_core(
                        pipe=pl.PipeType.FIX,
                        event_id=PV_READY_IDS[pv_eid],
                    )

                task_id = task_id + 1

        # Cube epilogue: drain last PV
        ctx_pre = ctx_arr[(task_id + 2) % 3]
        pv_eid = ctx_pre.task_id % FIFO_SIZE
        sv_off = ctx_pre.ki * TKV

        pl.system.wait_cross_core(pipe=pl.PipeType.MTE1, event_id=P_READY_IDS[pv_eid])

        cur_v = v_l1_db.next()
        cur_p = p_mat_db.next()
        pv_left = left_db.next()
        pv_right = right_db.next()
        pv_acc = acc_db.next()

        pl.load(cur_v, v, [sv_off, 0])
        pl.move(pv_left, cur_p)
        pl.move(pv_right, cur_v)
        pl.matmul(pv_acc, pv_left, pv_right)

        pv_t = pv_vec_db.next()
        pl.move(pv_t, pv_acc, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=PV_READY_IDS[pv_eid])

    # =================== VECTOR SECTION ===================
    with pl.section_vector():
        # Vector-only buffers (independent buf_id space: 0-11). Scratch tiles with
        # no mutex are plain make_tile; mutex'd buffers are single-tile groups.
        tmp_vec = pl.make_tile(
            pl.TileType(shape=[TS_HALF, TKV], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addr=VA1, size=VB4_KV
        )
        p_f16_g = pl.make_tile_group(
            type=pl.TileType(shape=[TS_HALF, TKV], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
            addrs=VA2,
            mutex_ids=[1],
        )
        p_f16 = p_f16_g.next()
        reduce_dst = pl.make_tile(
            pl.TileType(shape=[TS_HALF, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.DN),
            addr=VA3,
            size=VB_RED,
        )
        reduce_dst_rm = pl.make_tile(
            pl.TileType(shape=[1, TS_HALF], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addr=VA3, size=VB_RED
        )

        running_o = pl.make_tile(
            pl.TileType(shape=[TS_HALF, TD], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addr=VA7, size=VB4
        )
        o_f16_g = pl.make_tile_group(
            type=pl.TileType(shape=[TS_HALF, TD], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
            addrs=VA9,
            mutex_ids=[10],
        )
        o_f16 = o_f16_g.next()
        tile_nz_g = pl.make_tile_group(
            type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec, layout=pl.NZ),
            addrs=VA10,
            mutex_ids=[11],
        )
        tile_nz = tile_nz_g.next()

        # Double-buffered global state (per Q tile) -use tile tuples for dynamic
        # indexing by q_count % 2, since StructArray ctx references need runtime index.
        gmax_rm_type = pl.TileType(shape=[1, TS_HALF], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        gmax_rm_0 = pl.make_tile(gmax_rm_type, addr=VA_GMAX0, size=VB_RED)
        gmax_rm_1 = pl.make_tile(gmax_rm_type, addr=VA_GMAX1, size=VB_RED)
        global_max_rm_buf = (gmax_rm_0, gmax_rm_1)

        gsum_type = pl.TileType(shape=[TS_HALF, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.DN)
        gsum_rm_type = pl.TileType(shape=[1, TS_HALF], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        gsum_0 = pl.make_tile(gsum_type, addr=VA_GSUM0, size=VB_RED)
        gsum_1 = pl.make_tile(gsum_type, addr=VA_GSUM1, size=VB_RED)
        gsum_rm_0 = pl.make_tile(gsum_rm_type, addr=VA_GSUM0, size=VB_RED)
        gsum_rm_1 = pl.make_tile(gsum_rm_type, addr=VA_GSUM1, size=VB_RED)
        global_sum_buf = (gsum_0, gsum_1)
        global_sum_rm_buf = (gsum_rm_0, gsum_rm_1)

        # FIFO exp_corr -tile group with next() rotation (mutex ids 7,8)
        exp_corr_db = pl.make_tile_group(
            type=pl.TileType(shape=[TS_HALF, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.DN),
            addrs=VA_EXP_BASE,
            mutex_ids=[7, 8],
        )
        exp_corr_rm_db = pl.make_tile_group(
            type=pl.TileType(shape=[1, TS_HALF], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addrs=VA_EXP_BASE,
            mutex_ids=[7, 8],
        )

        sub_id = pl.get_subblock_idx()
        row_off = sub_id * TS_HALF
        task_id = 0
        q_count = 0

        # StructArray(3) for pipeline context tracking (same as original)
        ctx_arr = pl.struct_array(3, "VecCtx", sq_off=0, task_id=0, qi=0, ki=0, skv_tiles=0, q_count=0)

        for qi in pl.range(core_id, sq_tiles, num_cores):
            sq_off = qi * TS
            q_idx = q_count % 2  # noqa: F841

            for ki in pl.range(0, skv_tiles):
                # Save current context
                ctx_curr = ctx_arr[task_id % 3]
                ctx_curr.sq_off = sq_off
                ctx_curr.task_id = task_id
                ctx_curr.qi = qi
                ctx_curr.ki = ki
                ctx_curr.skv_tiles = skv_tiles
                ctx_curr.q_count = q_count

                # --- compute_p (delayed 1 step: uses ctx from task_id-1) ---
                if task_id > 0:
                    ctx_p = ctx_arr[(task_id + 2) % 3]
                    p_eid = ctx_p.task_id % FIFO_SIZE
                    pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=QK_READY_IDS[p_eid])

                    qk_t = qk_vec_db.next()
                    q_idx_p = ctx_p.q_count % 2
                    gmax_p = global_max_rm_buf[q_idx_p]
                    gsum_rm_p = global_sum_rm_buf[q_idx_p]

                    if ctx_p.ki == 0:
                        pl.maximum(reduce_dst, qk_t, tmp_vec, dim=0)
                        pl.expand_sub(tmp_vec, qk_t, reduce_dst)
                        pl.mul(gmax_p, reduce_dst_rm, 1.0)
                        pl.mul(tmp_vec, tmp_vec, SCALE)
                        pl.exp(qk_t, tmp_vec)
                        pl.sum(reduce_dst, qk_t, tmp_vec)
                        pl.mul(gsum_rm_p, reduce_dst_rm, 1.0)
                        pl.cast(p_f16, qk_t, mode=pl.RoundMode.CAST_ROUND)
                    if ctx_p.ki > 0:
                        exp_corr_rm = exp_corr_rm_db.next()
                        pl.maximum(reduce_dst, qk_t, tmp_vec, dim=0)
                        pl.maximum(reduce_dst_rm, reduce_dst_rm, gmax_p)
                        pl.sub(exp_corr_rm, gmax_p, reduce_dst_rm)
                        pl.mul(gmax_p, reduce_dst_rm, 1.0)
                        pl.expand_sub(tmp_vec, qk_t, reduce_dst)
                        pl.mul(exp_corr_rm, exp_corr_rm, SCALE)
                        pl.mul(tmp_vec, tmp_vec, SCALE)
                        pl.exp(exp_corr_rm, exp_corr_rm)
                        pl.exp(qk_t, tmp_vec)
                        pl.cast(p_f16, qk_t, mode=pl.RoundMode.CAST_ROUND)
                        pl.mul(gsum_rm_p, gsum_rm_p, exp_corr_rm)
                        pl.sum(reduce_dst, qk_t, tmp_vec)
                        pl.add(gsum_rm_p, gsum_rm_p, reduce_dst_rm)

                    pl.move(tile_nz, p_f16)
                    cur_p = p_mat_db.next()
                    pl.insert(cur_p, tile_nz, [64 * sub_id, 0])
                    pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=P_READY_IDS[p_eid])

                # --- compute_gu (delayed 2 steps: uses ctx from task_id-2) ---
                if task_id > 1:
                    ctx_gu = ctx_arr[(task_id + 1) % 3]
                    gu_eid = ctx_gu.task_id % FIFO_SIZE
                    pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=PV_READY_IDS[gu_eid])

                    pv_t = pv_vec_db.next()
                    if ctx_gu.ki == 0:
                        pl.move(running_o, pv_t)
                    if ctx_gu.ki > 0:
                        exp_corr_gu = exp_corr_db.next()
                        pl.expand_mul(running_o, running_o, exp_corr_gu)
                        pl.add(running_o, running_o, pv_t)
                    if ctx_gu.ki == ctx_gu.skv_tiles - 1:
                        gsum_gu = global_sum_buf[ctx_gu.q_count % 2]
                        pl.expand_div(running_o, running_o, gsum_gu)
                        pl.cast(o_f16, running_o, mode=pl.RoundMode.CAST_ROUND)
                        pl.store(o, o_f16, [ctx_gu.sq_off + row_off, 0])

                task_id = task_id + 1
            q_count = q_count + 1

        # --- Vector epilogue: drain pipeline ---
        # compute_p for last task
        ctx_p = ctx_arr[(task_id + 2) % 3]
        p_eid = ctx_p.task_id % FIFO_SIZE
        pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=QK_READY_IDS[p_eid])

        qk_t = qk_vec_db.next()
        q_idx_p = ctx_p.q_count % 2
        gmax_p = global_max_rm_buf[q_idx_p]
        gsum_rm_p = global_sum_rm_buf[q_idx_p]

        if ctx_p.ki == 0:
            pl.maximum(reduce_dst, qk_t, tmp_vec, dim=0)
            pl.expand_sub(tmp_vec, qk_t, reduce_dst)
            pl.mul(gmax_p, reduce_dst_rm, 1.0)
            pl.mul(tmp_vec, tmp_vec, SCALE)
            pl.exp(qk_t, tmp_vec)
            pl.sum(reduce_dst, qk_t, tmp_vec)
            pl.mul(gsum_rm_p, reduce_dst_rm, 1.0)
            pl.cast(p_f16, qk_t, mode=pl.RoundMode.CAST_ROUND)
        if ctx_p.ki > 0:
            exp_corr_rm = exp_corr_rm_db.next()
            pl.maximum(reduce_dst, qk_t, tmp_vec, dim=0)
            pl.maximum(reduce_dst_rm, reduce_dst_rm, gmax_p)
            pl.sub(exp_corr_rm, gmax_p, reduce_dst_rm)
            pl.mul(gmax_p, reduce_dst_rm, 1.0)
            pl.expand_sub(tmp_vec, qk_t, reduce_dst)
            pl.mul(exp_corr_rm, exp_corr_rm, SCALE)
            pl.mul(tmp_vec, tmp_vec, SCALE)
            pl.exp(exp_corr_rm, exp_corr_rm)
            pl.exp(qk_t, tmp_vec)
            pl.cast(p_f16, qk_t, mode=pl.RoundMode.CAST_ROUND)
            pl.mul(gsum_rm_p, gsum_rm_p, exp_corr_rm)
            pl.sum(reduce_dst, qk_t, tmp_vec)
            pl.add(gsum_rm_p, gsum_rm_p, reduce_dst_rm)

        pl.move(tile_nz, p_f16)
        cur_p = p_mat_db.next()
        pl.insert(cur_p, tile_nz, [64 * sub_id, 0])
        pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=P_READY_IDS[p_eid])

        # compute_gu for task_id-2
        if task_id > 1:
            ctx_gu = ctx_arr[(task_id + 1) % 3]
            gu_eid = ctx_gu.task_id % FIFO_SIZE
            pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=PV_READY_IDS[gu_eid])
            pv_t = pv_vec_db.next()
            if ctx_gu.ki == 0:
                pl.move(running_o, pv_t)
            if ctx_gu.ki > 0:
                exp_corr_gu = exp_corr_db.next()
                pl.expand_mul(running_o, running_o, exp_corr_gu)
                pl.add(running_o, running_o, pv_t)
            if ctx_gu.ki == ctx_gu.skv_tiles - 1:
                gsum_gu = global_sum_buf[ctx_gu.q_count % 2]
                pl.expand_div(running_o, running_o, gsum_gu)
                pl.cast(o_f16, running_o, mode=pl.RoundMode.CAST_ROUND)
                pl.store(o, o_f16, [ctx_gu.sq_off + row_off, 0])

        task_id = task_id + 1

        # Final compute_gu for task_id-2 (last task)
        ctx_gu = ctx_arr[(task_id + 1) % 3]
        gu_eid = ctx_gu.task_id % FIFO_SIZE
        pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=PV_READY_IDS[gu_eid])
        pv_t = pv_vec_db.next()
        if ctx_gu.ki == 0:
            pl.move(running_o, pv_t)
        if ctx_gu.ki > 0:
            exp_corr_gu = exp_corr_db.next()
            pl.expand_mul(running_o, running_o, exp_corr_gu)
            pl.add(running_o, running_o, pv_t)
        if ctx_gu.ki == ctx_gu.skv_tiles - 1:
            gsum_gu = global_sum_buf[ctx_gu.q_count % 2]
            pl.expand_div(running_o, running_o, gsum_gu)
            pl.cast(o_f16, running_o, mode=pl.RoundMode.CAST_ROUND)
            pl.store(o, o_f16, [ctx_gu.sq_off + row_off, 0])


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
def test_fa_perf_nbuf():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(42)

    for sq, skv, d, num_cores in [
        (8192, 8192, TD, 28),
    ]:
        logging.info("\nFA-Perf-NBuffer (%s,%s,%s) cores=%s QK_PRELOAD=%s", sq, skv, d, num_cores, QK_PRELOAD)
        q_t = torch.rand((sq, d), device=device, dtype=torch.float16)
        k_t = torch.rand((skv, d), device=device, dtype=torch.float16)
        v_t = torch.rand((skv, d), device=device, dtype=torch.float16)
        o_t = torch.zeros((sq, d), device=device, dtype=torch.float16)
        qk_t = torch.zeros((sq * FIFO_SIZE, skv), device=device, dtype=torch.float32)
        p_t = torch.zeros((sq * FIFO_SIZE, skv), device=device, dtype=torch.float16)
        pv_t = torch.zeros((48 * PV_CORE_STRIDE, d), device=device, dtype=torch.float32)

        fa_perf_tkv_preload_nbuf_kernel[None, num_cores](q_t, k_t, v_t, o_t, qk_t, p_t, pv_t)
        torch.npu.synchronize()

        qk_ref, x_exp_ref, o_ref = flash_attention_ref(q_t, k_t, v_t, d)
        diff = (o_t - o_ref).abs().max().item()
        logging.info("  max|diff|=%.4f", diff)
        torch.testing.assert_close(o_t, o_ref, rtol=5e-3, atol=5e-3)
        logging.info("  PASS")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.info("FA perf with NBuffer + auto_mutex (QK_PRELOAD=%s, FIFO=%s)", QK_PRELOAD, FIFO_SIZE)
    logging.info("%s", '=' * 60)
    test_fa_perf_nbuf()  # uncomment on A5 NPU
    logging.info("\nParse test passed!")
