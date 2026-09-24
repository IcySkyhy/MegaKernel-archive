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

"""Tile-group variants of the UB-insert transpose cases.

Mirrors test_insert_{left,right,transpose_left,transpose_right}.py, but the L1 (Mat) operand is
a 2-slot `pl.make_tile_group` accessed via `.current()` (dynamic cursor) in both the vector
(insert) and cube (move) sections. This checks that NormalizeMatTransposeLayout relabels ALL
slots of an insert-fed group for the reversed (transpose) move, and leaves the no-transpose
(same-shape) group untouched -- without flipping any load (the group is filled by insert, not
by a GM load).
"""

import logging
import os

import pypto_pro.language as pl
import pytest
import torch

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"


M = 64
N = 64
K = 128
SUB = M // 2
SUB_N = N // 2


def _require_a5(device):
    try:
        torch.npu.set_device(device)
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
    if "Ascend950" not in torch.npu.get_device_name():
        pytest.skip("not A5")


# --- LEFT transpose, insert into a group (mirror test_insert_transpose_left.py) --------------
@pl.jit()
def insert_group_transpose_left(
    d: pl.Tensor[[K, M], pl.DT_FP16], v: pl.Tensor[[K, N], pl.DT_FP16], out: pl.Tensor[[M, N], pl.DT_FP32]
):
    p_mat_db = pl.make_tile_group(
        type=pl.TileType(shape=[M, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
        addrs=0x20000,
        mutex_ids=[10, 11],
    )

    with pl.section_vector():
        sub_id = pl.get_subblock_idx()
        off = sub_id * SUB
        tile_d = pl.make_tile(
            pl.TileType(shape=[K, M // 2], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec), addr=0x0000, size=16384
        )
        tile_nz = pl.make_tile(
            pl.TileType(shape=[K, M // 2], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec, layout=pl.NZ),
            addr=0x6000,
            size=16384,
        )
        p_mat = p_mat_db.current()
        pl.load(tile_d, d, [0, off])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.move(tile_nz, tile_d)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=2)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=2)
        pl.insert(p_mat, tile_nz, [0, off])
        pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=2)

    with pl.section_cube():
        v_mat = pl.make_tile(
            pl.TileType(shape=[K, N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
            addr=0x0000,
            size=16384,
        )
        p_left = pl.make_tile(
            pl.TileType(shape=[M, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
            addr=0x0000,
            size=32768,
        )
        v_right = pl.make_tile(
            pl.TileType(shape=[K, N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right, layout=pl.ZN),
            addr=0x0000,
            size=16384,
        )
        c_l0c = pl.make_tile(
            pl.TileType(shape=[M, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024),
            addr=0x0000,
            size=32768,
        )
        p_mat = p_mat_db.current()
        pl.load(v_mat, v, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.move(v_right, v_mat)
        pl.system.wait_cross_core(pipe=pl.PipeType.MTE1, event_id=2, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
        pl.move(p_left, p_mat)  # Mat[K,M] group -> Left[M,K]: reversed -> swap all slots
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        pl.matmul(c_l0c, p_left, v_right)
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.store(out, c_l0c, [0, 0])


# --- RIGHT transpose, insert into a group (mirror test_insert_transpose_right.py) ------------
@pl.jit()
def insert_group_transpose_right(
    lhs: pl.Tensor[[M, K], pl.DT_FP16], d: pl.Tensor[[N, K], pl.DT_FP16], out: pl.Tensor[[M, N], pl.DT_FP32]
):
    rhs_mat_db = pl.make_tile_group(
        type=pl.TileType(shape=[K, N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
        addrs=0x20000,
        mutex_ids=[10, 11],
    )

    with pl.section_vector():
        sub_id = pl.get_subblock_idx()
        off = sub_id * SUB_N
        tile_d = pl.make_tile(
            pl.TileType(shape=[SUB_N, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec), addr=0x0000, size=16384
        )
        tile_nz = pl.make_tile(
            pl.TileType(shape=[SUB_N, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec, layout=pl.NZ),
            addr=0x6000,
            size=16896,
        )
        rhs_mat = rhs_mat_db.current()
        pl.load(tile_d, d, [off, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.move(tile_nz, tile_d)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=2)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=2)
        pl.insert(rhs_mat, tile_nz, [off, 0])
        pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=2)

    with pl.section_cube():
        lhs_mat = pl.make_tile(
            pl.TileType(shape=[M, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
            addr=0x0000,
            size=32768,
        )
        lhs_left = pl.make_tile(
            pl.TileType(shape=[M, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
            addr=0x0000,
            size=32768,
        )
        rhs_right = pl.make_tile(
            pl.TileType(shape=[K, N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right, layout=pl.ZN),
            addr=0x0000,
            size=32768,
        )
        c_l0c = pl.make_tile(
            pl.TileType(shape=[M, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024),
            addr=0x0000,
            size=32768,
        )
        rhs_mat = rhs_mat_db.current()
        pl.load(lhs_mat, lhs, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.move(lhs_left, lhs_mat)
        pl.system.wait_cross_core(pipe=pl.PipeType.MTE1, event_id=2, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
        pl.move(rhs_right, rhs_mat)  # Mat[N,K] group -> Right[K,N]: reversed -> swap all slots
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        pl.matmul(c_l0c, lhs_left, rhs_right)
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.store(out, c_l0c, [0, 0])


# --- LEFT no-transpose, insert into a group (mirror test_insert_left.py) ---------------------
# group Mat NZ [M, K] same-shape -> Left NZ [M, K]: must NOT be swapped.
@pl.jit()
def insert_group_left(
    d: pl.Tensor[[M, K], pl.DT_FP16], v: pl.Tensor[[K, N], pl.DT_FP16], out: pl.Tensor[[M, N], pl.DT_FP32]
):
    p_mat_db = pl.make_tile_group(
        type=pl.TileType(shape=[M, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x20000,
        mutex_ids=[10, 11],
    )

    with pl.section_vector():
        sub_id = pl.get_subblock_idx()
        off = sub_id * SUB
        tile_d = pl.make_tile(
            pl.TileType(shape=[M, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec), addr=0x0000, size=16384
        )
        tile_nz = pl.make_tile(
            pl.TileType(shape=[M, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec, layout=pl.NZ),
            addr=0x6000,
            size=16896,
        )
        p_mat = p_mat_db.current()
        pl.load(tile_d, d, [0, off])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.move(tile_nz, tile_d)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=2)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=2)
        pl.insert(p_mat, tile_nz, [0, off])
        pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=2)

    with pl.section_cube():
        v_mat = pl.make_tile(
            pl.TileType(shape=[K, N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
            addr=0x0000,
            size=16384,
        )
        p_left = pl.make_tile(
            pl.TileType(shape=[M, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
            addr=0x0000,
            size=32768,
        )
        v_right = pl.make_tile(
            pl.TileType(shape=[K, N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right, layout=pl.ZN),
            addr=0x0000,
            size=16384,
        )
        c_l0c = pl.make_tile(
            pl.TileType(shape=[M, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024),
            addr=0x0000,
            size=32768,
        )
        p_mat = p_mat_db.current()
        pl.load(v_mat, v, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.move(v_right, v_mat)
        pl.system.wait_cross_core(pipe=pl.PipeType.MTE1, event_id=2, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
        pl.move(p_left, p_mat)  # Mat[M,K] group -> Left[M,K]: same-shape, not swapped
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        pl.matmul(c_l0c, p_left, v_right)
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.store(out, c_l0c, [0, 0])


# --- RIGHT no-transpose, insert into a group (mirror test_insert_right.py) -------------------
# group Mat NZ [K, N] same-shape -> Right ZN [K, N]: must NOT be swapped.
@pl.jit()
def insert_group_right(
    lhs: pl.Tensor[[M, K], pl.DT_FP16], d: pl.Tensor[[K, N], pl.DT_FP16], out: pl.Tensor[[M, N], pl.DT_FP32]
):
    rhs_mat_db = pl.make_tile_group(
        type=pl.TileType(shape=[K, N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x20000,
        mutex_ids=[10, 11],
    )

    with pl.section_vector():
        sub_id = pl.get_subblock_idx()
        off = sub_id * SUB_N
        tile_d = pl.make_tile(
            pl.TileType(shape=[K, SUB_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec), addr=0x0000, size=16384
        )
        tile_nz = pl.make_tile(
            pl.TileType(shape=[K, SUB_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec, layout=pl.NZ),
            addr=0x6000,
            size=16896,
        )
        rhs_mat = rhs_mat_db.current()
        pl.load(tile_d, d, [0, off])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.move(tile_nz, tile_d)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=2)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=2)
        pl.insert(rhs_mat, tile_nz, [0, off])
        pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=2)

    with pl.section_cube():
        lhs_mat = pl.make_tile(
            pl.TileType(shape=[M, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
            addr=0x0000,
            size=32768,
        )
        lhs_left = pl.make_tile(
            pl.TileType(shape=[M, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
            addr=0x0000,
            size=32768,
        )
        rhs_right = pl.make_tile(
            pl.TileType(shape=[K, N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right, layout=pl.ZN),
            addr=0x0000,
            size=32768,
        )
        c_l0c = pl.make_tile(
            pl.TileType(shape=[M, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024),
            addr=0x0000,
            size=32768,
        )
        rhs_mat = rhs_mat_db.current()
        pl.load(lhs_mat, lhs, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.move(lhs_left, lhs_mat)
        pl.system.wait_cross_core(pipe=pl.PipeType.MTE1, event_id=2, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
        pl.move(rhs_right, rhs_mat)  # Mat[K,N] group -> Right[K,N]: same-shape, not swapped
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        pl.matmul(c_l0c, lhs_left, rhs_right)
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.store(out, c_l0c, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_insert_group_left():
    _require_a5(ST_DEVICE)
    torch.manual_seed(0)
    d = torch.randn([M, K], device=ST_DEVICE, dtype=torch.float16)
    v = torch.randn([K, N], device=ST_DEVICE, dtype=torch.float16)
    out = torch.zeros([M, N], device=ST_DEVICE, dtype=torch.float32)
    insert_group_left(d, v, out)
    torch.npu.synchronize()
    ref = torch.matmul(d.float(), v.float())
    logging.info("insert_group_left: max|out - d@v| = %s", (out - ref).abs().max().item())
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_insert_group_right():
    _require_a5(ST_DEVICE)
    torch.manual_seed(0)
    lhs = torch.randn([M, K], device=ST_DEVICE, dtype=torch.float16)
    d = torch.randn([K, N], device=ST_DEVICE, dtype=torch.float16)
    out = torch.zeros([M, N], device=ST_DEVICE, dtype=torch.float32)
    insert_group_right(lhs, d, out)
    torch.npu.synchronize()
    ref = torch.matmul(lhs.float(), d.float())
    logging.info("insert_group_right: max|out - L@d| = %s", (out - ref).abs().max().item())
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_insert_group_transpose_left():
    _require_a5(ST_DEVICE)
    torch.manual_seed(0)
    d = torch.randn([K, M], device=ST_DEVICE, dtype=torch.float16)
    v = torch.randn([K, N], device=ST_DEVICE, dtype=torch.float16)
    out = torch.zeros([M, N], device=ST_DEVICE, dtype=torch.float32)
    insert_group_transpose_left(d, v, out)
    torch.npu.synchronize()
    ref = torch.matmul(d.float().t(), v.float())
    logging.info("insert_group_transpose_left: max|out - d^T@v| = %s", (out - ref).abs().max().item())
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_insert_group_transpose_right():
    _require_a5(ST_DEVICE)
    torch.manual_seed(0)
    lhs = torch.randn([M, K], device=ST_DEVICE, dtype=torch.float16)
    d = torch.randn([N, K], device=ST_DEVICE, dtype=torch.float16)
    out = torch.zeros([M, N], device=ST_DEVICE, dtype=torch.float32)
    insert_group_transpose_right(lhs, d, out)
    torch.npu.synchronize()
    ref = torch.matmul(lhs.float(), d.float().t())
    logging.info("insert_group_transpose_right: max|out - L@d^T| = %s", (out - ref).abs().max().item())
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    test_insert_group_left()
    test_insert_group_right()
    test_insert_group_transpose_left()
    test_insert_group_transpose_right()
