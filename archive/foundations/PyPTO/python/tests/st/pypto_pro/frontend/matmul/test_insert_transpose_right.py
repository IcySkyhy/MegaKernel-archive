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

"""On-chip transpose of the RIGHT matmul operand -- no pl.transpose, no GM round trip.

Uses the FA test_fa_perf_tkv_preload_dn.py idea (move ND->NZ, then insert into a Mat buffer
whose layout flips the fractal). The transpose is realized entirely by the Mat->L0 move:

  RIGHT operand, TMovToRight (dst L0B is pinned ZN):
    * Mat layout = NZ  -> Right ZN : NO transpose  (B = inserted data as-is)   [standard path]
    * Mat layout = ZN  -> Right ZN : TRANSPOSE      (B = inserted data ^T)      <-- used here
  (LEFT operand is symmetric: Mat NZ->Left NZ = identity, Mat ZN->Left NZ = transpose.)

Verified on A5: with rhs_mat=ZN, out == L @ d^T (max diff ~1.5e-5); with rhs_mat=NZ it is
L @ d instead. Square N so the FA-style insert (tile.dim0 == matbuf.dim0) fits.

  out[M, N] = L[M, N] @ B[N, N],   B = (vector data d)^T
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
SUB_N = 32
K = 128


def _require_a5(device):
    try:
        torch.npu.set_device(device)
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
    if "Ascend950" not in torch.npu.get_device_name():
        pytest.skip("not A5")


@pl.jit()
def insert_zn_right_kernel(
    lhs: pl.Tensor[[M, K], pl.DT_FP16],
    d: pl.Tensor[[N, K], pl.DT_FP16],  # right-operand data; matmul uses B = d^T
    out: pl.Tensor[[M, N], pl.DT_FP32],
):
    # RIGHT-operand L1 buffer declared ZN -> the Mat(ZN)->Right(ZN) move transposes it.
    rhs_mat = pl.make_tile(
        pl.TileType(shape=[K, N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
        addr=0x20000,
        size=32768,
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
        pl.load(tile_d, d, [off, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.move(tile_nz, tile_d)  # ND -> NZ (same shape)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=2)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=2)
        pl.insert(rhs_mat, tile_nz, [off, 0])  # NZ tile -> ZN Mat, column offset (FA-style)
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

        pl.load(lhs_mat, lhs, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.move(lhs_left, lhs_mat)
        pl.system.wait_cross_core(pipe=pl.PipeType.MTE1, event_id=2, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
        pl.move(rhs_right, rhs_mat)  # Mat(ZN) -> Right(ZN): TRANSPOSE
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        pl.matmul(c_l0c, lhs_left, rhs_right)  # out = L @ d^T
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.store(out, c_l0c, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_insert_zn_right_kernel():
    device = ST_DEVICE
    _require_a5(device)
    torch.manual_seed(0)
    lhs = torch.randn([M, K], device=device, dtype=torch.float16)
    d = torch.randn([N, K], device=device, dtype=torch.float16)
    out = torch.zeros([M, N], device=device, dtype=torch.float32)

    insert_zn_right_kernel(lhs, d, out)
    torch.npu.synchronize()

    ref = torch.matmul(lhs.float(), d.float().t())  # B = d^T, transposed on-chip
    logging.info("max|out - L@d^T| = %s", (out - ref).abs().max().item())
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)
    logging.info("insert_zn_right_kernel OK: right operand transposed on-chip (no pl.transpose, no GM)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    test_insert_zn_right_kernel()
