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

"""On-chip transpose via NZ-insert into a ZN Mat buffer (the FA test_fa_perf_tkv_preload_dn trick).

No pl.transpose, no GM round trip.

FA mechanism (verified here): the vector produces the operand CONTRACTION-major, moves it
ND->NZ (same shape), and inserts it into a Mat buffer declared with layout=pl.ZN. The
subsequent Mat(ZN)->Left(NZ) move hits TMovToLeft's SFractal-mismatch branch
(TExtractToATrans), which flips the fractal so the cube reads the logical transpose.

Modelled on FA's P@V: P is the LEFT operand [M, M]; softmax emits P CONTRACTION-major as
[M(key), M(query)] = P^T, and the ZN-insert + move deliver P[query, key] to L0A.

  out[M, D] = P[M, M] @ V[M, D],   where P = (vector data d)^T

Constraint this exposes: the insert needs tile.dim0 == matbuf.dim0, so the two P axes must
match -> the trick lands cleanly when P is square (as in FA, seq==seq). The vector must also
emit contraction-major data; it does not conjure a transpose from arbitrary orientation.
"""

import logging
import os

import pypto_pro.language as pl
import pytest
import torch

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"


M = 64  # square P: [M, M]  (contraction == output rows, like FA TKV==TS)
N = 64
K = 128  # V is [M, D]
SUB = K // 2  # 64, subcore split along the query axis


def _require_a5(device):
    try:
        torch.npu.set_device(device)
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
    if "Ascend950" not in torch.npu.get_device_name():
        pytest.skip("not A5")


@pl.jit()
def insert_zn_left_kernel(
    d: pl.Tensor[[M, K], pl.DT_FP16],  # P produced CONTRACTION-major: d[key, query] = P^T
    v: pl.Tensor[[K, N], pl.DT_FP16],  # V[key, d]
    out: pl.Tensor[[M, N], pl.DT_FP32],
):
    # LEFT-operand L1 buffer, declared ZN (FA's p_mat). This is what encodes the transpose.
    p_mat = pl.make_tile(
        pl.TileType(shape=[M, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addr=0x20000,
        size=32768,
    )

    with pl.section_vector():
        sub_id = pl.get_subblock_idx()
        off = sub_id * SUB

        # vector holds P contraction-major, [M(key), SUB(query)] per subcore (FA [TKV, TS_HALF])
        tile_d = pl.make_tile(
            pl.TileType(shape=[M, K // 2], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec), addr=0x0000, size=16384
        )
        tile_nz = pl.make_tile(
            pl.TileType(shape=[M, K // 2], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec, layout=pl.NZ),
            addr=0x6000,
            size=16384,
        )

        pl.load(tile_d, d, [0, off])  # d[:, off:off+SUB] -> [M, SUB]
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.move(tile_nz, tile_d)  # ND -> NZ (same shape)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=2)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=2)
        pl.insert(p_mat, tile_nz, [0, off])  # NZ tile -> ZN Mat, column offset (FA-style)
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

        pl.load(v_mat, v, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.move(v_right, v_mat)  # V -> L0B

        pl.system.wait_cross_core(pipe=pl.PipeType.MTE1, event_id=2, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
        pl.move(p_left, p_mat)  # Mat(ZN) -> Left(NZ): SFractal mismatch -> TRANSPOSE

        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        pl.matmul(c_l0c, p_left, v_right)  # out = P @ V = d^T @ V

        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.store(out, c_l0c, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_insert_zn_left_kernel():
    device = ST_DEVICE
    _require_a5(device)
    torch.manual_seed(0)
    d = torch.randn([M, K], device=device, dtype=torch.float16)  # = P^T (contraction-major)
    v = torch.randn([K, N], device=device, dtype=torch.float16)
    out = torch.zeros([M, N], device=device, dtype=torch.float32)

    insert_zn_left_kernel(d, v, out)
    torch.npu.synchronize()

    ref = torch.matmul(d.float(), v.float())  # P @ V, with P = d^T
    diff = (out - ref).abs().max().item()
    logging.info("max|out - (d^T @ v)| = %s", diff)
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)
    logging.info("insert_zn_left_kernel OK: on-chip transpose via ZN-insert (no pl.transpose, no GM)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    test_insert_zn_left_kernel()
