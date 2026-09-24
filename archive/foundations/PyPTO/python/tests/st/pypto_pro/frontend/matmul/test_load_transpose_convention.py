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

"""GM-load layout convention for Mat matmul operands (verified on A5).

Convention under test (the "is_transpose <-> layout/shape" rule):

    need_transpose = False  ->  Mat layout = pl.NZ, Mat shape = GM shape        (load as-is)
    need_transpose = True   ->  Mat layout = pl.ZN, Mat shape = reverse(GM)      (load transposed)

The Mat->L0 move (TMOV) requires src and dst to have the SAME physical [Rows, Cols], so a
transpose-load is *useful* exactly when it lands the operand in the shape the L0 buffer wants,
turning the move into a same-shape (identity) reformat. That is how these kernels are written:

  C[M, N] = A[M, K] @ B[K, N]

  left  A -> L0A (Left,  NZ [M, K]):
      no-transpose : GM a  =[M, K]  -> a_mat NZ [M, K]  (load as-is)         then a_mat -> Left
      transpose    : GM a_t=[K, M]  -> a_mat ZN [M, K]  (is_transpose=True)  then a_mat -> Left
  right B -> L0B (Right, ZN [K, N]):
      no-transpose : GM b  =[K, N]  -> b_mat NZ [K, N]  (load as-is)         then b_mat -> Right
      transpose    : GM b_t=[N, K]  -> b_mat ZN [K, N]  (is_transpose=True)  then b_mat -> Right

Both transpose paths declare the Mat in the PHYSICAL form the frontend expects (shape already
matching L0), so NormalizeMatTransposeLayout must NOT relabel them -- it skips any Mat that is a
GM (block.load) destination. Non-square M/K/N below would expose an accidental transpose.
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
K = 128
N = 32

MA0 = 0x0000
MA1 = 0x10000
SZ = 32768


def _require_a5(device):
    try:
        torch.npu.set_device(device)
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
    if "Ascend950" not in torch.npu.get_device_name():
        pytest.skip("not A5")


# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------
@pl.jit()
def k_nn(a: pl.Tensor[[M, K], pl.DT_FP16], b: pl.Tensor[[K, N], pl.DT_FP16], out: pl.Tensor[[M, N], pl.DT_FP32]):
    with pl.section_cube():
        a_mat = pl.make_tile(
            pl.TileType(shape=[M, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
            addr=MA0,
            size=SZ,
        )
        b_mat = pl.make_tile(
            pl.TileType(shape=[K, N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
            addr=MA1,
            size=SZ,
        )
        a_left = pl.make_tile(
            pl.TileType(shape=[M, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
            addr=0x0,
            size=SZ,
        )
        b_right = pl.make_tile(
            pl.TileType(shape=[K, N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right, layout=pl.ZN),
            addr=0x0,
            size=SZ,
        )
        c = pl.make_tile(
            pl.TileType(shape=[M, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024),
            addr=0x0,
            size=SZ,
        )
        pl.load(a_mat, a, [0, 0])
        pl.load(b_mat, b, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.move(a_left, a_mat)
        pl.move(b_right, b_mat)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        pl.matmul(c, a_left, b_right)
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.store(out, c, [0, 0])


@pl.jit()
def k_nt(a: pl.Tensor[[M, K], pl.DT_FP16], b_t: pl.Tensor[[N, K], pl.DT_FP16], out: pl.Tensor[[M, N], pl.DT_FP32]):
    with pl.section_cube():
        a_mat = pl.make_tile(
            pl.TileType(shape=[M, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
            addr=MA0,
            size=SZ,
        )
        b_mat = pl.make_tile(
            pl.TileType(shape=[K, N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
            addr=MA1,
            size=SZ,
        )  # transpose-load -> ZN [K,N]
        a_left = pl.make_tile(
            pl.TileType(shape=[M, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
            addr=0x0,
            size=SZ,
        )
        b_right = pl.make_tile(
            pl.TileType(shape=[K, N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right, layout=pl.ZN),
            addr=0x0,
            size=SZ,
        )
        c = pl.make_tile(
            pl.TileType(shape=[M, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024),
            addr=0x0,
            size=SZ,
        )
        pl.load(a_mat, a, [0, 0])
        pl.load(b_mat, b_t, [0, 0], order=[1, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.move(a_left, a_mat)
        pl.move(b_right, b_mat)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        pl.matmul(c, a_left, b_right)
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.store(out, c, [0, 0])


@pl.jit()
def k_tn(a_t: pl.Tensor[[K, M], pl.DT_FP16], b: pl.Tensor[[K, N], pl.DT_FP16], out: pl.Tensor[[M, N], pl.DT_FP32]):
    with pl.section_cube():
        a_mat = pl.make_tile(
            pl.TileType(shape=[M, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
            addr=MA0,
            size=SZ,
        )  # transpose-load -> ZN [M,K]
        b_mat = pl.make_tile(
            pl.TileType(shape=[K, N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
            addr=MA1,
            size=SZ,
        )
        a_left = pl.make_tile(
            pl.TileType(shape=[M, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
            addr=0x0,
            size=SZ,
        )
        b_right = pl.make_tile(
            pl.TileType(shape=[K, N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right, layout=pl.ZN),
            addr=0x0,
            size=SZ,
        )
        c = pl.make_tile(
            pl.TileType(shape=[M, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024),
            addr=0x0,
            size=SZ,
        )
        pl.load(a_mat, a_t, [0, 0], order=[1, 0])
        pl.load(b_mat, b, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.move(a_left, a_mat)
        pl.move(b_right, b_mat)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        pl.matmul(c, a_left, b_right)
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.store(out, c, [0, 0])


@pl.jit()
def k_tt(a_t: pl.Tensor[[K, M], pl.DT_FP16], b_t: pl.Tensor[[N, K], pl.DT_FP16], out: pl.Tensor[[M, N], pl.DT_FP32]):
    with pl.section_cube():
        a_mat = pl.make_tile(
            pl.TileType(shape=[M, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
            addr=MA0,
            size=SZ,
        )  # transpose-load -> ZN [M,K]
        b_mat = pl.make_tile(
            pl.TileType(shape=[K, N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
            addr=MA1,
            size=SZ,
        )  # transpose-load -> ZN [K,N]
        a_left = pl.make_tile(
            pl.TileType(shape=[M, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
            addr=0x0,
            size=SZ,
        )
        b_right = pl.make_tile(
            pl.TileType(shape=[K, N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right, layout=pl.ZN),
            addr=0x0,
            size=SZ,
        )
        c = pl.make_tile(
            pl.TileType(shape=[M, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024),
            addr=0x0,
            size=SZ,
        )
        pl.load(a_mat, a_t, [0, 0], order=[1, 0])
        pl.load(b_mat, b_t, [0, 0], order=[1, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.move(a_left, a_mat)
        pl.move(b_right, b_mat)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        pl.matmul(c, a_left, b_right)
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.store(out, c, [0, 0])


def _run(kernel, a_arg, b_arg, a, b):
    device = ST_DEVICE
    out = torch.zeros([M, N], device=device, dtype=torch.float32)
    kernel(a_arg, b_arg, out)
    torch.npu.synchronize()
    ref = torch.matmul(a.float(), b.float())
    diff = (out - ref).abs().max().item()
    logging.info("%s: max|out - A@B| = %s", kernel.__name__, diff)
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


@pytest.fixture(scope="module")
def _ab():
    device = ST_DEVICE
    _require_a5(device)
    torch.manual_seed(0)
    a = torch.randn([M, K], device=device, dtype=torch.float16)
    b = torch.randn([K, N], device=device, dtype=torch.float16)
    return a, b


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_load_nn(_ab):
    """left as-is (NZ [M,K]), right as-is (NZ [K,N])."""
    a, b = _ab
    _run(k_nn, a, b, a, b)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_load_nt(_ab):
    """left as-is; right transpose-loaded from GM B^T [N,K] -> ZN [K,N]."""
    a, b = _ab
    _run(k_nt, a, b.t().contiguous(), a, b)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_load_tn(_ab):
    """left transpose-loaded from GM A^T [K,M] -> ZN [M,K]; right as-is."""
    a, b = _ab
    _run(k_tn, a.t().contiguous(), b, a, b)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_load_tt(_ab):
    """both operands transpose-loaded."""
    a, b = _ab
    _run(k_tt, a.t().contiguous(), b.t().contiguous(), a, b)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    device = ST_DEVICE
    _require_a5(device)
    torch.manual_seed(0)
    a = torch.randn([M, K], device=device, dtype=torch.float16)
    b = torch.randn([K, N], device=device, dtype=torch.float16)
    _run(k_nn, a, b, a, b)
    _run(k_nt, a, b.t().contiguous(), a, b)
    _run(k_tn, a.t().contiguous(), b, a, b)
    _run(k_tt, a.t().contiguous(), b.t().contiguous(), a, b)
