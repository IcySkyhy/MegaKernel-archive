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

"""order kwarg passed via a variable (regression for issue_3).

``pl.load`` / ``pl.load_tile`` accept ``order`` as a compile-time attribute kwarg.
A literal ``order=[1, 0]`` works, but threading the value through a variable used to
fail with "Kwarg 'order' ... has incompatible type" because the parser produced a
runtime IR Expr instead of folding it to a Python ``list``.

This module covers three forms on the same TN matmul (C = A @ B, A supplied transposed):

  * literal      — ``order=[1, 0]`` (baseline, always worked)
  * local var    — ``flag = [1, 0]; pl.load(..., order=flag)`` (assignment-fold path)
  * helper param — ``load_left(dst, src, order)`` inlined helper (inline-fold path)

All three must compile and produce the same result as ``torch.matmul(A.float(), B.float())``.
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


def load_left(dst, src, order):
    """Inlined helper that threads ``order`` as a parameter (the issue scenario)."""
    pl.load(dst, src, [0, 0], order=order)


def _make_tiles():
    a_mat = pl.make_tile(
        pl.TileType(shape=[M, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN), addr=MA0, size=SZ
    )
    b_mat = pl.make_tile(
        pl.TileType(shape=[K, N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ), addr=MA1, size=SZ
    )
    a_left = pl.make_tile(
        pl.TileType(shape=[M, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ), addr=0x0, size=SZ
    )
    b_right = pl.make_tile(
        pl.TileType(shape=[K, N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right, layout=pl.ZN), addr=0x0, size=SZ
    )
    c = pl.make_tile(
        pl.TileType(shape=[M, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024),
        addr=0x0,
        size=SZ,
    )
    return a_mat, b_mat, a_left, b_right, c


def _matmul_body(a_mat, b_mat, a_left, b_right, c, a_t, b, out):
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
def k_transpose_lit(
    a_t: pl.Tensor[[K, M], pl.DT_FP16], b: pl.Tensor[[K, N], pl.DT_FP16], out: pl.Tensor[[M, N], pl.DT_FP32]
):
    with pl.section_cube():
        a_mat, b_mat, a_left, b_right, c = _make_tiles()
        pl.load(a_mat, a_t, [0, 0], order=[1, 0])
        _matmul_body(a_mat, b_mat, a_left, b_right, c, a_t, b, out)


@pl.jit()
def k_transpose_local_var(
    a_t: pl.Tensor[[K, M], pl.DT_FP16], b: pl.Tensor[[K, N], pl.DT_FP16], out: pl.Tensor[[M, N], pl.DT_FP32]
):
    with pl.section_cube():
        a_mat, b_mat, a_left, b_right, c = _make_tiles()
        flag = [1, 0]
        pl.load(a_mat, a_t, [0, 0], order=flag)
        _matmul_body(a_mat, b_mat, a_left, b_right, c, a_t, b, out)


@pl.jit()
def k_transpose_helper_var(
    a_t: pl.Tensor[[K, M], pl.DT_FP16], b: pl.Tensor[[K, N], pl.DT_FP16], out: pl.Tensor[[M, N], pl.DT_FP32]
):
    with pl.section_cube():
        a_mat, b_mat, a_left, b_right, c = _make_tiles()
        load_left(a_mat, a_t, [1, 0])
        _matmul_body(a_mat, b_mat, a_left, b_right, c, a_t, b, out)


@pytest.fixture(scope="module")
def _ab():
    device = ST_DEVICE
    _require_a5(device)
    torch.manual_seed(0)
    a = torch.randn([M, K], device=device, dtype=torch.float16)
    b = torch.randn([K, N], device=device, dtype=torch.float16)
    return a, b


def _run(kernel, a_arg, b_arg, a, b):
    device = ST_DEVICE
    out = torch.zeros([M, N], device=device, dtype=torch.float32)
    kernel(a_arg, b_arg, out)
    torch.npu.synchronize()
    ref = torch.matmul(a.float(), b.float())
    diff = (out - ref).abs().max().item()
    logging.info("%s: max|out - A@B| = %s", kernel.__name__, diff)
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_transpose_literal(_ab):
    a, b = _ab
    _run(k_transpose_lit, a.t().contiguous(), b, a, b)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_transpose_local_var(_ab):
    """order from a local list assignment (assignment-fold path)."""
    a, b = _ab
    _run(k_transpose_local_var, a.t().contiguous(), b, a, b)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_transpose_helper_var(_ab):
    """order threaded through an inlined helper parameter (inline-fold path)."""
    a, b = _ab
    _run(k_transpose_helper_var, a.t().contiguous(), b, a, b)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    device = ST_DEVICE
    _require_a5(device)
    torch.manual_seed(0)
    a = torch.randn([M, K], device=device, dtype=torch.float16)
    b = torch.randn([K, N], device=device, dtype=torch.float16)
    _run(k_transpose_lit, a.t().contiguous(), b, a, b)
    _run(k_transpose_local_var, a.t().contiguous(), b, a, b)
    _run(k_transpose_helper_var, a.t().contiguous(), b, a, b)
