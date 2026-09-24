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

"""Tile-group (double-buffered) variants of test_load_transpose_convention.py.

Same GM-load clear-form convention, but every buffer is a `pl.make_tile_group` accessed via
`.next()` (dynamic cursor). This exercises NormalizeMatTransposeLayout on multi-slot groups:
a reversed Mat->L0 move on a group slot must relabel ALL sibling slots (and flip their feeding
load to is_transpose), not just the one the collector happens to see.

  C[M, N] = A[M, K] @ B[K, N]   (non-square to expose an accidental transpose)
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


def _require_a5(device):
    try:
        torch.npu.set_device(device)
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
    if "Ascend950" not in torch.npu.get_device_name():
        pytest.skip("not A5")


# nn: left as-is (NZ [M,K]), right as-is (NZ [K,N]).
@pl.jit(auto_mutex=True)
def k_nn_group(a: pl.Tensor[[M, K], pl.DT_FP16], b: pl.Tensor[[K, N], pl.DT_FP16], out: pl.Tensor[[M, N], pl.DT_FP32]):
    a_mat_db = pl.make_tile_group(
        type=pl.TileType(shape=[M, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x0,
        mutex_ids=[0, 1],
    )
    b_mat_db = pl.make_tile_group(
        type=pl.TileType(shape=[K, N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x10000,
        mutex_ids=[2, 3],
    )
    a_left_db = pl.make_tile_group(
        type=pl.TileType(shape=[M, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=0x0,
        mutex_ids=[4, 5],
    )
    b_right_db = pl.make_tile_group(
        type=pl.TileType(shape=[K, N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=0x0,
        mutex_ids=[6, 7],
    )
    acc_db = pl.make_tile_group(
        type=pl.TileType(shape=[M, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024),
        addrs=0x0,
        mutex_ids=[8, 9],
    )
    with pl.section_cube():
        a_mat = a_mat_db.next()
        b_mat = b_mat_db.next()
        a_left = a_left_db.next()
        b_right = b_right_db.next()
        acc = acc_db.next()
        pl.load(a_mat, a, [0, 0])
        pl.load(b_mat, b, [0, 0])
        pl.move(a_left, a_mat)
        pl.move(b_right, b_mat)
        pl.matmul(acc, a_left, b_right)
        pl.store(out, acc, [0, 0])


# nt: right transpose-loaded from GM B^T [N,K] into a clear-form NZ [N,K] group.
@pl.jit(auto_mutex=True)
def k_nt_group(
    a: pl.Tensor[[M, K], pl.DT_FP16],
    b_t: pl.Tensor[[N, K], pl.DT_FP16],
    out: pl.Tensor[[M, N], pl.DT_FP32],
):
    a_mat_db = pl.make_tile_group(
        type=pl.TileType(shape=[M, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x0,
        mutex_ids=[0, 1],
    )
    b_mat_db = pl.make_tile_group(
        type=pl.TileType(shape=[K, N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
        addrs=0x10000,
        mutex_ids=[2, 3],
    )
    a_left_db = pl.make_tile_group(
        type=pl.TileType(shape=[M, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=0x0,
        mutex_ids=[4, 5],
    )
    b_right_db = pl.make_tile_group(
        type=pl.TileType(shape=[K, N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=0x0,
        mutex_ids=[6, 7],
    )
    acc_db = pl.make_tile_group(
        type=pl.TileType(shape=[M, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024),
        addrs=0x0,
        mutex_ids=[8, 9],
    )
    with pl.section_cube():
        a_mat = a_mat_db.next()  # noqa: F841
        a_mat_temp1 = a_mat_db.current()
        a_mat_temp = a_mat_temp1
        b_mat = b_mat_db.next()
        a_left = a_left_db.next()
        b_right = b_right_db.next()
        acc = acc_db.next()
        pl.load(a_mat_temp, a, [0, 0])
        pl.load(b_mat, b_t, [0, 0], order=[1, 0])
        pl.move(a_left, a_mat_temp)
        pl.move(b_right, b_mat)  # Mat[N,K] group -> Right[K,N]: reversed -> swap all slots
        pl.matmul(acc, a_left, b_right)
        pl.store(out, acc, [0, 0])


# tn: left transpose-loaded from GM A^T [K,M] into a clear-form NZ [K,M] group.
@pl.jit(auto_mutex=True)
def k_tn_group(
    a_t: pl.Tensor[[K, M], pl.DT_FP16],
    b: pl.Tensor[[K, N], pl.DT_FP16],
    out: pl.Tensor[[M, N], pl.DT_FP32],
):
    a_mat_db = pl.make_tile_group(
        type=pl.TileType(shape=[M, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
        addrs=0x0,
        mutex_ids=[0, 1],
    )
    b_mat_db = pl.make_tile_group(
        type=pl.TileType(shape=[K, N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x10000,
        mutex_ids=[2, 3],
    )
    a_left_db = pl.make_tile_group(
        type=pl.TileType(shape=[M, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=0x0,
        mutex_ids=[4, 5],
    )
    b_right_db = pl.make_tile_group(
        type=pl.TileType(shape=[K, N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=0x0,
        mutex_ids=[6, 7],
    )
    acc_db = pl.make_tile_group(
        type=pl.TileType(shape=[M, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024),
        addrs=0x0,
        mutex_ids=[8, 9],
    )
    with pl.section_cube():
        a_mat = a_mat_db.next()
        b_mat = b_mat_db.next()
        a_left = a_left_db.next()
        b_right = b_right_db.next()
        acc = acc_db.next()
        pl.load(a_mat, a_t, [0, 0], order=[1, 0])
        pl.load(b_mat, b, [0, 0])
        pl.move(a_left, a_mat)  # Mat[K,M] group -> Left[M,K]: reversed -> swap all slots
        pl.move(b_right, b_mat)
        pl.matmul(acc, a_left, b_right)
        pl.store(out, acc, [0, 0])


# tt: both operands transpose-loaded into clear-form NZ groups.
@pl.jit(auto_mutex=True)
def k_tt_group(
    a_t: pl.Tensor[[K, M], pl.DT_FP16],
    b_t: pl.Tensor[[N, K], pl.DT_FP16],
    out: pl.Tensor[[M, N], pl.DT_FP32],
):
    a_mat_db = pl.make_tile_group(
        type=pl.TileType(shape=[M, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
        addrs=0x0,
        mutex_ids=[0, 1],
    )
    b_mat_db = pl.make_tile_group(
        type=pl.TileType(shape=[K, N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
        addrs=0x10000,
        mutex_ids=[2, 3],
    )
    a_left_db = pl.make_tile_group(
        type=pl.TileType(shape=[M, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=0x0,
        mutex_ids=[4, 5],
    )
    b_right_db = pl.make_tile_group(
        type=pl.TileType(shape=[K, N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=0x0,
        mutex_ids=[6, 7],
    )
    acc_db = pl.make_tile_group(
        type=pl.TileType(shape=[M, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024),
        addrs=0x0,
        mutex_ids=[8, 9],
    )
    with pl.section_cube():
        a_mat = a_mat_db.next()
        b_mat = b_mat_db.next()
        a_left = a_left_db.next()
        b_right = b_right_db.next()
        acc = acc_db.next()
        pl.load(a_mat, a_t, [0, 0], order=[1, 0])
        pl.load(b_mat, b_t, [0, 0], order=[1, 0])
        pl.move(a_left, a_mat)  # reversed -> swap all slots
        pl.move(b_right, b_mat)  # reversed -> swap all slots
        pl.matmul(acc, a_left, b_right)
        pl.store(out, acc, [0, 0])


def _run(kernel, a_arg, b_arg, a, b):
    out = torch.zeros([M, N], device=ST_DEVICE, dtype=torch.float32)
    kernel(a_arg, b_arg, out)
    torch.npu.synchronize()
    ref = torch.matmul(a.float(), b.float())
    diff = (out - ref).abs().max().item()
    logging.info("%s: max|out - A@B| = %s", kernel.__name__, diff)
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


@pytest.fixture(scope="module")
def _ab():
    _require_a5(ST_DEVICE)
    torch.manual_seed(0)
    a = torch.randn([M, K], device=ST_DEVICE, dtype=torch.float16)
    b = torch.randn([K, N], device=ST_DEVICE, dtype=torch.float16)
    return a, b


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_group_nn(_ab):
    a, b = _ab
    _run(k_nn_group, a, b, a, b)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_group_nt(_ab):
    a, b = _ab
    _run(k_nt_group, a, b.t().contiguous(), a, b)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_group_tn(_ab):
    a, b = _ab
    _run(k_tn_group, a.t().contiguous(), b, a, b)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_group_tt(_ab):
    a, b = _ab
    _run(k_tt_group, a.t().contiguous(), b.t().contiguous(), a, b)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _require_a5(ST_DEVICE)
    torch.manual_seed(0)
    a = torch.randn([M, K], device=ST_DEVICE, dtype=torch.float16)
    b = torch.randn([K, N], device=ST_DEVICE, dtype=torch.float16)
    _run(k_nn_group, a, b, a, b)
    _run(k_nt_group, a, b.t().contiguous(), a, b)
    _run(k_tn_group, a.t().contiguous(), b, a, b)
    _run(k_tt_group, a.t().contiguous(), b.t().contiguous(), a, b)
