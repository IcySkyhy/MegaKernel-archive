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

"""Tile subview tests for A[] slice syntax on A5.

3 tests:

  1. Subview without valid_shape: row+col offset, stride mismatch (Vector section)
  2. Subview with runtime valid_shape (pl.set_validshape): intersection clamp
  3. VF subview: 2D row+col offset, per-row load/store with stride
"""

import logging
import os

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813
import pytest
import torch
import torch_npu  # noqa: F401

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"


def _is_a5() -> bool:
    try:
        torch.npu.set_device(ST_DEVICE)
    except RuntimeError as exc:
        logging.info("NPU unavailable: %s", exc)
        return False
    name = torch.npu.get_device_name()
    if "Ascend950" not in name:
        logging.info("Current device is %s, not A5 (Ascend950). Skip.", name)
        return False
    return True


# ===========================================================================
# Test 1: Subview without valid_shape — row+col offset, stride mismatch
#
# tile[2:6, 16:50] on [8, 64]
#   offset = 2*64+16 = 144, valid_shape = [4, 34], stride mismatch (34 != 64)
# ===========================================================================
@pl.jit(auto_mutex=True)
def tile_subview_kernel(
    a: pl.Tensor[[8, 64], pl.DT_FP16],
    out: pl.Tensor[[4, 34], pl.DT_FP16],
):
    tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[8, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids=[0],
    )
    with pl.section_vector():
        tile = tile_group.next()
        pl.load(tile, a, [0, 0])
        sub = tile[2:6, 16:50]
        pl.store(out, sub, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_tile_subview():
    if not _is_a5():
        return
    device = ST_DEVICE
    torch.manual_seed(0)
    a = torch.rand(8, 64, device=device, dtype=torch.float16)
    out = torch.zeros(4, 34, device=device, dtype=torch.float16)

    tile_subview_kernel(a, out)
    torch.npu.synchronize()

    torch.testing.assert_close(out, a[2:6, 16:50], rtol=0, atol=0)
    logging.info("test_tile_subview passed.")


# ===========================================================================
# Test 2: Subview with valid_shape intersection — runtime set_validshape
#
# tile shape [8, 64], valid_shape [6, 40] (via pl.set_validshape), slice [2:8, 8:64]
#
# Covers: rt_valid via set_validshape, intersection clamp
# ===========================================================================
@pl.jit(auto_mutex=True)
def tile_subview_vshape_kernel(
    a: pl.Tensor[[8, 64], pl.DT_FP16],
    out1: pl.Tensor[[4, 24], pl.DT_FP16],
):
    tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[8, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]),
        addrs=0x0000,
        mutex_ids=[0],
    )
    with pl.section_vector():
        tile = tile_group.next()
        pl.set_validshape(tile, [6, 40])
        pl.load(tile, a, [0, 0])
        sub1 = tile[2:6, 16:50]
        pl.store(out1, sub1, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_tile_subview_valid_shape():
    if not _is_a5():
        return
    device = ST_DEVICE
    torch.manual_seed(1)
    a = torch.rand(8, 64, device=device, dtype=torch.float16)
    out1 = torch.zeros(4, 24, device=device, dtype=torch.float16)

    tile_subview_vshape_kernel(a, out1)
    torch.npu.synchronize()

    expected = a[2:6, 16:40]
    torch.testing.assert_close(out1, expected, rtol=0, atol=0)
    logging.info("test_tile_subview_valid_shape passed.")


# ===========================================================================
# Test 3: VF subview — 2D row+col offset, per-row load/store with stride
#
# vf_src tile shape [4, 64], subview [1:4, 16:48] → 3 rows × 32 cols
# VF function loops over 3 rows, each row loads 32 FP16 from offset
# (row_start * 64 + col_start) and stores to dst row-by-row.
#
# Covers: VF pointer arithmetic with 2D subview, per-row stride walk
# ===========================================================================
@pl.vector_function
def vf_copy_subview_row(dst_tile, src_tile, n_rows, n_cols, stride):
    preg = vf.update_mask(n_cols, dtype=pl.DT_FP16)
    for r in pl.range(n_rows):
        vreg = vf.load_align(src_tile, r * stride)
        vf.store_align(dst_tile + r * n_cols, vreg, preg)


@pl.jit(auto_mutex=True)
def tile_subview_vf_kernel(
    vf_a: pl.Tensor[[4, 64], pl.DT_FP16],
    vf_out: pl.Tensor[[3, 32], pl.DT_FP16],
):
    vf_src_group = pl.make_tile_group(
        type=pl.TileType(shape=[4, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x4000,
        mutex_ids=[1],
    )
    vf_dst_group = pl.make_tile_group(
        type=pl.TileType(shape=[3, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x5000,
        mutex_ids=[2],
    )
    with pl.section_vector():
        vf_src = vf_src_group.next()
        vf_dst = vf_dst_group.next()
        pl.load(vf_src, vf_a, [0, 0])
        vf_copy_subview_row(vf_dst, vf_src[1:4, 16:48], 3, 32, 64)
        pl.store(vf_out, vf_dst, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_tile_subview_vf():
    if not _is_a5():
        return
    device = ST_DEVICE
    torch.manual_seed(2)
    vf_a = torch.rand(4, 64, device=device, dtype=torch.float16)
    vf_out = torch.zeros(3, 32, device=device, dtype=torch.float16)

    tile_subview_vf_kernel(vf_a, vf_out)
    torch.npu.synchronize()

    torch.testing.assert_close(vf_out, vf_a[1:4, 16:48], rtol=0, atol=0)
    logging.info("test_tile_subview_vf passed.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    test_tile_subview()
    test_tile_subview_valid_shape()
    test_tile_subview_vf()
    logging.info("\nAll subview tests passed!")
