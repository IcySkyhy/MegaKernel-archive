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

"""Test mutex dedup with double-buffer inplace cast (dynamic mutex_id).

Two tile_groups share the same address and mutex_ids=[4,5] (double buffer).
pl.cast from one to the other is an inplace op. The framework should generate
runtime if-else dedup for get_buf/rls_buf (since mutex_id is dynamic at runtime).

This test does a single cast (no loop) to verify the dedup codegen works without
cursor synchronization concerns.
"""

import logging
import os

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813
import pytest
import torch

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"


TILE_M = 64
TILE_N = 64


@pl.jit(auto_mutex=True)
def cast_dedup_db_kernel(
    src: pl.Tensor[[TILE_M, TILE_N], pl.DT_FP32],
    dst: pl.Tensor[[TILE_M, TILE_N], pl.DT_FP16],
):
    """Single cast FP32->FP16 with inplace double-buffer groups (verifies dedup codegen)."""
    tile_type_f32 = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_type_f16 = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)

    # Double buffer groups sharing address and mutex_ids (inplace alias)
    # FP16 tile occupies less bytes than FP32, so they can share the same base.
    f32_group = pl.make_tile_group(type=tile_type_f32, addrs=0x0000, mutex_ids=[4, 5])
    f16_group = pl.make_tile_group(type=tile_type_f16, addrs=0x0000, mutex_ids=[4, 5])

    with pl.section_vector():
        # Both call next() once — cursors start at same initial value, will select same slot
        f32_tile = f32_group.next()
        f16_tile = f16_group.next()

        # Load FP32 source
        pl.load(f32_tile, src, [0, 0])
        # In-place cast FP32 -> FP16 (TCVT supports this)
        pl.cast(f16_tile, f32_tile, mode=pl.RoundMode.CAST_ROUND)
        # Store FP16 result
        pl.store(dst, f16_tile, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_cast_dedup_double_buffer():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)

    src = torch.rand((TILE_M, TILE_N), device=device, dtype=torch.float32)
    dst = torch.zeros((TILE_M, TILE_N), device=device, dtype=torch.float16)

    cast_dedup_db_kernel[None, 1](src, dst)
    torch.npu.synchronize()

    expected = src.half()
    torch.testing.assert_close(dst, expected, rtol=1e-3, atol=1e-3)
    logging.info("test_cast_dedup_double_buffer PASSED")


# =========================================================================
# VF function version: dedup at VF function entry/exit
# =========================================================================


@pl.vector_function
def vf_inplace_muls(src_tile, dst_tile, scale):
    """VF function that reads src_tile and writes dst_tile (inplace aliased)."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    vreg = vf.load_align(src_tile, 0)
    vreg = vf.muls(vreg, scale, preg)
    vf.store_align(dst_tile, vreg, preg)


# VF processes one vreg per load/store; use a [1, 64] tile so a single
# load/muls/store covers the whole tile.
VF_M = 1
VF_N = 64


@pl.jit(auto_mutex=True)
def vf_dedup_db_kernel(
    src: pl.Tensor[[VF_M, VF_N], pl.DT_FP32],
    dst: pl.Tensor[[VF_M, VF_N], pl.DT_FP32],
):
    """VF function with two inplace-aliased double-buffer tiles (verifies VF dedup)."""
    tile_type = pl.TileType(shape=[VF_M, VF_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)

    # Two groups: same address, same mutex_ids — inplace alias (double buffer)
    in_group = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[4, 5])
    out_group = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[4, 5])

    with pl.section_vector():
        in_tile = in_group.next()
        out_tile = out_group.next()

        # Load data
        pl.load(in_tile, src, [0, 0])
        # VF function: reads in_tile, writes out_tile (same physical buffer)
        # Framework should dedup the VF entry lock (both tiles share mutex_ids)
        vf_inplace_muls(in_tile, out_tile, 2.0)
        # Store result
        pl.store(dst, out_tile, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_vf_dedup_double_buffer():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)

    src = torch.rand((VF_M, VF_N), device=device, dtype=torch.float32)
    dst = torch.zeros((VF_M, VF_N), device=device, dtype=torch.float32)

    vf_dedup_db_kernel[None, 1](src, dst)
    torch.npu.synchronize()

    expected = src * 2.0
    torch.testing.assert_close(dst, expected, rtol=1e-3, atol=1e-3)
    logging.info("test_vf_dedup_double_buffer PASSED")


# =========================================================================
# U4: Partial overlap [1] vs [1,2] (single/double buffer mix)
# =========================================================================


@pl.vector_function
def vf_muls_u4(tile_a, tile_b, scale):
    """VF: reads tile_a (single buf), writes tile_b (double buf). Partial overlap."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    vreg = vf.load_align(tile_a, 0)
    vreg = vf.muls(vreg, scale, preg)
    vf.store_align(tile_b, vreg, preg)


@pl.jit(auto_mutex=True)
def vf_partial_overlap_kernel(
    src: pl.Tensor[[VF_M, VF_N], pl.DT_FP32],
    dst: pl.Tensor[[VF_M, VF_N], pl.DT_FP32],
):
    tile_type = pl.TileType(shape=[VF_M, VF_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)

    group_a = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[1])
    group_b = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[1, 2])

    with pl.section_vector():
        tile_a = group_a.next()
        tile_b = group_b.next()
        pl.load(tile_a, src, [0, 0])
        vf_muls_u4(tile_a, tile_b, 3.0)
        pl.store(dst, tile_b, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_vf_dedup_partial_overlap():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)

    src = torch.rand((VF_M, VF_N), device=device, dtype=torch.float32)
    dst = torch.zeros((VF_M, VF_N), device=device, dtype=torch.float32)

    vf_partial_overlap_kernel[None, 1](src, dst)
    torch.npu.synchronize()

    expected = src * 3.0
    torch.testing.assert_close(dst, expected, rtol=1e-3, atol=1e-3)
    logging.info("test_vf_dedup_partial_overlap PASSED")


# =========================================================================
# U5: Three transitive overlap [0,1] [1,2] [2,3]
# =========================================================================


@pl.vector_function
def vf_muls_u5(tile_a, tile_b, tile_c, scale):
    """VF: reads tile_a, writes tile_b and tile_c. Three transitive overlap."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    vreg = vf.load_align(tile_a, 0)
    vreg = vf.muls(vreg, scale, preg)
    vf.store_align(tile_b, vreg, preg)
    vf.store_align(tile_c, vreg, preg)


@pl.jit(auto_mutex=True)
def vf_three_transitive_kernel(
    src: pl.Tensor[[VF_M, VF_N], pl.DT_FP32],
    dst: pl.Tensor[[VF_M, VF_N], pl.DT_FP32],
):
    """Three groups: [0,1] [1,2] [2,3]. A-B overlap, B-C overlap, A-C no direct overlap.
    All three connected by transitivity -> single dedup group, 3-way if-guard.
    """
    tile_type = pl.TileType(shape=[VF_M, VF_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)

    group_a = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0, 1])
    group_b = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[1, 2])
    group_c = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[2, 3])

    with pl.section_vector():
        tile_a = group_a.next()
        tile_b = group_b.next()
        tile_c = group_c.next()
        pl.load(tile_a, src, [0, 0])
        vf_muls_u5(tile_a, tile_b, tile_c, 4.0)
        pl.store(dst, tile_c, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_vf_dedup_three_transitive():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)

    src = torch.rand((VF_M, VF_N), device=device, dtype=torch.float32)
    dst = torch.zeros((VF_M, VF_N), device=device, dtype=torch.float32)

    vf_three_transitive_kernel[None, 1](src, dst)
    torch.npu.synchronize()

    expected = src * 4.0
    torch.testing.assert_close(dst, expected, rtol=1e-3, atol=1e-3)
    logging.info("test_vf_dedup_three_transitive PASSED")


# =========================================================================
# U6: No overlap [0] vs [1] (negative test — should NOT dedup)
# =========================================================================


@pl.vector_function
def vf_muls_u6(tile_a, tile_b, scale):
    """VF: reads tile_a, writes tile_b. No overlap — should get two independent locks."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    vreg = vf.load_align(tile_a, 0)
    vreg = vf.muls(vreg, scale, preg)
    vf.store_align(tile_b, vreg, preg)


@pl.jit(auto_mutex=True)
def vf_no_overlap_kernel(
    src: pl.Tensor[[VF_M, VF_N], pl.DT_FP32],
    dst: pl.Tensor[[VF_M, VF_N], pl.DT_FP32],
):
    tile_type = pl.TileType(shape=[VF_M, VF_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)

    group_a = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0])
    group_b = pl.make_tile_group(type=tile_type, addrs=0x1000, mutex_ids=[1])

    with pl.section_vector():
        tile_a = group_a.next()
        tile_b = group_b.next()
        pl.load(tile_a, src, [0, 0])
        vf_muls_u6(tile_a, tile_b, 5.0)
        pl.store(dst, tile_b, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_vf_no_dedup_no_overlap():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)

    src = torch.rand((VF_M, VF_N), device=device, dtype=torch.float32)
    dst = torch.zeros((VF_M, VF_N), device=device, dtype=torch.float32)

    vf_no_overlap_kernel[None, 1](src, dst)
    torch.npu.synchronize()

    expected = src * 5.0
    torch.testing.assert_close(dst, expected, rtol=1e-3, atol=1e-3)
    logging.info("test_vf_no_dedup_no_overlap PASSED")


# =========================================================================
# U7: Partial group [0,1] [1,2] [5,6] — A,B grouped; C independent
# =========================================================================


@pl.vector_function
def vf_muls_u7(tile_a, tile_b, tile_c, scale):
    """VF: reads tile_a, writes tile_b and tile_c. A,B overlap; C independent."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    vreg = vf.load_align(tile_a, 0)
    vreg = vf.muls(vreg, scale, preg)
    vf.store_align(tile_b, vreg, preg)
    vf.store_align(tile_c, vreg, preg)


@pl.jit(auto_mutex=True)
def vf_partial_group_kernel(
    src: pl.Tensor[[VF_M, VF_N], pl.DT_FP32],
    dst: pl.Tensor[[VF_M, VF_N], pl.DT_FP32],
):
    """[0,1] [1,2] [5,6]: A-B overlap (dedup), C independent (plain lock)."""
    tile_type = pl.TileType(shape=[VF_M, VF_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)

    group_a = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0, 1])
    group_b = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[1, 2])
    group_c = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[5, 6])

    with pl.section_vector():
        tile_a = group_a.next()
        tile_b = group_b.next()
        tile_c = group_c.next()
        pl.load(tile_a, src, [0, 0])
        vf_muls_u7(tile_a, tile_b, tile_c, 6.0)
        pl.store(dst, tile_c, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_vf_dedup_partial_group():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)

    src = torch.rand((VF_M, VF_N), device=device, dtype=torch.float32)
    dst = torch.zeros((VF_M, VF_N), device=device, dtype=torch.float32)

    vf_partial_group_kernel[None, 1](src, dst)
    torch.npu.synchronize()

    expected = src * 6.0
    torch.testing.assert_close(dst, expected, rtol=1e-3, atol=1e-3)
    logging.info("test_vf_dedup_partial_group PASSED")


# =========================================================================
# U8: Three same [1,2] [1,2] [1,2] — all dynamic, all same group
# =========================================================================


@pl.vector_function
def vf_muls_u8(tile_a, tile_b, tile_c, scale):
    """VF: reads tile_a, writes tile_b and tile_c. All three same mutex_ids."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    vreg = vf.load_align(tile_a, 0)
    vreg = vf.muls(vreg, scale, preg)
    vf.store_align(tile_b, vreg, preg)
    vf.store_align(tile_c, vreg, preg)


@pl.jit(auto_mutex=True)
def vf_three_same_kernel(
    src: pl.Tensor[[VF_M, VF_N], pl.DT_FP32],
    dst: pl.Tensor[[VF_M, VF_N], pl.DT_FP32],
):
    """Three groups all [1,2]: all dynamic, fully overlapping -> 3-way dedup if-guard."""
    tile_type = pl.TileType(shape=[VF_M, VF_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)

    group_a = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[1, 2])
    group_b = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[1, 2])
    group_c = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[1, 2])

    with pl.section_vector():
        tile_a = group_a.next()
        tile_b = group_b.next()
        tile_c = group_c.next()
        pl.load(tile_a, src, [0, 0])
        vf_muls_u8(tile_a, tile_b, tile_c, 7.0)
        pl.store(dst, tile_c, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_vf_dedup_three_same():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)

    src = torch.rand((VF_M, VF_N), device=device, dtype=torch.float32)
    dst = torch.zeros((VF_M, VF_N), device=device, dtype=torch.float32)

    vf_three_same_kernel[None, 1](src, dst)
    torch.npu.synchronize()

    expected = src * 7.0
    torch.testing.assert_close(dst, expected, rtol=1e-3, atol=1e-3)
    logging.info("test_vf_dedup_three_same PASSED")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_cast_dedup_double_buffer()
    test_vf_dedup_double_buffer()
    test_vf_dedup_partial_overlap()
    test_vf_dedup_three_transitive()
    test_vf_no_dedup_no_overlap()
    test_vf_dedup_partial_group()
    test_vf_dedup_three_same()
