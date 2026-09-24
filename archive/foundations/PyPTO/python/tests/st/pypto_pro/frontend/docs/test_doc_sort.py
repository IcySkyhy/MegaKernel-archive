# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""UT for doc examples — sort ops.

Verifies kernel examples from:
  docs/zh/pypto_pro/api/SIMD-API/operation/memory_vector_computation/sorting/mrgsort.md
  docs/zh/pypto_pro/api/SIMD-API/operation/memory_vector_computation/sorting/mrgsort2.md
  docs/zh/pypto_pro/api/SIMD-API/operation/memory_vector_computation/sorting/sort32.md
"""

import logging
import os

import pypto_pro.language as pl
import pytest
import torch

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"


# ============================================================================
# mrgsort
# ============================================================================


@pl.jit(auto_mutex=True)
def mrgsort_kernel(
    a: pl.Tensor[[1, 1024], pl.DT_FP16],
    sorted_out: pl.Tensor[[1, 1024], pl.DT_FP16],
):
    tt = pl.TileType(shape=[1, 1024], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile_src = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_dst = pl.make_tile_group(type=tt, addrs=0x0800, mutex_ids=[1])
    with pl.section_vector():
        cur_src = tile_src.current()
        cur_dst = tile_dst.current()
        pl.load(cur_src, a, [0, 0])
        pl.mrgsort(cur_dst, cur_src, block_len=256)
        pl.store(sorted_out, cur_dst, [0, 0])


# ============================================================================
# mrgsort2
# ============================================================================


@pl.jit(auto_mutex=True)
def mrgsort2_kernel(
    src0_tensor: pl.Tensor[[1, 256], pl.DT_FP32],
    src1_tensor: pl.Tensor[[1, 256], pl.DT_FP32],
    sorted_out: pl.Tensor[[1, 256], pl.DT_FP32],
):
    tt = pl.TileType(shape=[1, 256], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_src0_group = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_src1_group = pl.make_tile_group(type=tt, addrs=0x0400, mutex_ids=[1])
    tile_dst_group = pl.make_tile_group(type=tt, addrs=0x0800, mutex_ids=[2])
    tile_tmp_group = pl.make_tile_group(type=tt, addrs=0x0C00, mutex_ids=[3])
    with pl.section_vector():
        tile_src0 = tile_src0_group.current()
        tile_src1 = tile_src1_group.current()
        tile_dst = tile_dst_group.current()
        tile_tmp = tile_tmp_group.current()
        pl.load(tile_src0, src0_tensor, [0, 0])
        pl.load(tile_src1, src1_tensor, [0, 0])
        pl.mrgsort2(tile_src0, tile_src1, tile_dst, tile_tmp, exhausted=False)
        pl.store(sorted_out, tile_dst, [0, 0])


# ============================================================================
# sort32
# ============================================================================


@pl.jit(auto_mutex=True)
def sort32_kernel(
    a: pl.Tensor[[1, 32], pl.DT_FP16],
    idx_in: pl.Tensor[[1, 32], pl.DT_UINT32],
    sorted_out: pl.Tensor[[1, 128], pl.DT_FP16],
):
    tt_src = pl.TileType(shape=[1, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tt_dst = pl.TileType(shape=[1, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tt_idx = pl.TileType(shape=[1, 32], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec)
    tile_src = pl.make_tile_group(type=tt_src, addrs=0x0000, mutex_ids=[0])
    tile_dst = pl.make_tile_group(type=tt_dst, addrs=0x0040, mutex_ids=[1])
    tile_idx = pl.make_tile_group(type=tt_idx, addrs=0x0140, mutex_ids=[2])
    with pl.section_vector():
        cur_src = tile_src.current()
        cur_dst = tile_dst.current()
        cur_idx = tile_idx.current()
        pl.load(cur_src, a, [0, 0])
        pl.load(cur_idx, idx_in, [0, 0])
        pl.sort32(cur_dst, cur_src, cur_idx)
        pl.store(sorted_out, cur_dst, [0, 0])


@pl.jit(auto_mutex=True)
def sort32_tail_kernel(
    a: pl.Tensor[[1, 16], pl.DT_FP16],
    idx_in: pl.Tensor[[1, 16], pl.DT_UINT32],
    sorted_out: pl.Tensor[[1, 64], pl.DT_FP16],
):
    tt_src = pl.TileType(shape=[1, 16], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tt_dst = pl.TileType(shape=[1, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tt_idx = pl.TileType(shape=[1, 16], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec)
    tt_tmp = pl.TileType(shape=[1, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile_src = pl.make_tile_group(type=tt_src, addrs=0x0000, mutex_ids=[0])
    tile_dst = pl.make_tile_group(type=tt_dst, addrs=0x0020, mutex_ids=[1])
    tile_idx = pl.make_tile_group(type=tt_idx, addrs=0x00A0, mutex_ids=[2])
    tile_tmp = pl.make_tile_group(type=tt_tmp, addrs=0x0120, mutex_ids=[3])
    with pl.section_vector():
        cur_src = tile_src.current()
        cur_dst = tile_dst.current()
        cur_idx = tile_idx.current()
        cur_tmp = tile_tmp.current()
        pl.load(cur_src, a, [0, 0])
        pl.load(cur_idx, idx_in, [0, 0])
        pl.sort32(cur_dst, cur_src, cur_idx, cur_tmp)
        pl.store(sorted_out, cur_dst, [0, 0])


# ============================================================================
# Test functions
# ============================================================================


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_mrgsort():
    """Smoke test: verify mrgsort kernel compiles and runs without crash.

    Note: mrgsort requires val-idx pair format input with pre-sorted blocks.
    Full correctness testing is in test_mrgsort.py (A3 tests).
    """
    torch.npu.set_device(ST_DEVICE)
    a = torch.zeros(1, 1024, device=ST_DEVICE, dtype=torch.float16)
    # Fill with descending values for pre-sorted blocks
    for i in range(256):
        a[0, i * 4] = float(255 - i)  # val (descending)
        a[0, i * 4 + 1] = 0.0  # pad
        a[0, i * 4 + 2] = float(i & 0xFF)  # idx low byte
        a[0, i * 4 + 3] = 0.0  # idx high byte
    sorted_out = torch.empty(1, 1024, device=ST_DEVICE, dtype=torch.float16)
    mrgsort_kernel(a, sorted_out)
    torch.npu.synchronize()
    logging.info("test_mrgsort passed (smoke test)!")


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_mrgsort2():
    """Smoke test: verify mrgsort2 kernel compiles and runs without crash.

    Note: mrgsort2 requires pre-sorted FP32 val-idx pair inputs.
    Full correctness testing is in test_mrgsort.py (A3 tests).
    """
    torch.npu.set_device(ST_DEVICE)
    src0 = torch.zeros(1, 256, device=ST_DEVICE, dtype=torch.float32)
    src1 = torch.zeros(1, 256, device=ST_DEVICE, dtype=torch.float32)
    for i in range(128):
        src0[0, i * 2] = float(255 - i * 2)  # val (descending, even)
        src0[0, i * 2 + 1] = float(i * 2)  # idx
        src1[0, i * 2] = float(254 - i * 2)  # val (descending, odd)
        src1[0, i * 2 + 1] = float(i * 2 + 1)  # idx
    sorted_out = torch.empty(1, 256, device=ST_DEVICE, dtype=torch.float32)
    mrgsort2_kernel(src0, src1, sorted_out)
    torch.npu.synchronize()
    logging.info("test_mrgsort2 passed (smoke test)!")


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_sort32():
    torch.npu.set_device(ST_DEVICE)
    a = torch.randn(1, 32, device=ST_DEVICE, dtype=torch.float16)
    idx_in = torch.arange(32, dtype=torch.int32).unsqueeze(0).to(torch.uint32).to(ST_DEVICE)
    sorted_out = torch.empty(1, 128, device=ST_DEVICE, dtype=torch.float16)
    sort32_kernel(a, idx_in, sorted_out)
    torch.npu.synchronize()
    logging.info("test_sort32 passed!")


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_sort32_tail():
    torch.npu.set_device(ST_DEVICE)
    a = torch.randn(1, 16, device=ST_DEVICE, dtype=torch.float16)
    idx_in = torch.arange(16, device=ST_DEVICE, dtype=torch.int32).unsqueeze(0).to(torch.uint32)
    sorted_out = torch.empty(1, 64, device=ST_DEVICE, dtype=torch.float16)
    sort32_tail_kernel(a, idx_in, sorted_out)
    torch.npu.synchronize()
    logging.info("test_sort32_tail passed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cases = [test_mrgsort, test_mrgsort2, test_sort32, test_sort32_tail]
    for case in cases:
        case()
    logging.info("\nAll sort tests passed!")
