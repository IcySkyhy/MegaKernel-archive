# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Gather variants: gathermask (pattern_mode) and gatherb (byte-offset block gather).

Doc: docs/zh/pypto_pro/api/SIMD-API/计算API/Memory矢量计算/离散与聚合

Pure vector mode with make_tile_group and auto_mutex. Verifies masked column
selection and byte-offset block gather with identity offsets.
"""

import logging
import os

import pypto_pro.language as pl
import pytest
import torch

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

# gatherb 常量
DTYPE_SIZE = 2  # FP16
BLOCK_BYTES = 32
BLOCK_ELEMS = BLOCK_BYTES // DTYPE_SIZE  # 16
ROWS, COLS = 64, 128
OFFSETS_PER_ROW = COLS // BLOCK_ELEMS  # 8


def _require_a5(device):
    try:
        torch.npu.set_device(device)
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
    name = torch.npu.get_device_name()
    if "Ascend950" not in name:
        pytest.skip(f"Current device is {name}, not A5 (Ascend950). Skip.")


# ===========================================================================
# gathermask —— 按位模式抽取列。调用顺序 (dst, src, pattern_mode=)
# ===========================================================================
@pl.jit(auto_mutex=True)
def gathermask_p1_kernel(
    src: pl.Tensor[[64, 128], pl.DT_FP16],
    dst: pl.Tensor[[64, 64], pl.DT_FP16],
):
    tile_src_group = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids=[0],
    )
    tile_dst_group = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x4000,
        mutex_ids=[1],
    )
    with pl.section_vector():
        tile_src = tile_src_group.current()
        tile_dst = tile_dst_group.current()
        pl.load(tile_src, src, [0, 0])
        pl.gathermask(tile_dst, tile_src, pattern_mode=1)
        pl.store(dst, tile_dst, [0, 0])


@pl.jit(auto_mutex=True)
def gathermask_p2_kernel(
    src: pl.Tensor[[64, 128], pl.DT_FP16],
    dst: pl.Tensor[[64, 64], pl.DT_FP16],
):
    tile_src_group = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids=[0],
    )
    tile_dst_group = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x4000,
        mutex_ids=[1],
    )
    with pl.section_vector():
        tile_src = tile_src_group.current()
        tile_dst = tile_dst_group.current()
        pl.load(tile_src, src, [0, 0])
        pl.gathermask(tile_dst, tile_src, pattern_mode=2)
        pl.store(dst, tile_dst, [0, 0])


@pl.jit(auto_mutex=True)
def gathermask_p7_kernel(
    src: pl.Tensor[[64, 128], pl.DT_FP16],
    dst: pl.Tensor[[64, 128], pl.DT_FP16],
):
    tile_src_group = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids=[0],
    )
    tile_dst_group = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x4000,
        mutex_ids=[1],
    )
    with pl.section_vector():
        tile_src = tile_src_group.current()
        tile_dst = tile_dst_group.current()
        pl.load(tile_src, src, [0, 0])
        pl.gathermask(tile_dst, tile_src, pattern_mode=7)
        pl.store(dst, tile_dst, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_gathermask_p1():
    device = ST_DEVICE
    _require_a5(device)
    torch.manual_seed(0)
    src = torch.rand(64, 128, device=device, dtype=torch.float16)
    dst = torch.zeros(64, 64, device=device, dtype=torch.float16)
    gathermask_p1_kernel(src, dst)
    torch.npu.synchronize()
    torch.testing.assert_close(dst, src[:, 0::2].contiguous(), rtol=1e-3, atol=1e-3)
    logging.info("gathermask pattern_mode=1 result equal!")


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_gathermask_p2():
    device = ST_DEVICE
    _require_a5(device)
    torch.manual_seed(0)
    src = torch.rand(64, 128, device=device, dtype=torch.float16)
    dst = torch.zeros(64, 64, device=device, dtype=torch.float16)
    gathermask_p2_kernel(src, dst)
    torch.npu.synchronize()
    torch.testing.assert_close(dst, src[:, 1::2].contiguous(), rtol=1e-3, atol=1e-3)
    logging.info("gathermask pattern_mode=2 result equal!")


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_gathermask_p7():
    device = ST_DEVICE
    _require_a5(device)
    torch.manual_seed(0)
    src = torch.rand(64, 128, device=device, dtype=torch.float16)
    dst = torch.zeros(64, 128, device=device, dtype=torch.float16)
    gathermask_p7_kernel(src, dst)
    torch.npu.synchronize()
    torch.testing.assert_close(dst, src, rtol=1e-3, atol=1e-3)
    logging.info("gathermask pattern_mode=7 (copy) result equal!")


# ===========================================================================
# gatherb —— 按 32 字节块的字节偏移聚合。offsets [64,8] UINT32。
#   identity offset：offset[i,j] = (i*8 + j)*32 字节，复现 src。
# ===========================================================================
@pl.jit(auto_mutex=True)
def gatherb_kernel(
    src: pl.Tensor[[64, 128], pl.DT_FP16],
    offsets: pl.Tensor[[64, 8], pl.DT_UINT32],
    dst: pl.Tensor[[64, 128], pl.DT_FP16],
):
    tile_src_group = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids=[0],
    )
    tile_offsets_group = pl.make_tile_group(
        type=pl.TileType(shape=[64, 8], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x4000,
        mutex_ids=[1],
    )
    tile_dst_group = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x4800,
        mutex_ids=[2],
    )
    with pl.section_vector():
        tile_src = tile_src_group.current()
        tile_offsets = tile_offsets_group.current()
        tile_dst = tile_dst_group.current()
        pl.load(tile_src, src, [0, 0])
        pl.load(tile_offsets, offsets, [0, 0])
        pl.gatherb(tile_dst, tile_src, tile_offsets)
        pl.store(dst, tile_dst, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_gatherb_identity():
    device = ST_DEVICE
    _require_a5(device)
    torch.manual_seed(0)
    src = torch.rand([ROWS, COLS], device=device, dtype=torch.float16)
    # identity：每个 offset 指向行顺序对应的 32 字节块
    total_blocks = ROWS * OFFSETS_PER_ROW
    offsets = (torch.arange(total_blocks, device=device, dtype=torch.int32) * BLOCK_BYTES).reshape(
        ROWS, OFFSETS_PER_ROW
    )
    dst = torch.empty([ROWS, COLS], device=device, dtype=torch.float16)
    gatherb_kernel(src, offsets, dst)
    torch.npu.synchronize()
    # identity offset 复现 src
    torch.testing.assert_close(dst, src, rtol=1e-3, atol=1e-3)
    logging.info("gatherb identity result equal!")


def _doc_gathermask(ctx, kernel):
    src = ctx.base_fp16((64, 128), 0.25, 1.0)
    dst = torch.zeros((64, 64), device=ctx.device, dtype=torch.float16)
    kernel(src, dst)
    ctx.synchronize()
    return ctx.snippet("gathermask", {"src": src, "pattern_mode": torch.tensor(1)}, {"dst": dst})


def _doc_gatherb(ctx, kernel):
    src = ctx.base_fp16((64, 128), 0.25, 1.0)
    offsets = (torch.arange(64 * 8, device=ctx.device, dtype=torch.int32) * 32).reshape(64, 8)
    dst = torch.empty((64, 128), device=ctx.device, dtype=torch.float16)
    kernel(src, offsets, dst)
    ctx.synchronize()
    return ctx.snippet("gatherb", {"src": src, "offsets": offsets}, {"dst": dst})


DOC_OUTPUT_CASES = {
    "gathermask": _doc_gathermask,
    "gatherb": _doc_gatherb,
}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    test_gathermask_p1()
    test_gathermask_p2()
    test_gathermask_p7()
    test_gatherb_identity()
    logging.info("\nAll batch-6 gathermask/gatherb examples passed!")
