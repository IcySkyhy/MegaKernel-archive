# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Quantization, type cast, and scatter ops.

Doc: docs/zh/pypto_pro/api/SIMD-API/计算API/{量化, Memory矢量计算/类型转换, Memory矢量计算/离散与聚合}

make_tile_group + auto_mutex style. Verifies quant/dequant,
pl.cast type conversion, and scatter correctness.
"""

import logging
import os

import pypto_pro.language as pl
import pytest
import torch

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"


def _require_a5(device):
    try:
        torch.npu.set_device(device)
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
    name = torch.npu.get_device_name()
    if "Ascend950" not in name:
        pytest.skip(f"Current device is {name}, not A5 (Ascend950). Skip.")


# ===========================================================================
# quant —— 对称量化 FP32 -> INT8，per-row scale [64,1]
# ===========================================================================
@pl.jit(auto_mutex=True)
def quant_kernel(
    src: pl.Tensor[[64, 128], pl.DT_FP32],
    scale: pl.Tensor[[64, 1], pl.DT_FP32],
    out: pl.Tensor[[64, 128], pl.DT_INT8],
):
    tile_src = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids=[0],
    )
    tile_scale = pl.make_tile_group(
        type=pl.TileType(shape=[64, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=0x8000, mutex_ids=[1]
    )
    tile_out = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_INT8, target_memory=pl.MemorySpace.Vec),
        addrs=0xA000,
        mutex_ids=[2],
    )
    with pl.section_vector():
        cur_src = tile_src.current()
        cur_scale = tile_scale.current()
        cur_out = tile_out.current()
        pl.load(cur_src, src, [0, 0])
        pl.load(cur_scale, scale, [0, 0])
        pl.quant(cur_out, cur_src, cur_scale, mode=pl.QuantMode.SYM)
        pl.store(out, cur_out, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_quant():
    device = ST_DEVICE
    _require_a5(device)
    torch.manual_seed(0)
    src = torch.randn(64, 128, device=device, dtype=torch.float32)
    scale = torch.rand(64, 1, device=device, dtype=torch.float32) + 0.01
    out = torch.zeros(64, 128, device=device, dtype=torch.int8)
    quant_kernel(src, scale, out)
    torch.npu.synchronize()
    out_ref = torch.clamp(torch.round(src * scale), -128, 127).to(torch.int8)
    torch.testing.assert_close(out, out_ref)
    logging.info("quant result equal!")


# ===========================================================================
# dequant —— INT8 -> FP32，per-row scale + offset
# ===========================================================================
@pl.jit(auto_mutex=True)
def dequant_kernel(
    src: pl.Tensor[[64, 128], pl.DT_INT8],
    scale: pl.Tensor[[64, 1], pl.DT_FP32],
    offset: pl.Tensor[[64, 1], pl.DT_FP32],
    out: pl.Tensor[[64, 128], pl.DT_FP32],
):
    tile_src = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_INT8, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids=[0],
    )
    tile_scale = pl.make_tile_group(
        type=pl.TileType(shape=[64, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=0x4000, mutex_ids=[1]
    )
    tile_offset = pl.make_tile_group(
        type=pl.TileType(shape=[64, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=0x5000, mutex_ids=[2]
    )
    tile_out = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x6000,
        mutex_ids=[3],
    )
    with pl.section_vector():
        cur_src = tile_src.current()
        cur_scale = tile_scale.current()
        cur_offset = tile_offset.current()
        cur_out = tile_out.current()
        pl.load(cur_src, src, [0, 0])
        pl.load(cur_scale, scale, [0, 0])
        pl.load(cur_offset, offset, [0, 0])
        pl.dequant(cur_out, cur_src, cur_scale, cur_offset)
        pl.store(out, cur_out, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_dequant():
    device = ST_DEVICE
    _require_a5(device)
    torch.manual_seed(0)
    src = torch.randint(-128, 127, (64, 128), device=device, dtype=torch.int8)
    scale = torch.rand(64, 1, device=device, dtype=torch.float32) + 0.01
    offset = torch.zeros(64, 1, device=device, dtype=torch.float32)
    out = torch.zeros(64, 128, device=device, dtype=torch.float32)
    dequant_kernel(src, scale, offset, out)
    torch.npu.synchronize()
    out_ref = (src.to(torch.float32) - offset) * scale
    torch.testing.assert_close(out, out_ref, rtol=1e-2, atol=1e-2)
    logging.info("dequant result equal!")


# ===========================================================================
# cast —— 类型转换 FP16 -> FP32（目标类型由 out tile 推断）
# ===========================================================================
@pl.jit(auto_mutex=True)
def cast_kernel(
    src: pl.Tensor[[64, 128], pl.DT_FP16],
    out: pl.Tensor[[64, 128], pl.DT_FP32],
):
    tt_in = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tt_out = pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_src = pl.make_tile_group(type=tt_in, addrs=0x0000, mutex_ids=[0])
    tile_out = pl.make_tile_group(type=tt_out, addrs=0x4000, mutex_ids=[1])
    with pl.section_vector():
        cur_src = tile_src.current()
        cur_out = tile_out.current()
        pl.load(cur_src, src, [0, 0])
        pl.cast(cur_out, cur_src, mode=pl.RoundMode.CAST_ROUND)
        pl.store(out, cur_out, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_cast():
    device = ST_DEVICE
    _require_a5(device)
    torch.manual_seed(0)
    src = torch.randn(64, 128, device=device, dtype=torch.float16)
    out = torch.zeros(64, 128, device=device, dtype=torch.float32)
    cast_kernel(src, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, src.float(), rtol=1e-2, atol=1e-2)
    logging.info("cast result equal!")


# ===========================================================================
# scatter —— dst_flat[indices[i,j]] = src[i,j]，index 为目的的扁平元素偏移
# ===========================================================================
@pl.jit(auto_mutex=True)
def scatter_kernel(
    src: pl.Tensor[[64, 128], pl.DT_FP32],
    indices: pl.Tensor[[64, 128], pl.DT_INT32],
    dst: pl.Tensor[[64, 128], pl.DT_FP32],
):
    tile_src = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids=[0],
    )
    tile_idx = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x8000,
        mutex_ids=[1],
    )
    tile_dst = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x10000,
        mutex_ids=[2],
    )
    with pl.section_vector():
        cur_src = tile_src.current()
        cur_idx = tile_idx.current()
        cur_dst = tile_dst.current()
        pl.load(cur_src, src, [0, 0])
        pl.load(cur_idx, indices, [0, 0])
        pl.scatter(cur_dst, cur_src, cur_idx)
        pl.store(dst, cur_dst, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_scatter():
    device = ST_DEVICE
    _require_a5(device)
    torch.manual_seed(0)
    shape = [64, 128]
    src = torch.rand(shape, device=device, dtype=torch.float32)
    # 用排列，保证每个 dst 位置只被写一次（无写冲突）
    num_elements = shape[0] * shape[1]
    indices = torch.randperm(num_elements, device=device).to(torch.int32).reshape(shape)
    dst = torch.zeros(shape, device=device, dtype=torch.float32)
    scatter_kernel(src, indices, dst)
    torch.npu.synchronize()
    dst_ref = torch.zeros(num_elements, device="cpu", dtype=torch.float32)
    src_cpu, idx_cpu = src.cpu(), indices.cpu()
    for i in range(shape[0]):
        for j in range(shape[1]):
            dst_ref[idx_cpu[i, j].item()] = src_cpu[i, j]
    dst_ref = dst_ref.reshape(shape).to(device)
    torch.testing.assert_close(dst, dst_ref, rtol=1e-2, atol=1e-2)
    logging.info("scatter result equal!")


def _doc_quant(ctx, kernel):
    src = ctx.base_fp32((64, 128), 0.25, -16.0)
    scale = torch.full((64, 1), 4.0, device=ctx.device, dtype=torch.float32)
    out = torch.zeros((64, 128), device=ctx.device, dtype=torch.int8)
    kernel(src, scale, out)
    ctx.synchronize()
    return ctx.snippet("quant", {"src": src, "scale": scale}, {"out": out})


def _doc_dequant(ctx, kernel):
    src = (torch.arange(64 * 128, device=ctx.device, dtype=torch.int32).reshape(64, 128) % 64 - 32).to(torch.int8)
    scale = torch.full((64, 1), 0.25, device=ctx.device, dtype=torch.float32)
    offset = torch.zeros((64, 1), device=ctx.device, dtype=torch.float32)
    out = torch.zeros((64, 128), device=ctx.device, dtype=torch.float32)
    kernel(src, scale, offset, out)
    ctx.synchronize()
    return ctx.snippet("dequant", {"src": src, "scale": scale, "offset": offset}, {"out": out})


def _doc_cast(ctx, kernel):
    src = ctx.base_fp16((64, 128), 0.25, -4.0)
    out = torch.zeros((64, 128), device=ctx.device, dtype=torch.float32)
    kernel(src, out)
    ctx.synchronize()
    return ctx.snippet("cast", {"src": src}, {"out": out})


def _doc_scatter(ctx, kernel):
    src = ctx.base_fp32((64, 128), 0.25, 1.0)
    numel = 64 * 128
    indices = torch.arange(numel - 1, -1, -1, device=ctx.device, dtype=torch.int32).reshape(64, 128)
    dst = torch.zeros((64, 128), device=ctx.device, dtype=torch.float32)
    kernel(src, indices, dst)
    ctx.synchronize()
    return ctx.snippet("scatter", {"src": src, "indices": indices}, {"dst": dst})


DOC_OUTPUT_CASES = {
    "quant": _doc_quant,
    "dequant": _doc_dequant,
    "cast": _doc_cast,
    "scatter": _doc_scatter,
}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for fn in [test_quant, test_dequant, test_cast, test_scatter]:
        fn()
    logging.info("\nAll batch-3 quant/cast/scatter examples passed!")
