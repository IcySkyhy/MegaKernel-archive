# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Gather and system variable APIs (get_block_idx / get_block_num / get_subblock_idx).

Uses make_tile_group + auto_mutex style.

Doc: docs/zh/pypto_pro/api/SIMD-API/计算API/{Memory矢量计算/离散与聚合, 系统访问变量}

Pure vector mode. Verifies indexed gather and multi-core block variable
access for row offset computation.
"""

import logging
import os

import pypto_pro.language as pl
import pytest
import torch

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

NUM_CORES = 2


def _require_a5(device):
    try:
        torch.npu.set_device(device)
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
    name = torch.npu.get_device_name()
    if "Ascend950" not in name:
        pytest.skip(f"Current device is {name}, not A5 (Ascend950). Skip.")


# ===========================================================================
# gather —— dst_flat[i] = src_flat[idx[i]]，idx 为 INT32 扁平索引，需 tile_tmp
# ===========================================================================
@pl.jit(auto_mutex=True)
def gather_kernel(
    src: pl.Tensor[[64, 128], pl.DT_FP16],
    indices: pl.Tensor[[64, 128], pl.DT_INT32],
    dst: pl.Tensor[[64, 128], pl.DT_FP16],
):
    tt_fp16 = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tt_int32 = pl.TileType(shape=[64, 128], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    tile_src = pl.make_tile_group(type=tt_fp16, addrs=0x0000, mutex_ids=[0])
    tile_idx = pl.make_tile_group(type=tt_int32, addrs=0x4000, mutex_ids=[1])
    tile_tmp = pl.make_tile_group(type=tt_int32, addrs=0xC000, mutex_ids=[2])
    tile_dst = pl.make_tile_group(type=tt_fp16, addrs=0x14000, mutex_ids=[3])
    with pl.section_vector():
        cur_src = tile_src.current()
        cur_idx = tile_idx.current()
        cur_tmp = tile_tmp.current()
        cur_dst = tile_dst.current()
        pl.load(cur_src, src, [0, 0])
        pl.load(cur_idx, indices, [0, 0])
        pl.gather(cur_dst, cur_src, cur_idx, cur_tmp)
        pl.store(dst, cur_dst, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_gather():
    device = ST_DEVICE
    _require_a5(device)
    torch.manual_seed(0)
    shape = [64, 128]
    src = torch.rand(shape, device=device, dtype=torch.float16)
    indices = torch.randint(0, shape[0] * shape[1], shape, device=device, dtype=torch.int32)
    dst = torch.empty(shape, device=device, dtype=torch.float16)
    gather_kernel(src, indices, dst)
    torch.npu.synchronize()
    src_flat = src.flatten()
    idx_flat = indices.flatten()
    dst_ref = src_flat[idx_flat.long()].reshape(shape)
    torch.testing.assert_close(dst, dst_ref)
    logging.info("gather result equal!")


# ===========================================================================
# get_block_idx / get_block_num / get_subblock_idx —— 多核切分
#   2 核各处理 64 行，kernel[None, NUM_CORES](...) 启动。golden = x + y。
# ===========================================================================
@pl.jit(auto_mutex=True)
def multicore_add_kernel(
    x: pl.Tensor[[128, 128], pl.DT_FP16],
    y: pl.Tensor[[128, 128], pl.DT_FP16],
    z: pl.Tensor[[128, 128], pl.DT_FP16],
):
    tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_b = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    tile_c = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[2])
    with pl.section_vector():
        vidx = pl.get_block_idx()  # 当前核号
        _bnum = pl.get_block_num()  # 核总数（此处读出验证可调用）
        offset = vidx * 64  # 第 vidx 核处理第 [vidx*64, +64) 行
        cur_a = tile_a.current()
        cur_b = tile_b.current()
        cur_c = tile_c.current()
        pl.load(cur_a, x, [offset, 0])
        pl.load(cur_b, y, [offset, 0])
        pl.add(cur_c, cur_a, cur_b)
        pl.store(z, cur_c, [offset, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_multicore_block_vars():
    device = ST_DEVICE
    _require_a5(device)
    torch.manual_seed(0)
    shape = [128, 128]
    x = torch.rand(shape, device=device, dtype=torch.float16)
    y = torch.rand(shape, device=device, dtype=torch.float16)
    z = torch.empty(shape, device=device, dtype=torch.float16)
    # [None, NUM_CORES] 指定核数；2 核各处理 64 行
    multicore_add_kernel[None, NUM_CORES](x, y, z)
    torch.npu.synchronize()
    torch.testing.assert_close(z, x + y, rtol=1e-2, atol=1e-2)
    logging.info("get_block_idx/get_block_num result equal!")


# ===========================================================================
# get_subblock_idx —— 子核号（0/1），仅验证可调用
#   纯 vector kernel 中两个子核共享 MTE 管道，不能各自独立 store 到 GM。
#   实际场景中 get_subblock_idx() 用于 insert+cube 模式或条件执行。
#   这里用完整 64x64 tile 做 add，get_subblock_idx() 读出但不影响结果。
# ===========================================================================
@pl.jit(auto_mutex=True)
def subblock_add_kernel(
    x: pl.Tensor[[64, 64], pl.DT_FP32],
    y: pl.Tensor[[64, 64], pl.DT_FP32],
    out: pl.Tensor[[64, 64], pl.DT_FP32],
):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_x = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_y = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    tile_sum = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[2])
    with pl.section_vector():
        _sub_index = pl.get_subblock_idx()
        cur_x = tile_x.current()
        cur_y = tile_y.current()
        cur_sum = tile_sum.current()
        pl.load(cur_x, x, [0, 0])
        pl.load(cur_y, y, [0, 0])
        pl.add(cur_sum, cur_x, cur_y)
        pl.store(out, cur_sum, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_subblock_idx():
    device = ST_DEVICE
    _require_a5(device)
    torch.manual_seed(0)
    x = torch.randn(64, 64, device=device, dtype=torch.float32)
    y = torch.randn(64, 64, device=device, dtype=torch.float32)
    out = torch.zeros(64, 64, device=device, dtype=torch.float32)
    subblock_add_kernel(x, y, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, x + y, rtol=1e-2, atol=1e-2)
    logging.info("get_subblock_idx result equal!")


def _doc_gather(ctx, kernel):
    src = ctx.base_fp16((64, 128), 0.25, 1.0)
    indices = torch.arange(64 * 128, device=ctx.device, dtype=torch.int32).reshape(64, 128).flip(1)
    dst = torch.empty((64, 128), device=ctx.device, dtype=torch.float16)
    kernel(src, indices, dst)
    ctx.synchronize()
    return ctx.snippet("gather", {"src": src, "indices": indices}, {"dst": dst})


DOC_OUTPUT_CASES = {"gather": _doc_gather}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    test_gather()
    test_multicore_block_vars()
    test_subblock_idx()
    logging.info("\nAll batch-5 gather/block-vars examples passed!")
