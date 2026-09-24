# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 system test for scalar, void, and nested SIMT callees."""

import os

import pypto_pro.language as pl
import pytest
import torch

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"
THREADS = 32
TILE_BYTES = THREADS * 4


def _require_a5():
    try:
        torch.npu.set_device(ST_DEVICE)
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
    name = torch.npu.get_device_name()
    if "Ascend950" not in name:
        pytest.skip(f"Current device is {name}, not A5 (Ascend950). Skip.")


@pl.simt.function
def affine(value: pl.DT_FP32, scale: pl.DT_FP32, delta: pl.DT_FP32) -> pl.DT_FP32:
    return value * scale + delta


@pl.simt.function
def store_value(
    dst,
    index: pl.DT_UINT32,
    value: pl.DT_FP32,
):
    dst[0, index] = value


@pl.simt.function
def transform_one(
    dst,
    src,
    index: pl.DT_UINT32,
    scale: pl.DT_FP32,
    delta: pl.DT_FP32,
):
    value = affine(src[0, index], scale, delta)
    store_value(dst, index, value)


@pl.simt.function(max_threads=THREADS)
def transform_tile(
    dst,
    src,
    scale: pl.DT_FP32,
    delta: pl.DT_FP32,
):
    tid = pl.simt.linear_thread_idx()
    transform_one(dst, src, tid, scale, delta)


@pl.jit(arch="a5")
def simt_callee_kernel(
    x: pl.Tensor[[1, THREADS], pl.DT_FP32],
    out: pl.Tensor[[1, THREADS], pl.DT_FP32],
    scale: pl.DT_FP32,
    delta: pl.DT_FP32,
):
    tile_type = pl.TileType(shape=[1, THREADS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    src = pl.make_tile(tile_type, addr=0x0000, size=TILE_BYTES)
    dst = pl.make_tile(tile_type, addr=0x0080, size=TILE_BYTES)
    with pl.section_vector():
        pl.load(src, x, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.simt.launch(transform_tile, threads=THREADS, args=(dst, src, scale, delta))
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, dst, [0, 0])


@pytest.mark.soc("950")
def test_simt_callee():
    _require_a5()

    scale = 1.5
    delta = -2.0
    x = torch.arange(THREADS, dtype=torch.float32).reshape(1, THREADS).to(ST_DEVICE)
    out = torch.empty_like(x)

    simt_callee_kernel(x, out, scale, delta)
    torch.npu.synchronize()

    torch.testing.assert_close(out.cpu(), x.cpu() * scale + delta, rtol=0, atol=0)


if __name__ == "__main__":
    test_simt_callee()
