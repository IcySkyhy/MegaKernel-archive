# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 end-to-end tests for the SIMT launch interface."""

import os

import pypto_pro.language as pl
import pytest
import torch

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"
THREADS_1D = 256
THREADS_X_2D = 16
THREADS_Y_2D = 16
THREADS_2D = THREADS_X_2D * THREADS_Y_2D
THREADS_X_3D = 8
THREADS_Y_3D = 4
THREADS_Z_3D = 8
THREADS_3D = THREADS_X_3D * THREADS_Y_3D * THREADS_Z_3D
MAX_THREADS = 2048
NON_WARP_THREADS = 33
NON_POWER_THREADS_X = 7
NON_POWER_THREADS_Y = 5
NON_POWER_THREADS_Z = 3
NON_POWER_THREADS = NON_POWER_THREADS_X * NON_POWER_THREADS_Y * NON_POWER_THREADS_Z


def _require_a5():
    try:
        torch.npu.set_device(ST_DEVICE)
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
    name = torch.npu.get_device_name()
    if "Ascend950" not in name:
        pytest.skip(f"Current device is {name}, not A5 (Ascend950). Skip.")


@pl.simt.function(max_threads=THREADS_1D)
def add_1d(data, delta: pl.DT_FP32):
    tid = pl.simt.thread_idx().x
    data[0, tid] = data[0, tid] + delta


@pl.simt.function(max_threads=THREADS_2D)
def add_2d(data, delta: pl.DT_FP32):
    thread = pl.simt.thread_idx()
    tid = thread.x + thread.y * THREADS_X_2D
    data[0, tid] = data[0, tid] + delta


@pl.simt.function(max_threads=THREADS_3D)
def add_3d(data, delta: pl.DT_FP32):
    thread = pl.simt.thread_idx()
    tid = thread.x + thread.y * THREADS_X_3D + thread.z * THREADS_X_3D * THREADS_Y_3D
    data[0, tid] = data[0, tid] + delta


@pl.simt.function(max_threads=MAX_THREADS)
def write_linear_thread_id(out: pl.Tensor[[1, MAX_THREADS], pl.DT_UINT32]):
    tid = pl.simt.linear_thread_idx()
    out[0, tid] = tid


@pl.jit(arch="a5")
def simt_1d_launch(
    x: pl.Tensor[[1, THREADS_1D], pl.DT_FP32],
    out: pl.Tensor[[1, THREADS_1D], pl.DT_FP32],
    delta: pl.DT_FP32,
):
    tile_type = pl.TileType(shape=[1, THREADS_1D], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    data = pl.make_tile(tile_type, addr=0x0000, size=THREADS_1D * 4)
    with pl.section_vector():
        pl.load(data, x, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.simt.launch(add_1d, threads=THREADS_1D, args=(data, delta))
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, data, [0, 0])


@pl.jit(arch="a5")
def simt_2d_launch(
    x: pl.Tensor[[1, THREADS_2D], pl.DT_FP32],
    out: pl.Tensor[[1, THREADS_2D], pl.DT_FP32],
    delta: pl.DT_FP32,
):
    tile_type = pl.TileType(shape=[1, THREADS_2D], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    data = pl.make_tile(tile_type, addr=0x0000, size=THREADS_2D * 4)
    with pl.section_vector():
        pl.load(data, x, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.simt.launch(add_2d, threads=(THREADS_X_2D, THREADS_Y_2D), args=(data, delta))
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, data, [0, 0])


@pl.jit(arch="a5")
def simt_3d_launch(
    x: pl.Tensor[[1, THREADS_3D], pl.DT_FP32],
    out: pl.Tensor[[1, THREADS_3D], pl.DT_FP32],
    delta: pl.DT_FP32,
):
    tile_type = pl.TileType(shape=[1, THREADS_3D], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    data = pl.make_tile(tile_type, addr=0x0000, size=THREADS_3D * 4)
    with pl.section_vector():
        pl.load(data, x, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.simt.launch(add_3d, threads=(THREADS_X_3D, THREADS_Y_3D, THREADS_Z_3D), args=(data, delta))
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, data, [0, 0])


@pl.jit(arch="a5")
def simt_launch_non_warp(out: pl.Tensor[[1, MAX_THREADS], pl.DT_UINT32]):
    with pl.section_vector():
        pl.simt.launch(write_linear_thread_id, threads=NON_WARP_THREADS, args=(out,))


@pl.jit(arch="a5")
def simt_launch_non_power_of_two_3d(out: pl.Tensor[[1, MAX_THREADS], pl.DT_UINT32]):
    with pl.section_vector():
        pl.simt.launch(
            write_linear_thread_id,
            threads=(NON_POWER_THREADS_X, NON_POWER_THREADS_Y, NON_POWER_THREADS_Z),
            args=(out,),
        )


@pl.jit(arch="a5")
def simt_launch_hardware_limit(out: pl.Tensor[[1, MAX_THREADS], pl.DT_UINT32]):
    with pl.section_vector():
        pl.simt.launch(write_linear_thread_id, threads=MAX_THREADS, args=(out,))


def _run_add_launch(kernel, threads, delta):
    x = torch.arange(threads, dtype=torch.float32).reshape(1, threads).to(ST_DEVICE)
    out = torch.empty_like(x)
    kernel(x, out, delta)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), x.cpu() + delta, rtol=0, atol=0)


def _run_boundary_launch(kernel, active_threads):
    sentinel = torch.iinfo(torch.uint32).max
    out = torch.full((1, MAX_THREADS), -1, dtype=torch.int32).to(torch.uint32).to(ST_DEVICE)
    kernel(out)
    torch.npu.synchronize()
    expected = torch.full((1, MAX_THREADS), sentinel, dtype=torch.uint32)
    expected[0, :active_threads] = torch.arange(active_threads, dtype=torch.int64).to(torch.uint32)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


@pytest.mark.soc("950")
def test_1d_launch():
    _require_a5()
    _run_add_launch(simt_1d_launch, THREADS_1D, 2.5)


@pytest.mark.soc("950")
def test_2d_launch():
    _require_a5()
    _run_add_launch(simt_2d_launch, THREADS_2D, 1.5)


@pytest.mark.soc("950")
def test_3d_launch():
    _require_a5()
    _run_add_launch(simt_3d_launch, THREADS_3D, 0.75)


@pytest.mark.soc("950")
def test_1d_launch_non_warp_thread_count_preserves_inactive_tail():
    _require_a5()
    _run_boundary_launch(simt_launch_non_warp, NON_WARP_THREADS)


@pytest.mark.soc("950")
def test_3d_launch_non_power_of_two_dimensions_preserves_inactive_tail():
    _require_a5()
    _run_boundary_launch(simt_launch_non_power_of_two_3d, NON_POWER_THREADS)


@pytest.mark.soc("950")
def test_1d_launch_at_hardware_thread_limit():
    _require_a5()
    _run_boundary_launch(simt_launch_hardware_limit, MAX_THREADS)


if __name__ == "__main__":
    test_1d_launch()
    test_2d_launch()
    test_3d_launch()
