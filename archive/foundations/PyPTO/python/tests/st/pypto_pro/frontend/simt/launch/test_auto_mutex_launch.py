# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 ST: simt.launch with auto_mutex — no manual sync_src/sync_dst needed.

Uses make_tile_group(mutex_ids=[0]) so auto_mutex inserts mutex_lock(PIPE_V) /
mutex_unlock(PIPE_V) around load, simt.launch, and store automatically.
"""

import os

import pypto_pro.language as pl
import pytest
import torch

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"
THREADS = 256
TILE_BYTES = THREADS * 4


def _require_a5():
    try:
        torch.npu.set_device(ST_DEVICE)
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
    name = torch.npu.get_device_name()
    if "Ascend950" not in name:
        pytest.skip(f"Current device is {name}, not A5 (Ascend950). Skip.")


@pl.simt.function(max_threads=THREADS)
def auto_mutex_add(data, delta: pl.DT_FP32):
    tid = pl.simt.linear_thread_idx()
    data[0, tid] = data[0, tid] + delta


@pl.jit(arch="a5")
def simt_auto_mutex_kernel(
    x: pl.Tensor[[1, THREADS], pl.DT_FP32],
    out: pl.Tensor[[1, THREADS], pl.DT_FP32],
    delta: pl.DT_FP32,
):
    tile_type = pl.TileType(shape=[1, THREADS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    data = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0])
    with pl.section_vector():
        pl.load(data.current(), x, [0, 0])
        pl.simt.launch(auto_mutex_add, threads=THREADS, args=(data.current(), delta))
        pl.store(out, data.current(), [0, 0])


@pytest.mark.soc("950")
def test_auto_mutex_launch():
    _require_a5()

    delta = 3.5
    x = torch.arange(THREADS, dtype=torch.float32).reshape(1, THREADS).to(ST_DEVICE)
    out = torch.empty_like(x)

    simt_auto_mutex_kernel(x, out, delta)
    torch.npu.synchronize()

    torch.testing.assert_close(out.cpu(), x.cpu() + delta, rtol=0, atol=0)


if __name__ == "__main__":
    test_auto_mutex_launch()
