# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 end-to-end test for launching a SIMT function on multiple AIV cores."""

import os

import pypto_pro.language as pl
import pytest
import torch

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"
GRID_BLOCKS = 4
THREADS = 64
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
def write_multicore_result(dst):
    tid = pl.simt.linear_thread_idx()
    core_id = pl.simt.block_idx().x
    dst[0, tid] = core_id * THREADS + tid


@pl.jit(arch="a5")
def simt_multicore(out: pl.Tensor[[GRID_BLOCKS, THREADS], pl.DT_UINT32]):
    tile_type = pl.TileType(shape=[1, THREADS], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec)
    dst = pl.make_tile(tile_type, addr=0x0000, size=TILE_BYTES)
    with pl.section_vector():
        core_id = pl.get_block_idx()
        pl.simt.launch(write_multicore_result, threads=THREADS, args=(dst,))
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
        pl.store(out, dst, [core_id, 0])


@pytest.mark.soc("950")
def test_multicore():
    _require_a5()

    out = torch.empty((GRID_BLOCKS, THREADS), dtype=torch.int32).to(torch.uint32).to(ST_DEVICE)
    simt_multicore[None, GRID_BLOCKS](out)
    torch.npu.synchronize()

    expected = torch.arange(GRID_BLOCKS * THREADS, dtype=torch.int64).reshape(GRID_BLOCKS, THREADS)
    torch.testing.assert_close(out.cpu().to(torch.int64), expected, rtol=0, atol=0)


if __name__ == "__main__":
    test_multicore()
