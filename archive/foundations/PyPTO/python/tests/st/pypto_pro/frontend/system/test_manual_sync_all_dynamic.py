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
"""A5 all-core ST for dynamic-shape ``pl.system.sync_all``.

The E prefix identifies ``sync_all`` scenarios:
  - E01: HARD+AIV_ONLY, covered by ``test_manual_sync_dynamic.py``
  - E02: HARD+AIC_ONLY, covered here by T11
  - E03: HARD+MIX, covered here by T11

Every MIX call is reached by both the cube and vector sections.  This is
required to avoid waiting for a core type that never reaches the barrier.
"""

import logging
import os

import pypto_pro.language as pl
import pytest
import torch

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

TILE_N = 64

DYNAMIC_SHAPES = [
    pytest.param((1, 64), id="one-tile"),
    pytest.param((3, 96), id="unaligned-tail"),
    pytest.param((2048, 2048), id="stress-2048x2048"),
]


def _require_a5() -> None:
    try:
        torch.npu.set_device(ST_DEVICE)
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        pytest.skip(f"Current device is {device_name}, not A5 (Ascend950)")


# === T11: HARD AIC_ONLY followed by HARD MIX ========================================
# E02/E03


@pl.jit(auto_mutex=False)
def t11_hard_aic_mix(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    tile_type = pl.TileType(
        shape=[1, TILE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]
    )
    tile_x = pl.make_tile(tile_type, addr=0x0000, size=256)
    tile_y = pl.make_tile(tile_type, addr=0x0100, size=256)
    tile_out = pl.make_tile(tile_type, addr=0x0200, size=256)

    with pl.section_cube():
        pl.system.sync_all(core_type=pl.SyncCoreType.AIC_ONLY)
        pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
        pl.system.sync_all(core_type=pl.SyncCoreType.MIX)

    with pl.section_vector():
        pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
        if pl.get_block_idx() == 0:
            for row in pl.range(0, x.shape[0]):
                for col in pl.range(0, x.shape[1], TILE_N):
                    valid_n = x.shape[1] - col
                    if valid_n >= TILE_N:
                        pl.set_validshape(tile_x, [1, TILE_N])
                        pl.set_validshape(tile_y, [1, TILE_N])
                        pl.set_validshape(tile_out, [1, TILE_N])
                    else:
                        pl.set_validshape(tile_x, [1, valid_n])
                        pl.set_validshape(tile_y, [1, valid_n])
                        pl.set_validshape(tile_out, [1, valid_n])
                    pl.load(tile_x, x, [row, col])
                    pl.load(tile_y, y, [row, col])
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                    pl.add(tile_out, tile_x, tile_y)
                    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=2)
                    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                    pl.store(out, tile_out, [row, col])
                    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=2)
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=3)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=3)
        pl.system.sync_all(core_type=pl.SyncCoreType.MIX)


def _make_inputs(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, device=ST_DEVICE, dtype=torch.float32)
    y = torch.randn(shape, device=ST_DEVICE, dtype=torch.float32)
    out = torch.empty(shape, device=ST_DEVICE, dtype=torch.float32)
    return x, y, out


@pytest.mark.soc("950")
@pytest.mark.parametrize("shape", DYNAMIC_SHAPES)
@pytest.mark.parametrize("used_cores", [1, 2], ids=["one-core", "two-cores"])
def test_t11_e02_e03_hard_aic_mix(shape, used_cores):
    """T11: AIC_ONLY and MIX HARD barriers execute with the requested launch count."""
    _require_a5()
    x, y, out = _make_inputs(shape)
    t11_hard_aic_mix[None, used_cores](x, y, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, x + y, rtol=1e-4, atol=1e-4)
    logging.info("T11 HARD sync_all shape=%s used_cores=%d passed", shape, used_cores)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for dynamic_shape in ((1, 64), (3, 96)):
        for core_count in (1, 2):
            test_t11_e02_e03_hard_aic_mix(dynamic_shape, core_count)
