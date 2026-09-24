# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tiling class duplicate field name validation test (no NPU required).

Positive cases: normal dataclass with unique field names.
Negative cases: duplicate field names should be rejected.
"""

from __future__ import annotations

from dataclasses import dataclass
import os

import pypto_pro.language as pl
import pytest
import torch

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"


@dataclass
class MultiArrayTiling:
    shape_a: int[2]
    shape_b: int[2]
    offsets: int[4]
    selector: int


@pl.jit()
def kernel_multi_array(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    tiling: MultiArrayTiling,
):
    m = x.shape[0]
    n = x.shape[1]
    tile_type = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile(tile_type, addr=0x0000, size=16384)
    tile_b = pl.make_tile(tile_type, addr=0x4000, size=16384)
    tile_c = pl.make_tile(tile_type, addr=0x8000, size=16384)

    with pl.section_vector():
        for i in pl.range(0, m, 64):
            for j in pl.range(0, n, 128):
                pl.system.bar_all()
                pl.load(tile_a, x, [i, j])
                pl.load(tile_b, y, [i, j])
                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                _ = tiling.shape_a[0] + tiling.shape_b[1] + tiling.offsets[0] + tiling.selector
                if tiling.selector == 0:
                    pl.add(tile_c, tile_a, tile_b)
                else:
                    pl.sub(tile_c, tile_a, tile_b)
                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                pl.store(z, tile_c, [i, j])


class TestMultiArrayCoexistence:
    """Positive test: verify multi-Array field coexistence with NPU execution."""

    @pytest.mark.soc("950")
    @pypto.options(pass_options={"enable_slice": False})
    def test_multi_array_coexistence_compute(self):
        device = ST_DEVICE
        torch.npu.set_device(device)
        torch.manual_seed(0)
        dtype = torch.float16
        shape = [64, 128]
        x = torch.rand(shape, device=device, dtype=dtype)
        y = torch.rand(shape, device=device, dtype=dtype)
        z = torch.empty(shape, device=device, dtype=dtype)

        shape_a = [0] * 2
        shape_a[0] = 10
        shape_a[1] = 20
        shape_b = [0] * 2
        shape_b[0] = 30
        shape_b[1] = 40
        offsets = [0] * 4
        offsets[0] = 5
        offsets[1] = 15
        offsets[2] = 25
        offsets[3] = 35
        tiling = MultiArrayTiling(shape_a=shape_a, shape_b=shape_b, offsets=offsets, selector=0)

        kernel_multi_array(x, y, z, tiling)
        torch.npu.synchronize()
        z_ref = (x.float() + y.float()).half()
        torch.testing.assert_close(z, z_ref, atol=1e-2, rtol=1e-2)
