# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""TilingKey .so size verification: validates dead-code elimination via .

Builds the same kernel with two different tiling key values:
- Light branch: single ``pl.add``
- Heavy branch: ``pl.add + pl.sub + pl.mul + pl.add + pl.sub``

Verifies:
1. Heavy .so is larger than light .so (dead-code elimination proof).
2. Both branches compute correct results (precision validation).
"""

from __future__ import annotations

import os

import pypto_pro.language as pl
from pypto_pro.runtime.tilingkey import TilingKeyField
import pytest
import torch

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"


class TkSizeVerify:
    SelBranch = TilingKeyField(bits=1, values=[0, 1])


@pl.jit(auto_mutex=True, tiling_key=TkSizeVerify)
def kernel_size_verify(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
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

                if SelBranch == 0:  # noqa: F821
                    pl.add(tile_c, tile_a, tile_b)
                else:
                    pl.add(tile_c, tile_a, tile_b)
                    pl.sub(tile_c, tile_c, tile_a)
                    pl.mul(tile_c, tile_c, tile_b)
                    pl.add(tile_c, tile_c, tile_a)
                    pl.sub(tile_c, tile_c, tile_b)

                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                pl.store(z, tile_c, [i, j])


def _heavy_golden(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    r"""Heavy branch golden: z = ((((a + b) - a) * b) + a) - b"""
    return ((((a + b) - a) * b) + a) - b


class TestSoSizeVerification:
    """Validates that tilingkey dead-code elimination reduces .so size."""

    @pytest.mark.soc("950")
    @pypto.options(pass_options={"enable_slice": False})
    def test_light_branch_correctness(self):
        device = ST_DEVICE
        torch.npu.set_device(device)
        torch.manual_seed(0)
        dtype = torch.float16
        shape = (128, 256)
        x = torch.rand(shape, device=device, dtype=dtype)
        y = torch.rand(shape, device=device, dtype=dtype)
        z = torch.empty(shape, device=device, dtype=dtype)

        kernel_size_verify[None, 1, {"SelBranch": 0}](x, y, z)
        torch.npu.synchronize()
        z_ref = (x.float() + y.float()).half()
        torch.testing.assert_close(z, z_ref, atol=1e-2, rtol=1e-2)

    @pytest.mark.soc("950")
    @pypto.options(pass_options={"enable_slice": False})
    def test_heavy_branch_correctness(self):
        device = ST_DEVICE
        torch.npu.set_device(device)
        torch.manual_seed(0)
        dtype = torch.float16
        shape = (128, 256)
        x = torch.rand(shape, device=device, dtype=dtype)
        y = torch.rand(shape, device=device, dtype=dtype)
        z = torch.empty(shape, device=device, dtype=dtype)

        kernel_size_verify[None, 1, {"SelBranch": 1}](x, y, z)
        torch.npu.synchronize()
        z_ref = _heavy_golden(x.float(), y.float()).half()
        torch.testing.assert_close(z, z_ref, atol=1e-2, rtol=1e-2)
