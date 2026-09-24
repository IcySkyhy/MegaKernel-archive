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

"""Tiling op test (A5 CCE): verifies struct-tiling with array fields end-to-end on NPU.

The tiling class mixes scalar and int[N] fields. ``opkind`` is an int[8]
whose real operation selector lives at ``opkind[4]``; the kernel reads ``tiling.opkind[4]``
to pick add/sub/mul. This exercises in-kernel array-element access of a tiling struct.

NOTE: requires physical NPU hardware (npu:1) and the Ascend toolkit; it compiles and
executes a kernel on real hardware.
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
class OpTiling:
    placeholder_before_1: int[60]  # array padding field before opkind
    placeholder_before_2: int  # scalar padding field before opkind
    opkind: int[8]  # operation selector array; real value at opkind[4]
    placeholder_after: int  # scalar padding field after opkind


@pl.jit()
def tiling_op_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    tiling: OpTiling,
):
    tile_type = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile(tile_type, addr=0x0000, size=16384)
    tile_b = pl.make_tile(
        pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addr=0x4000,
        size=16384,
    )
    tile_c = pl.make_tile(
        pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addr=0x8000,
        size=16384,
    )

    with pl.section_vector():
        m_dim = x.shape[0]
        n_dim = x.shape[1]
        for i in pl.range(0, m_dim, 64):
            for j in pl.range(0, n_dim, 128):
                pl.system.bar_all()
                pl.load(tile_a, x, [i, j])
                pl.load(tile_b, y, [i, j])
                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)

                if tiling.opkind[4] == 0:
                    pl.add(tile_c, tile_a, tile_b)
                elif tiling.opkind[4] == 1:
                    pl.sub(tile_c, tile_a, tile_b)
                else:
                    pl.mul(tile_c, tile_a, tile_b)

                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                pl.store(z, tile_c, [i, j])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_tiling_op():
    device = ST_DEVICE
    torch.npu.set_device(device)
    shapes = [[64, 128], [128, 256]]
    torch.manual_seed(0)
    dtype = torch.float16

    op_cases = [
        (0, lambda a, b:a + b, "add"),
        (1, lambda a, b:a - b, "sub"),
        (2, lambda a, b:a * b, "mul"),
    ]

    for shape in shapes:
        x = torch.rand(shape, device=device, dtype=dtype)
        y = torch.rand(shape, device=device, dtype=dtype)

        for opkind, ref_fn, _op_name in op_cases:
            z = torch.empty(shape, device=device, dtype=dtype)
            placeholder = [0] * 60
            for i in range(60):
                placeholder[i] = i
            opkind_arr = [0] * 8
            opkind_arr[4] = opkind  # real operation selector lives at index 4
            tiling = OpTiling(
                placeholder_before_1=placeholder,
                placeholder_before_2=0,
                opkind=opkind_arr,
                placeholder_after=0,
            )
            tiling_op_kernel(x, y, z, tiling)
            torch.npu.synchronize()
            z_ref = ref_fn(x.float(), y.float()).half()
            torch.testing.assert_close(z, z_ref, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
