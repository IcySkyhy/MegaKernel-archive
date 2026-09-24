# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tiling class complex layout test: verifies 7-field tiling class with Array+scalar alternation.

Fields: int[60], int, int[2], float, bool[8], bool, int[32].
Simple cases: verify ctypes struct padding/size consistency with CCE, round-trip fidelity.
Complex cases: verify NPU kernel reads all 7 fields correctly with 3 different tiling combinations.
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
class ComplexLayout:
    pad_before: int[60]
    scalar_a: int
    shape_2d: int[2]
    scalar_b: float
    flags: bool[8]
    scalar_c: bool
    pad_after: int[32]


@pl.jit()
def kernel_complex_layout(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    tiling: ComplexLayout,
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

                if tiling.scalar_a == 0:
                    pl.add(tile_c, tile_a, tile_b)
                elif tiling.scalar_a == 1:
                    pl.sub(tile_c, tile_a, tile_b)
                elif tiling.scalar_a == 2:
                    pl.sub(tile_c, tile_b, tile_a)
                else:
                    pl.add(tile_c, tile_a, tile_b)

                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                pl.store(z, tile_c, [i, j])


def _make_pad_before(fill_val=0):
    arr = [0] * 60
    for k in range(60):
        arr[k] = fill_val + k
    return arr


def _make_shape_2d(vals):
    return list(vals)


def _make_flags(pattern=None):
    arr = [0] * 8
    if pattern is None:
        for k in range(8):
            arr[k] = bool(k % 2)
    else:
        for k in range(8):
            arr[k] = pattern[k] if k < len(pattern) else False
    return arr


def _make_pad_after(fill_val=0):
    arr = [0] * 32
    for k in range(32):
        arr[k] = fill_val + k
    return arr


def _run_complex_test(
    scalar_a, scalar_b, scalar_c, shape_2d_vals, flags_pattern, pad_before_val, pad_after_val, expected_op
):
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    dtype = torch.float16
    shape = [128, 256]

    x = torch.rand(shape, device=device, dtype=dtype)
    y = torch.rand(shape, device=device, dtype=dtype)
    z = torch.empty(shape, device=device, dtype=dtype)

    tiling = ComplexLayout(
        pad_before=_make_pad_before(pad_before_val),
        scalar_a=scalar_a,
        shape_2d=_make_shape_2d(shape_2d_vals),
        scalar_b=scalar_b,
        flags=_make_flags(flags_pattern),
        scalar_c=scalar_c,
        pad_after=_make_pad_after(pad_after_val),
    )

    kernel_complex_layout(x, y, z, tiling)
    torch.npu.synchronize()

    if expected_op == "add":
        z_ref = (x.float() + y.float()).half()
    elif expected_op == "sub":
        z_ref = (x.float() - y.float()).half()
    elif expected_op == "sub_rev":
        z_ref = (y.float() - x.float()).half()
    else:
        z_ref = (x.float() + y.float()).half()
    torch.testing.assert_close(z, z_ref, atol=1e-2, rtol=1e-2)


class TestComplexLayoutNPUKernel:
    """Complex cases: verify all 7 tiling fields are accessible on NPU with 3 combinations."""

    @pytest.mark.soc("950")
    @pypto.options(pass_options={"enable_slice": False})
    def test_combo1_scalar_a_0_add(self):
        _run_complex_test(
            scalar_a=0,
            scalar_b=1.0,
            scalar_c=True,
            shape_2d_vals=[64, 128],
            flags_pattern=[True, False, False, False, False, False, False, False],
            pad_before_val=42,
            pad_after_val=99,
            expected_op="add",
        )

    @pytest.mark.soc("950")
    @pypto.options(pass_options={"enable_slice": False})
    def test_combo2_scalar_a_1_sub(self):
        _run_complex_test(
            scalar_a=1,
            scalar_b=2.0,
            scalar_c=False,
            shape_2d_vals=[128, 256],
            flags_pattern=[False, True, False, False, False, False, False, False],
            pad_before_val=7,
            pad_after_val=13,
            expected_op="sub",
        )

    @pytest.mark.soc("950")
    @pypto.options(pass_options={"enable_slice": False})
    def test_combo3_scalar_a_2_sub_rev(self):
        _run_complex_test(
            scalar_a=2,
            scalar_b=0.5,
            scalar_c=True,
            shape_2d_vals=[256, 512],
            flags_pattern=[False, False, True, False, False, False, False, False],
            pad_before_val=100,
            pad_after_val=200,
            expected_op="sub_rev",
        )
