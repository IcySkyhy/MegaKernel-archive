# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tiling class scalar-only test: verifies tiling with int, float, and bool scalar fields only.

Simple cases: single scalar field in tiling class.
Complex cases: mixed scalar fields, including scenarios where pl.constexpr cannot be used.
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
class TilingIntOnly:
    selector: int


@dataclass
class TilingFloatOnly:
    scale: float


@dataclass
class TilingBoolOnly:
    enabled: bool


@dataclass
class TilingIntFloat:
    opkind: int
    multiplier: float


@dataclass
class TilingAllScalars:
    opkind: int
    scale: float
    enabled: bool


@dataclass
class TilingFloatMulti:
    scale_a: float
    scale_b: float
    bias: float


@pl.jit()
def kernel_int_only(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    tiling: TilingIntOnly,
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
                if tiling.selector == 0:
                    pl.add(tile_c, tile_a, tile_b)
                else:
                    pl.sub(tile_c, tile_a, tile_b)
                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                pl.store(z, tile_c, [i, j])


@pl.jit()
def kernel_float_only(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    tiling: TilingFloatOnly,
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
                if tiling.scale == tiling.scale:
                    pl.add(tile_c, tile_a, tile_b)
                else:
                    pl.sub(tile_c, tile_a, tile_b)
                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                pl.store(z, tile_c, [i, j])


@pl.jit()
def kernel_bool_only(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    tiling: TilingBoolOnly,
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
                if tiling.enabled:
                    pl.add(tile_c, tile_a, tile_b)
                else:
                    pl.sub(tile_c, tile_a, tile_b)
                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                pl.store(z, tile_c, [i, j])


@pl.jit()
def kernel_int_float(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    tiling: TilingIntFloat,
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
                if tiling.opkind == 0:
                    pl.add(tile_c, tile_a, tile_b)
                else:
                    pl.sub(tile_c, tile_a, tile_b)
                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                pl.store(z, tile_c, [i, j])


@pl.jit()
def kernel_all_scalars(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    tiling: TilingAllScalars,
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
                if tiling.opkind == 0:
                    pl.add(tile_c, tile_a, tile_b)
                elif tiling.opkind == 1:
                    pl.sub(tile_c, tile_a, tile_b)
                elif tiling.opkind == 2:
                    if tiling.enabled:
                        pl.mul(tile_c, tile_a, tile_b)
                    else:
                        pl.add(tile_c, tile_a, tile_b)
                else:
                    pl.add(tile_c, tile_a, tile_b)
                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                pl.store(z, tile_c, [i, j])


@pl.jit()
def kernel_float_multi_user(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    tiling: TilingFloatMulti,
):
    """Kernel that uses multiple float fields — a scenario pl.constexpr cannot replace."""
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
                if tiling.scale_a == tiling.scale_b:
                    pl.add(tile_c, tile_a, tile_b)
                else:
                    pl.sub(tile_c, tile_a, tile_b)
                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                pl.store(z, tile_c, [i, j])


class TestScalarOnlySimple:
    """Simple cases: single scalar field in tiling class."""

    @pytest.mark.soc("950")
    @pypto.options(pass_options={"enable_slice": False})
    def test_int_only(self):
        device = ST_DEVICE
        torch.npu.set_device(device)
        torch.manual_seed(0)
        dtype = torch.float16
        shape = [128, 256]
        x = torch.rand(shape, device=device, dtype=dtype)
        y = torch.rand(shape, device=device, dtype=dtype)
        z = torch.empty(shape, device=device, dtype=dtype)
        tiling = TilingIntOnly(selector=0)
        kernel_int_only(x, y, z, tiling)
        torch.npu.synchronize()
        z_ref = (x.float() + y.float()).half()
        torch.testing.assert_close(z, z_ref, atol=1e-2, rtol=1e-2)

    @pytest.mark.soc("950")
    @pypto.options(pass_options={"enable_slice": False})
    def test_float_only(self):
        device = ST_DEVICE
        torch.npu.set_device(device)
        torch.manual_seed(0)
        dtype = torch.float16
        shape = [128, 256]
        x = torch.rand(shape, device=device, dtype=dtype)
        y = torch.rand(shape, device=device, dtype=dtype)
        z = torch.empty(shape, device=device, dtype=dtype)
        tiling = TilingFloatOnly(scale=0.0)
        kernel_float_only(x, y, z, tiling)
        torch.npu.synchronize()
        z_ref = (x.float() + y.float()).half()
        torch.testing.assert_close(z, z_ref, atol=1e-2, rtol=1e-2)

    @pytest.mark.soc("950")
    @pypto.options(pass_options={"enable_slice": False})
    def test_bool_only(self):
        device = ST_DEVICE
        torch.npu.set_device(device)
        torch.manual_seed(0)
        dtype = torch.float16
        shape = [128, 256]
        x = torch.rand(shape, device=device, dtype=dtype)
        y = torch.rand(shape, device=device, dtype=dtype)
        z = torch.empty(shape, device=device, dtype=dtype)
        tiling = TilingBoolOnly(enabled=True)
        kernel_bool_only(x, y, z, tiling)
        torch.npu.synchronize()
        z_ref = (x.float() + y.float()).half()
        torch.testing.assert_close(z, z_ref, atol=1e-2, rtol=1e-2)


class TestScalarOnlyComplex:
    """Complex cases: mixed scalar fields or field combinations."""

    @pytest.mark.soc("950")
    @pypto.options(pass_options={"enable_slice": False})
    def test_int_and_float(self):
        device = ST_DEVICE
        torch.npu.set_device(device)
        torch.manual_seed(0)
        dtype = torch.float16
        shape = [128, 256]
        x = torch.rand(shape, device=device, dtype=dtype)
        y = torch.rand(shape, device=device, dtype=dtype)
        z = torch.empty(shape, device=device, dtype=dtype)
        tiling = TilingIntFloat(opkind=0, multiplier=1.0)
        kernel_int_float(x, y, z, tiling)
        torch.npu.synchronize()
        z_ref = (x.float() + y.float()).half()
        torch.testing.assert_close(z, z_ref, atol=1e-2, rtol=1e-2)

    @pytest.mark.soc("950")
    @pypto.options(pass_options={"enable_slice": False})
    def test_all_scalars_mixed(self):
        device = ST_DEVICE
        torch.npu.set_device(device)
        torch.manual_seed(0)
        dtype = torch.float16
        shape = [128, 256]
        x = torch.rand(shape, device=device, dtype=dtype)
        y = torch.rand(shape, device=device, dtype=dtype)
        z = torch.empty(shape, device=device, dtype=dtype)
        tiling = TilingAllScalars(opkind=1, scale=1.0, enabled=False)
        kernel_all_scalars(x, y, z, tiling)
        torch.npu.synchronize()
        z_ref = (x.float() - y.float()).half()
        torch.testing.assert_close(z, z_ref, atol=1e-2, rtol=1e-2)

    @pytest.mark.soc("950")
    @pypto.options(pass_options={"enable_slice": False})
    def test_float_multi_user(self):
        device = ST_DEVICE
        torch.npu.set_device(device)
        torch.manual_seed(0)
        dtype = torch.float16
        shape = [128, 256]
        x = torch.rand(shape, device=device, dtype=dtype)
        y = torch.rand(shape, device=device, dtype=dtype)
        z = torch.empty(shape, device=device, dtype=dtype)
        tiling = TilingFloatMulti(scale_a=2.0, scale_b=2.0, bias=0.0)
        kernel_float_multi_user(x, y, z, tiling)
        torch.npu.synchronize()
        z_ref = (x.float() + y.float()).half()
        torch.testing.assert_close(z, z_ref, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
