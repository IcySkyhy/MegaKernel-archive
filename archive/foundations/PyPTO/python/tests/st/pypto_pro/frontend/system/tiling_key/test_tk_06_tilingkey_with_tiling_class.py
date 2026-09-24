# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""TilingKey + Tiling Class 结合测试。

验证 kernel 同时有 ``tiling_key`` 和 ``tiling`` 参数时的正确性。
调用语法：``kernel[stream, blocks, {key_dict}](tensors, tiling_instance)``
"""

from __future__ import annotations

from dataclasses import dataclass
import os

import pypto_pro.language as pl
from pypto_pro.runtime.tilingkey import TilingKeyField
import pytest
import torch

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"


# ---- TilingKey 定义 -----------------------------------------------------
class TkOpSelector:
    OpType = TilingKeyField(bits=2, values=[0, 1, 2])


# ---- Tiling 定义 ---------------------------------------------------------
@dataclass
class DimTiling:
    placeholder_before: int[60]
    scalar_field: int
    opkind: int[8]
    placeholder_after: int


# ---- kernel: tiling_key 选运算 + tiling 提供数据 ------------------------
@pl.jit(auto_mutex=True, tiling_key=TkOpSelector)
def kernel_key_op_tiling_data(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    tiling: DimTiling,
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

                if OpType == 0:  # noqa: F821
                    pl.add(tile_c, tile_a, tile_b)
                elif OpType == 1:  # noqa: F821
                    pl.sub(tile_c, tile_a, tile_b)
                else:
                    if tiling.scalar_field == 0:
                        pl.mul(tile_c, tile_a, tile_b)
                    else:
                        pl.add(tile_c, tile_a, tile_b)

                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                pl.store(z, tile_c, [i, j])


# ---- 辅助函数 ------------------------------------------------------------
def _make_tiling(opkind_val):
    placeholder = [0] * 60
    for i in range(60):
        placeholder[i] = i
    opkind_arr = [0] * 8
    opkind_arr[4] = opkind_val
    return DimTiling(
        placeholder_before=placeholder,
        scalar_field=0,
        opkind=opkind_arr,
        placeholder_after=0,
    )


# ---- 简单用例 -----------------------------------------------------------
@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_key_change_tiling_fixed():
    """key 变化 + tiling 固定"""
    device = ST_DEVICE
    torch.npu.set_device(device)
    shape = (128, 256)
    dtype = torch.float16
    tiling = _make_tiling(0)

    for op_val, ref_fn in [(0, lambda a, b:a + b), (1, lambda a, b:a - b), (2, lambda a, b:a * b)]:
        x = torch.rand(shape, device=device, dtype=dtype)
        y = torch.rand(shape, device=device, dtype=dtype)
        z = torch.empty(shape, device=device, dtype=dtype)
        kernel_key_op_tiling_data[None, 1, {"OpType": op_val}](x, y, z, tiling)
        torch.npu.synchronize()
        z_ref = ref_fn(x.float(), y.float()).half()
        torch.testing.assert_close(z, z_ref, atol=1e-2, rtol=1e-2)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_key_fixed_tiling_change():
    """key 固定 + tiling 变化"""
    device = ST_DEVICE
    torch.npu.set_device(device)
    shape = (128, 256)
    dtype = torch.float16

    tiling_add = _make_tiling(0)
    tiling_mul = _make_tiling(2)

    x = torch.rand(shape, device=device, dtype=dtype)
    y = torch.rand(shape, device=device, dtype=dtype)

    z1 = torch.empty(shape, device=device, dtype=dtype)
    kernel_key_op_tiling_data[None, 1, {"OpType": 0}](x, y, z1, tiling_add)
    torch.npu.synchronize()
    z_ref_add = (x.float() + y.float()).half()
    torch.testing.assert_close(z1, z_ref_add, atol=1e-2, rtol=1e-2)

    z2 = torch.empty(shape, device=device, dtype=dtype)
    kernel_key_op_tiling_data[None, 1, {"OpType": 2}](x, y, z2, tiling_mul)
    torch.npu.synchronize()
    z_ref_mul = (x.float() * y.float()).half()
    torch.testing.assert_close(z2, z_ref_mul, atol=1e-2, rtol=1e-2)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_both_change():
    """key 和 tiling 都变化"""
    device = ST_DEVICE
    torch.npu.set_device(device)
    shape = (128, 256)
    dtype = torch.float16

    combos = [
        (0, 0, lambda a, b:a + b),
        (1, 1, lambda a, b:a - b),
        (2, 2, lambda a, b:a * b),
    ]

    for op_val, tiling_val, ref_fn in combos:
        x = torch.rand(shape, device=device, dtype=dtype)
        y = torch.rand(shape, device=device, dtype=dtype)
        z = torch.empty(shape, device=device, dtype=dtype)
        tiling = _make_tiling(tiling_val)
        kernel_key_op_tiling_data[None, 1, {"OpType": op_val}](x, y, z, tiling)
        torch.npu.synchronize()
        z_ref = ref_fn(x.float(), y.float()).half()
        torch.testing.assert_close(z, z_ref, atol=1e-2, rtol=1e-2)


# ---- 复杂用例 -----------------------------------------------------------
@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_multi_block_key_tiling():
    """多核 blocks=4 + key + tiling 组合"""
    device = ST_DEVICE
    torch.npu.set_device(device)
    shape = (128, 256)
    dtype = torch.float16
    tiling = _make_tiling(0)

    x = torch.rand(shape, device=device, dtype=dtype)
    y = torch.rand(shape, device=device, dtype=dtype)
    z = torch.empty(shape, device=device, dtype=dtype)

    kernel_key_op_tiling_data[None, 4, {"OpType": 0}](x, y, z, tiling)
    torch.npu.synchronize()
    z_ref = (x.float() + y.float()).half()
    torch.testing.assert_close(z, z_ref, atol=1e-2, rtol=1e-2)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_key_op_select_tiling_data_2():
    """不同 key 值穿越所有合法组合"""
    device = ST_DEVICE
    torch.npu.set_device(device)
    shape = (128, 256)
    dtype = torch.float16

    tiling_values = [0, 1, 2]
    ref_fns = [lambda a, b:a + b, lambda a, b:a - b, lambda a, b:a * b]

    for tiling_val, ref_fn in zip(tiling_values, ref_fns):
        x = torch.rand(shape, device=device, dtype=dtype)
        y = torch.rand(shape, device=device, dtype=dtype)
        z = torch.empty(shape, device=device, dtype=dtype)
        tiling = _make_tiling(tiling_val)
        kernel_key_op_tiling_data[None, 1, {"OpType": tiling_val}](x, y, z, tiling)
        torch.npu.synchronize()
        z_ref = ref_fn(x.float(), y.float()).half()
        torch.testing.assert_close(z, z_ref, atol=1e-2, rtol=1e-2)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_larger_tensor_key_tiling():
    """更大 tensor 尺寸 key + tiling 组合"""
    device = ST_DEVICE
    torch.npu.set_device(device)
    shape = (256, 512)
    dtype = torch.float16
    tiling = _make_tiling(2)

    x = torch.rand(shape, device=device, dtype=dtype)
    y = torch.rand(shape, device=device, dtype=dtype)
    z = torch.empty(shape, device=device, dtype=dtype)

    kernel_key_op_tiling_data[None, 1, {"OpType": 2}](x, y, z, tiling)
    torch.npu.synchronize()
    z_ref = (x.float() * y.float()).half()
    torch.testing.assert_close(z, z_ref, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
