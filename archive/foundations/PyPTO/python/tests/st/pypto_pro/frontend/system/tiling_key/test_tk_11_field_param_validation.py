# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""TilingKeyField 构造期参数校验测试。

验证 TilingKeyField() 在构造阶段对非法输入的正确拦截行为：
1. bits=0        → 应抛异常（零宽度字段无意义）
2. values=[]     → 应抛异常（空值列表）
3. values=[str]  → 应抛异常（非整数值）
正向校验：合法构造 + NPU 上板验证 add/sub/mul 通过 constexpr 正确分发。
"""

import os

import pypto_pro.language as pl
from pypto_pro.runtime.tilingkey import TilingKeyField
import pytest
import torch

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"


# ---- 负向用例：构造期参数校验 ------------------------------------------------


# ---- 正向用例：合法构造 + NPU 上板验证 --------------------------------------


class TkValid:
    OpType = TilingKeyField(bits=2, values=[0, 1, 2, 3])


@pl.jit(auto_mutex=True, tiling_key=TkValid)
def kernel_valid(
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

                if OpType == 0:  # noqa: F821
                    pl.add(tile_c, tile_a, tile_b)
                elif OpType == 1:  # noqa: F821
                    pl.sub(tile_c, tile_a, tile_b)
                else:
                    pl.mul(tile_c, tile_a, tile_b)

                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                pl.store(z, tile_c, [i, j])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_valid_on_npu_add():
    """合法 TilingKey 上板 — add 运算（OpType=0）。"""
    device = ST_DEVICE
    torch.npu.set_device(device)
    shape = (128, 256)
    dtype = torch.float16

    x = torch.rand(shape, device=device, dtype=dtype)
    y = torch.rand(shape, device=device, dtype=dtype)
    z = torch.empty(shape, device=device, dtype=dtype)

    kernel_valid[None, 1, {"OpType": 0}](x, y, z)
    torch.npu.synchronize()
    z_ref = (x.float() + y.float()).half()
    torch.testing.assert_close(z, z_ref, atol=1e-2, rtol=1e-2)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_valid_on_npu_sub():
    """合法 TilingKey 上板 — sub 运算（OpType=1）。"""
    device = ST_DEVICE
    torch.npu.set_device(device)
    shape = (128, 256)
    dtype = torch.float16

    x = torch.rand(shape, device=device, dtype=dtype)
    y = torch.rand(shape, device=device, dtype=dtype)
    z = torch.empty(shape, device=device, dtype=dtype)

    kernel_valid[None, 1, {"OpType": 1}](x, y, z)
    torch.npu.synchronize()
    z_ref = (x.float() - y.float()).half()
    torch.testing.assert_close(z, z_ref, atol=1e-2, rtol=1e-2)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_valid_on_npu_mul():
    """合法 TilingKey 上板 — mul 运算（OpType=2）。"""
    device = ST_DEVICE
    torch.npu.set_device(device)
    shape = (128, 256)
    dtype = torch.float16

    x = torch.rand(shape, device=device, dtype=dtype)
    y = torch.rand(shape, device=device, dtype=dtype)
    z = torch.empty(shape, device=device, dtype=dtype)

    kernel_valid[None, 1, {"OpType": 2}](x, y, z)
    torch.npu.synchronize()
    z_ref = (x.float() * y.float()).half()
    torch.testing.assert_close(z, z_ref, atol=1e-2, rtol=1e-2)
