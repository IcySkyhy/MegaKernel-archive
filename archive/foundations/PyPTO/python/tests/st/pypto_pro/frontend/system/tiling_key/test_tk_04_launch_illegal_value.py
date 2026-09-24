# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""TilingKey kernel launch 非法 value 验证测试。

验证 kernel launch 时传入不在候选集中的 tiling_key 值会抛出 ``ValueError``。
正例：candidate set 内的值正确执行并校验精度。
"""

import os

import pypto_pro.language as pl
from pypto_pro.runtime.tilingkey import TilingKeyField
import pytest
import torch

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"


# ---- 严格 TilingKey：仅允许 0 和 1 ----------------------------------------
class StrictKey04:
    StrictVal = TilingKeyField(bits=2, values=[0, 1])


# ---- kernel：StrictVal==0 做 add，否则做 sub --------------------------------
@pl.jit(auto_mutex=True, tiling_key=StrictKey04)
def kernel_strict(
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

                if StrictVal == 0:  # noqa: F821
                    pl.add(tile_c, tile_a, tile_b)
                else:
                    pl.sub(tile_c, tile_a, tile_b)

                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                pl.store(z, tile_c, [i, j])


# ---- 辅助函数 --------------------------------------------------------------
def _run_npu_test(key, ref_fn, shape=(128, 256)):
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    dtype = torch.float16
    x = torch.rand(shape, device=device, dtype=dtype)
    y = torch.rand(shape, device=device, dtype=dtype)
    z = torch.empty(shape, device=device, dtype=dtype)
    kernel_strict[None, 1, key](x, y, z)
    torch.npu.synchronize()
    z_ref = ref_fn(x.float(), y.float()).half()
    torch.testing.assert_close(z, z_ref, atol=1e-2, rtol=1e-2)


# ---- 简单用例：正例 --------------------------------------------------------
@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_strictval_0_add():
    """StrictVal=0 应走 add 分支"""
    _run_npu_test({"StrictVal": 0}, lambda a, b: a + b)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_strictval_1_sub():
    """StrictVal=1 应走 sub 分支"""
    _run_npu_test({"StrictVal": 1}, lambda a, b: a - b)


# ---- 反例：非法 value 在 launch 时被拒绝 ----------------------------------
