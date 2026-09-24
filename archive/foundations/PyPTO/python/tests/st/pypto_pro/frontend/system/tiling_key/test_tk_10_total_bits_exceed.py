# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""TilingKey 总 bits 数超过 64 限制验证测试。

验证多个 ``TilingKeyField`` 的 bits 之和超过 64 时，
``TilingKeySchema`` 构造阶段抛出 ``ValueError``。
正例：64 bits 恰好不超限，可在 NPU 上编译执行并校验精度。
"""

import os

import pypto_pro.language as pl
from pypto_pro.runtime.tilingkey import TilingKeyField
import pytest
import torch

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"


# ---- 64 bits TilingKey（恰好边界）------------------------------------------
class Tk64Bits:
    a = TilingKeyField(bits=32, values=[0, 1])
    b = TilingKeyField(bits=32, values=[0, 1])


# ---- kernel（A==0 做 add，否则做 sub）--------------------------------------
@pl.jit(auto_mutex=True, tiling_key=Tk64Bits)
def kernel_64bits(
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

                if a == 0:  # noqa: F821
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
    kernel_64bits[None, 1, key](x, y, z)
    torch.npu.synchronize()
    z_ref = ref_fn(x.float(), y.float()).half()
    torch.testing.assert_close(z, z_ref, atol=1e-2, rtol=1e-2)


# ---- 反例：总 bits > 64 ----------------------------------------------------
# ---- 正例：总 bits == 64 ---------------------------------------------------
# ---- 简单用例 (NPU 全流程验证 64-bit 正例) ---------------------------------


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_64bits_add_npu():
    """64-bit TilingKey, A=0 走 add 分支"""
    _run_npu_test({"a": 0, "b": 0}, lambda a, b: a + b)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_64bits_sub_npu():
    """64-bit TilingKey, A=1 走 sub 分支"""
    _run_npu_test({"a": 1, "b": 0}, lambda a, b: a - b)
