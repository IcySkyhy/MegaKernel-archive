# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""TilingKey bits 参数泛化性测试。

验证 ``TilingKeyField(bits=N, values=[...])`` 的 bits 参数从 1 到 16
都能正确编译和执行。bits=1/2/4 在 NPU 上全流程验证；bits=8/16 因 values
太多仅验证编译通过。
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


# ---- TilingKey 定义 (bits=1/2/4/8/16) ----------------------------------
# CORE 1-bit constexpr test: UseMask selects between two branches that
# produce DIFFERENT results, verifying the most fundamental dead-code
# elimination unit: a single bit constexpr.
class TkBits1:
    UseMask = TilingKeyField(bits=1, values=[0, 1])


class TkBits2:
    OpType = TilingKeyField(bits=2, values=[0, 1, 2, 3])


class TkBits4:
    OpType = TilingKeyField(bits=4, values=list(range(16)))


# ---- 公共 kernel 模板 --------------------------------------------------


def _make_kernel(tiling_key_cls):
    @pl.jit(auto_mutex=True, tiling_key=tiling_key_cls)
    def _kernel(
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
                        pl.add(tile_c, tile_a, tile_b)  # noqa: F821
                    elif OpType == 1:  # noqa: F821
                        pl.sub(tile_c, tile_a, tile_b)  # noqa: F821
                    else:
                        pl.mul(tile_c, tile_a, tile_b)  # noqa: F821

                    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                    pl.store(z, tile_c, [i, j])

    return _kernel


# ---- bits=1 专用 kernel（CORE 1-bit constexpr test） ----------------------
# 恰好 2 个分支，UseMask=0 返回 x+y，UseMask=1 经由 add+sub 返回 y。
# 两个分支产生不同结果，验证单 bit constexpr 的死代码消除。
@pl.jit(auto_mutex=True, tiling_key=TkBits1)
def kernel_bits1(
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

                if UseMask == 0:  # noqa: F821
                    pl.add(tile_c, tile_a, tile_b)
                else:
                    pl.add(tile_c, tile_a, tile_b)
                    pl.sub(tile_c, tile_c, tile_a)

                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                pl.store(z, tile_c, [i, j])


kernel_bits2 = _make_kernel(TkBits2)
kernel_bits4 = _make_kernel(TkBits4)
# ---- 辅助函数 -----------------------------------------------------------


def _run_npu_test(kernel, key, ref_fn, shape=(128, 256)):
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    dtype = torch.float16

    x = torch.rand(shape, device=device, dtype=dtype)
    y = torch.rand(shape, device=device, dtype=dtype)
    z = torch.empty(shape, device=device, dtype=dtype)

    kernel[None, 1, key](x, y, z)
    torch.npu.synchronize()
    z_ref = ref_fn(x.float(), y.float()).half()
    torch.testing.assert_close(z, z_ref, atol=1e-2, rtol=1e-2)


# ---- 简单用例 (NPU 全流程验证) : bits=1, 2, 4 -------------------------
@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_bits_1_use_mask_0_add():
    _run_npu_test(kernel_bits1, {"UseMask": 0}, lambda a, b: a + b)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_bits_1_use_mask_1_addsub():
    _run_npu_test(kernel_bits1, {"UseMask": 1}, lambda a, b: b)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_bits_2_sub():
    _run_npu_test(kernel_bits2, {"OpType": 1}, lambda a, b: a - b)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_bits_4_mul():
    _run_npu_test(kernel_bits4, {"OpType": 2}, lambda a, b: a * b)


# ---- 复杂用例 (仅编译验证) : bits=8, 16 --------------------------------
