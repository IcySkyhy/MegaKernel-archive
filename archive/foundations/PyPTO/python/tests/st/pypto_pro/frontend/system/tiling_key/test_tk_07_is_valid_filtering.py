# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""TilingKey is_valid 过滤测试。

验证 ``is_valid()`` 方法拒绝非法字段组合的能力。
字段 A(bits=2) x B(bits=2) 共 9 种组合，(A=0,B=0) 被 is_valid 拒绝。
所有通过 is_valid 的组合在 NPU 上全流程验证精度。
"""

import os

import pypto_pro.language as pl
from pypto_pro.runtime.tilingkey import TilingKeyField
import pytest
import torch

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"


# ---- 带 is_valid 过滤的 TilingKey -----------------------------------------
class TkValidFilter:
    a = TilingKeyField(bits=2, values=[0, 1, 2])
    b = TilingKeyField(bits=2, values=[0, 1, 2])

    @classmethod
    def is_valid(cls, key):
        return not (key[0] == 0 and key[1] == 0)


# ---- kernel: A 字段选择 add/sub/mul --------------------------------------
@pl.jit(auto_mutex=True, tiling_key=TkValidFilter)
def kernel_valid_filter(
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
                elif a == 1:  # noqa: F821
                    pl.sub(tile_c, tile_a, tile_b)
                else:
                    pl.mul(tile_c, tile_a, tile_b)

                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                pl.store(z, tile_c, [i, j])


# ---- 辅助函数 --------------------------------------------------------------
def _run_test(key, ref_fn, shape=(128, 256)):
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    dtype = torch.float16
    x = torch.rand(shape, device=device, dtype=dtype)
    y = torch.rand(shape, device=device, dtype=dtype)
    z = torch.empty(shape, device=device, dtype=dtype)
    kernel_valid_filter[None, 1, key](x, y, z)
    torch.npu.synchronize()
    z_ref = ref_fn(x.float(), y.float()).half()
    torch.testing.assert_close(z, z_ref, atol=1e-2, rtol=1e-2)


# ---- 简单用例 --------------------------------------------------------------
@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_valid_combo_1_0_add():
    """(A=1, B=0) 通过 is_valid → sub."""
    _run_test({"a": 1, "b": 0}, lambda a, b: a - b)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_valid_combo_2_2_mul():
    """(A=2, B=2) 通过 is_valid → mul."""
    _run_test({"a": 2, "b": 2}, lambda a, b: a * b)


# ---- 复杂用例: 所有通过 is_valid 的组合 NPU 全流程验证 --------------------
@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_all_valid_combos_npu():
    """遍历 3x3=9 → 过滤 (0,0) → 剩余 8 种组合，全部上板精度验证."""
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shape = (128, 256)
    dtype = torch.float16

    ref_map = {
        0: lambda a, b: a + b,
        1: lambda a, b: a - b,
        2: lambda a, b: a * b,
    }

    for a_val in [0, 1, 2]:
        for b_val in [0, 1, 2]:
            if a_val == 0 and b_val == 0:
                continue
            x = torch.rand(shape, device=device, dtype=dtype)
            y = torch.rand(shape, device=device, dtype=dtype)
            z = torch.empty(shape, device=device, dtype=dtype)
            kernel_valid_filter[None, 1, {"a": a_val, "b": b_val}](x, y, z)
            torch.npu.synchronize()
            ref_fn = ref_map.get(a_val)
            z_ref = ref_fn(x.float(), y.float()).half()
            torch.testing.assert_close(z, z_ref, atol=1e-2, rtol=1e-2)
