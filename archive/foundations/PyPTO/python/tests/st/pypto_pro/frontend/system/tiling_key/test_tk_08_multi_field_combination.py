# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""TilingKey 多字段组合测试。

验证 2-3 个 ``TilingKeyField`` 同时定义时，各种值组合的正确性。
"""

import os

import pypto_pro.language as pl
from pypto_pro.runtime.tilingkey import TilingKeyField
import pytest
import torch

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"


# ---- 2 字段 TilingKey (2 值 x 2 值 = 4 组合) ----------------------------
class Tk2x2:
    OperandA = TilingKeyField(bits=1, values=[0, 1])
    OperandB = TilingKeyField(bits=1, values=[0, 1])


# ---- 2 字段 TilingKey (3 值 x 2 值 = 6 组合) ----------------------------
class Tk3x2:
    OpType = TilingKeyField(bits=2, values=[0, 1, 2])
    Mode = TilingKeyField(bits=1, values=[0, 1])


# ---- 3 字段 TilingKey ---------------------------------------------------
class Tk3Fields:
    OpType = TilingKeyField(bits=2, values=[0, 1, 2])
    ReuseA = TilingKeyField(bits=1, values=[0, 1])
    ReuseB = TilingKeyField(bits=1, values=[0, 1])


# ---- 带 is_valid 约束的多字段 TilingKey ----------------------------------
class TkConstrained:
    OpType = TilingKeyField(bits=1, values=[0, 1])
    Mode = TilingKeyField(bits=1, values=[0, 1])

    @classmethod
    def is_valid(cls, key):
        return not (key[0] == 0 and key[1] == 0)


# ---- kernel: 2 字段 (2x2) -----------------------------------------------
@pl.jit(auto_mutex=True, tiling_key=Tk2x2)
def kernel_2x2(
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

                if OperandA == 0:  # noqa: F821
                    pl.add(tile_c, tile_a, tile_b)
                elif OperandA == 1:  # noqa: F821
                    if OperandB == 0:  # noqa: F821
                        pl.sub(tile_c, tile_a, tile_b)
                    else:
                        pl.mul(tile_c, tile_a, tile_b)

                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                pl.store(z, tile_c, [i, j])


# ---- kernel: 2 字段 (3x2) -----------------------------------------------
@pl.jit(auto_mutex=True, tiling_key=Tk3x2)
def kernel_3x2(
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

                if Mode == 0:  # noqa: F821
                    if OpType == 0:  # noqa: F821
                        pl.add(tile_c, tile_a, tile_b)
                    elif OpType == 1:  # noqa: F821
                        pl.sub(tile_c, tile_a, tile_b)
                    else:
                        pl.mul(tile_c, tile_a, tile_b)
                else:
                    if OpType == 0:  # noqa: F821
                        pl.mul(tile_c, tile_a, tile_b)
                    elif OpType == 1:  # noqa: F821
                        pl.add(tile_c, tile_a, tile_b)
                    else:
                        pl.add(tile_c, tile_a, tile_b)

                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                pl.store(z, tile_c, [i, j])


# ---- 辅助函数 -----------------------------------------------------------
def _run_test(kernel, key, ref_fn, shape=(128, 256)):
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


# ---- 简单用例 -----------------------------------------------------------
@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_2field_a_add():
    _run_test(kernel_2x2, {"OperandA": 0, "OperandB": 0}, lambda a, b: a + b)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_2field_sub():
    _run_test(kernel_2x2, {"OperandA": 1, "OperandB": 0}, lambda a, b: a - b)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_3x2_mode0_sub():
    _run_test(kernel_3x2, {"OpType": 1, "Mode": 0}, lambda a, b: a - b)


# ---- 复杂用例 -----------------------------------------------------------
@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_3x2_mode1_min():
    _run_test(kernel_3x2, {"OpType": 1, "Mode": 1}, lambda a, b: a + b)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_2field_mul():
    _run_test(kernel_2x2, {"OperandA": 1, "OperandB": 1}, lambda a, b: a * b)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_3x2_all_combos():
    """遍历 3x2 全部 6 种组合"""
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shape = (128, 256)
    dtype = torch.float16

    ref_map = {
        (0, 0): lambda a, b: a + b,
        (1, 0): lambda a, b: a - b,
        (2, 0): lambda a, b: a * b,
        (0, 1): lambda a, b: a * b,
        (1, 1): lambda a, b: a + b,
        (2, 1): lambda a, b: a + b,
    }

    for (op_type, mode), ref_fn in ref_map.items():
        x = torch.rand(shape, device=device, dtype=dtype)
        y = torch.rand(shape, device=device, dtype=dtype)
        z = torch.empty(shape, device=device, dtype=dtype)
        kernel_3x2[None, 1, {"OpType": op_type, "Mode": mode}](x, y, z)
        torch.npu.synchronize()
        z_ref = ref_fn(x.float(), y.float()).half()
        torch.testing.assert_close(z, z_ref, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
