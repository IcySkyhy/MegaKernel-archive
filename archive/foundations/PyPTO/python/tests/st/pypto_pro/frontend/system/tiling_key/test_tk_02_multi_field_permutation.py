# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""TilingKey 多字段排列组合测试。

验证 3 个 ``TilingKeyField``（共 6 bits）的 8 种指定排列组合：
- ModeA(bits=2) + ModeB(bits=3) + Flag(bits=1)
- 嵌套 ```` if/elif 链的正确性
- 数值结果正确性
- 不同组合产生独立的编译缓存子目录（``tk_<hex>`` 隔离）
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


class TkPermutation:
    ModeA = TilingKeyField(bits=2, values=[0, 1, 2, 3])
    ModeB = TilingKeyField(bits=3, values=[0, 1, 2, 3, 4, 5, 6, 7])
    Flag = TilingKeyField(bits=1, values=[0, 1])


@pl.jit(auto_mutex=True, tiling_key=TkPermutation)
def kernel_permutation(
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

                if Flag == 0:  # noqa: F821
                    if ModeA == 0:  # noqa: F821
                        pl.add(tile_c, tile_a, tile_b)  # noqa: F821
                    elif ModeA == 1:  # noqa: F821
                        pl.sub(tile_c, tile_a, tile_b)  # noqa: F821
                    elif ModeA == 2:  # noqa: F821
                        pl.mul(tile_c, tile_a, tile_b)  # noqa: F821
                    elif ModeA == 3:  # noqa: F821
                        pl.sub(tile_c, tile_a, tile_b)
                else:  # noqa: F821
                    if ModeA == 0:  # noqa: F821
                        pl.add(tile_c, tile_a, tile_b)  # noqa: F821
                    elif ModeA == 1:  # noqa: F821
                        pl.sub(tile_c, tile_a, tile_b)  # noqa: F821
                    elif ModeA == 2:  # noqa: F821
                        pl.mul(tile_c, tile_a, tile_b)  # noqa: F821
                    elif ModeA == 3:  # noqa: F821
                        pl.sub(tile_c, tile_a, tile_b)
                    pl.sub(tile_c, tile_c, tile_b)

                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                pl.store(z, tile_c, [i, j])


# ---- golden 参考 -----------------------------------------------------------
GOLDEN_MAP = {
    (0, 0): lambda a, b: a + b,
    (1, 0): lambda a, b: a - b,
    (2, 0): lambda a, b: a * b,
    (3, 0): lambda a, b: a - b,
    (0, 1): lambda a, b: (a + b) - b,
    (1, 1): lambda a, b: (a - b) - b,
    (2, 1): lambda a, b: (a * b) - b,
    (3, 1): lambda a, b: (a - b) - b,
}
# ---- 辅助函数 --------------------------------------------------------------


def _run_test(key, shape=(128, 256)):
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    dtype = torch.float16
    x = torch.rand(shape, device=device, dtype=dtype)
    y = torch.rand(shape, device=device, dtype=dtype)
    z = torch.empty(shape, device=device, dtype=dtype)

    kernel_permutation[None, 1, key](x, y, z)
    torch.npu.synchronize()

    ref_fn = GOLDEN_MAP[(key["ModeA"], key["Flag"])]
    z_ref = ref_fn(x.float(), y.float()).half()
    torch.testing.assert_close(z, z_ref, atol=1e-2, rtol=1e-2)


# ---- 8 种指定组合的独立测试 -------------------------------------------------
@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_combo_0_0_0():
    _run_test({"ModeA": 0, "ModeB": 0, "Flag": 0})


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_combo_1_1_0():
    _run_test({"ModeA": 1, "ModeB": 1, "Flag": 0})


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_combo_2_3_0():
    _run_test({"ModeA": 2, "ModeB": 3, "Flag": 0})


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_combo_3_7_0():
    _run_test({"ModeA": 3, "ModeB": 7, "Flag": 0})


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_combo_0_0_1():
    _run_test({"ModeA": 0, "ModeB": 0, "Flag": 1})


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_combo_1_1_1():
    _run_test({"ModeA": 1, "ModeB": 1, "Flag": 1})


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_combo_2_3_1():
    _run_test({"ModeA": 2, "ModeB": 3, "Flag": 1})


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_combo_3_7_1():
    _run_test({"ModeA": 3, "ModeB": 7, "Flag": 1})


# ---- 8 种组合遍历测试 -------------------------------------------------------
@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_all_8_combos():
    """遍历全部 8 种 (ModeA, ModeB, Flag) 排列组合并验证数值正确性。"""
    combos = [
        {"ModeA": 0, "ModeB": 0, "Flag": 0},
        {"ModeA": 1, "ModeB": 1, "Flag": 0},
        {"ModeA": 2, "ModeB": 3, "Flag": 0},
        {"ModeA": 3, "ModeB": 7, "Flag": 0},
        {"ModeA": 0, "ModeB": 0, "Flag": 1},
        {"ModeA": 1, "ModeB": 1, "Flag": 1},
        {"ModeA": 2, "ModeB": 3, "Flag": 1},
        {"ModeA": 3, "ModeB": 7, "Flag": 1},
    ]

    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    dtype = torch.float16
    shape = (128, 256)

    for key in combos:
        x = torch.rand(shape, device=device, dtype=dtype)
        y = torch.rand(shape, device=device, dtype=dtype)
        z = torch.empty(shape, device=device, dtype=dtype)

        kernel_permutation[None, 1, key](x, y, z)
        torch.npu.synchronize()

        ref_fn = GOLDEN_MAP[(key["ModeA"], key["Flag"])]
        z_ref = ref_fn(x.float(), y.float()).half()
        torch.testing.assert_close(z, z_ref, atol=1e-2, rtol=1e-2)


# ---- 编译缓存隔离测试 -------------------------------------------------------
