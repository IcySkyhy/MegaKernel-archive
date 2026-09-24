# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""TilingKey 字段候选数量超出 bits 容量验证测试。

验证 ``TilingKeyField(bits=N, values=[...])`` 的候选数量超过 ``2**N`` 时，
``TilingKeySchema`` 构造阶段抛出 ``ValueError``。字段的实际值本身不受 bit
宽限制，bit 字段存储的是其在 ``values`` 中的下标。
"""

import os

import pypto_pro.language as pl
from pypto_pro.runtime.tilingkey import TilingKeyField
import pytest
import torch

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"


# ---- valid TilingKey 类（bits=1 和 bits=2 各一个） -------------------------
class TkBits1:
    OpType = TilingKeyField(bits=1, values=[0, 1])


class TkBits2:
    OpType = TilingKeyField(bits=2, values=[0, 3])


# ---- kernel 模板 ----------------------------------------------------------
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
        tile_a_db = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[1])
        tile_b_db = pl.make_tile_group(type=tile_type, addrs=0x8000, mutex_ids=[2])
        tile_c_db = pl.make_tile_group(type=tile_type, addrs=0xC000, mutex_ids=[3])

        with pl.section_vector():
            for i in pl.range(0, m, 64):
                for j in pl.range(0, n, 128):
                    tile_a = tile_a_db.next()
                    tile_b = tile_b_db.next()
                    tile_c = tile_c_db.next()
                    pl.load(tile_a, x, [i, j])
                    pl.load(tile_b, y, [i, j])

                    if OpType == 0:  # noqa: F821
                        pl.add(tile_c, tile_a, tile_b)
                    else:
                        pl.sub(tile_c, tile_a, tile_b)

                    pl.store(z, tile_c, [i, j])

    return _kernel


kernel_bits1 = _make_kernel(TkBits1)
kernel_bits2 = _make_kernel(TkBits2)
# ---- 辅助函数 --------------------------------------------------------------


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


# ---- 反例: 候选数量超出 bits 容量 ------------------------------------------
# ---- 正例: 候选值可超出 bits 范围 ------------------------------------------
# ---- 简单用例 (NPU 全流程验证正例) -----------------------------------------


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_bits_1_add():
    _run_npu_test(kernel_bits1, {"OpType": 0}, lambda a, b: a + b)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_bits_1_sub():
    _run_npu_test(kernel_bits1, {"OpType": 1}, lambda a, b: a - b)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_bits_2_add():
    _run_npu_test(kernel_bits2, {"OpType": 0}, lambda a, b: a + b)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_bits_2_sub():
    _run_npu_test(kernel_bits2, {"OpType": 3}, lambda a, b: a - b)
