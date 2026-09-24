# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Cache control (dcci) and tail-block shape (set_validshape).

Doc: docs/zh/pypto_pro/api/SIMD-API/计算API/{缓存控制/dcci, Memory矢量计算/转置与元素读写/set_validshape}

Verifies GM cache invalidation for host-device visibility and valid shape
setting for tail-block processing.
"""

import logging
import os

import pypto_pro.language as pl
import pytest
import torch

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"


def _require_a5(device):
    try:
        torch.npu.set_device(device)
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
    name = torch.npu.get_device_name()
    if "Ascend950" not in name:
        pytest.skip(f"Current device is {name}, not A5 (Ascend950). Skip.")


# ===========================================================================
# dcci —— GM 缓存失效（确保 host 写入对 device 可见）
#   照搬 a3/system/test_system_dcci.py。
# ===========================================================================
@pl.jit()
def dcci_kernel(
    inp: pl.Tensor[[16, 16], pl.DT_FP32],
    output: pl.Tensor[[16, 16], pl.DT_FP32],
):
    pl.system.dcci(inp, 512, cache_line=pl.CacheLine.SINGLE_CACHE_LINE)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_dcci():
    device = ST_DEVICE
    _require_a5(device)
    inp = torch.ones(16, 16, dtype=torch.float32, device=device)
    out = torch.ones(16, 16, dtype=torch.float32, device=device)
    dcci_kernel[None, 1](inp, out)
    torch.npu.synchronize()
    assert torch.allclose(out.cpu(), inp.cpu(), rtol=1e-5, atol=1e-5)
    logging.info("dcci result equal!")


# ===========================================================================
# set_validshape —— 尾块有效形状设置
#   tile shape [64,128]，实际数据只有 [rows, cols] 有效。
#   照搬 ir/operators/test_block_ops.py 的 set_validshape 用法。
# ===========================================================================
@pl.jit(auto_mutex=True)
def validshape_kernel(
    a: pl.Tensor[[64, 128], pl.DT_FP32],
    rows: pl.DT_INT64,
    cols: pl.DT_INT64,
    out: pl.Tensor[[64, 128], pl.DT_FP32],
):
    tile_type = pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    tile_group = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0])
    with pl.section_vector():
        tile = tile_group.current()
        pl.set_validshape(tile, [rows, cols])
        pl.load(tile, a, [0, 0])
        pl.store(out, tile, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_set_validshape():
    device = ST_DEVICE
    _require_a5(device)
    torch.manual_seed(0)
    rows, cols = 37, 91
    a = torch.randn(64, 128, device=device, dtype=torch.float32)
    out = torch.zeros(64, 128, device=device, dtype=torch.float32)
    validshape_kernel(a, rows, cols, out)
    torch.npu.synchronize()
    expected = torch.zeros_like(a)
    expected[:rows, :cols] = a[:rows, :cols]
    torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)
    logging.info("set_validshape result equal!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    test_dcci()
    test_set_validshape()
    logging.info("\nAll dcci/validshape examples passed!")
