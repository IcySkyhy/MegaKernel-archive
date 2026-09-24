# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Dynamic shape syntax in Tensor type declarations.

Doc: docs/zh/pypto_pro/api/SIMD-API/basic_data_structures/Tensor.md

Verifies pl.DYNAMIC tensor declarations compile and run correctly
(single-tile scenario).
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


# 动态维度语法验证：单 tile 场景（无循环）
@pl.jit(auto_mutex=True)
def dynamic_tensor_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a_group = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_b_group = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    tile_out_group = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[2])
    with pl.section_vector():
        tile_a = tile_a_group.current()
        tile_b = tile_b_group.current()
        tile_out = tile_out_group.current()
        pl.load(tile_a, a, [0, 0])
        pl.load(tile_b, b, [0, 0])
        pl.add(tile_out, tile_a, tile_b)
        pl.store(out, tile_out, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_dynamic_tensor():
    """验证 pl.DYNAMIC 语法可用（单 tile 场景）"""
    device = ST_DEVICE
    _require_a5(device)
    torch.manual_seed(0)
    a = torch.randn(64, 64, device=device, dtype=torch.float32)
    b = torch.randn(64, 64, device=device, dtype=torch.float32)
    out = torch.zeros(64, 64, device=device, dtype=torch.float32)
    dynamic_tensor_kernel(a, b, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a + b, rtol=1e-2, atol=1e-2)
    logging.info("Dynamic shape syntax validation passed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    test_dynamic_tensor()
    logging.info("\nDynamic shape syntax test passed!")
