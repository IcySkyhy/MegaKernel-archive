# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Fillpad-family ops: fillpad, fillpad_expand, fillpad_inplace.

Doc: docs/zh/pypto_pro/api/SIMD-API/计算API/Memory矢量计算/{fillpad, fillpad_expand, fillpad_inplace}

make_tile_group + auto_mutex style. Verifies padding region fill
after set_validshape, including expand and inplace variants.
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


def _make_fillpad_inputs(device, output_shape):
    x = torch.full((8, 8), -99, device=device, dtype=torch.int32)
    x[:5, :7] = torch.arange(35, device=device, dtype=torch.int32).reshape(5, 7)
    z = torch.empty(output_shape, device=device, dtype=torch.int32)
    z_ref = torch.zeros(output_shape, device=device, dtype=torch.int32)
    z_ref[:5, :7] = x[:5, :7]
    return x, z, z_ref


# ===========================================================================
# fillpad —— 填充 padding 区域，dst 与 src shape 一致、地址不同
# ===========================================================================
@pl.jit(auto_mutex=True)
def fillpad_kernel(
    x: pl.Tensor[[8, 8], pl.DT_INT32],
    z: pl.Tensor[[8, 8], pl.DT_INT32],
):
    src_type = pl.TileType(shape=[8, 8], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    dst_type = pl.TileType(shape=[8, 8], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec, pad=pl.TilePad.zero)
    src = pl.make_tile_group(type=src_type, addrs=0x0000, mutex_ids=[0])
    dst = pl.make_tile_group(type=dst_type, addrs=0x0100, mutex_ids=[1])
    with pl.section_vector():
        cur_src = src.current()
        cur_dst = dst.current()
        pl.set_validshape(cur_src, [5, 7])
        pl.load(cur_src, x, [0, 0])
        pl.fillpad(cur_dst, cur_src)
        pl.store(z, cur_dst, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_fillpad():
    device = ST_DEVICE
    _require_a5(device)
    x, z, z_ref = _make_fillpad_inputs(device, (8, 8))
    fillpad_kernel(x, z)
    torch.npu.synchronize()
    torch.testing.assert_close(z, z_ref)
    logging.info("fillpad result equal!")


# ===========================================================================
# fillpad_expand —— 扩展填充，dst shape 大于 src
# ===========================================================================
@pl.jit(auto_mutex=True)
def fillpad_expand_kernel(
    x: pl.Tensor[[8, 8], pl.DT_INT32],
    z: pl.Tensor[[8, 16], pl.DT_INT32],
):
    src_type = pl.TileType(shape=[8, 8], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    dst_type = pl.TileType(shape=[8, 16], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec, pad=pl.TilePad.zero)
    src = pl.make_tile_group(type=src_type, addrs=0x0000, mutex_ids=[0])
    dst = pl.make_tile_group(type=dst_type, addrs=0x0100, mutex_ids=[1])
    with pl.section_vector():
        cur_src = src.current()
        cur_dst = dst.current()
        pl.set_validshape(cur_src, [5, 7])
        pl.load(cur_src, x, [0, 0])
        pl.fillpad(cur_dst, cur_src, mode=pl.FillPadMode.EXPAND)
        pl.store(z, cur_dst, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_fillpad_expand():
    device = ST_DEVICE
    _require_a5(device)
    x, z, z_ref = _make_fillpad_inputs(device, (8, 16))
    fillpad_expand_kernel(x, z)
    torch.npu.synchronize()
    torch.testing.assert_close(z, z_ref)
    logging.info("fillpad_expand result equal!")


# ===========================================================================
# fillpad_inplace —— 就地填充，dst 与 src 同一地址
#   src/dst 共用 addr=0x0000，但用不同 mutex_id 避免 auto_mutex 自死锁。
# ===========================================================================
@pl.jit(auto_mutex=True)
def fillpad_inplace_kernel(
    x: pl.Tensor[[8, 8], pl.DT_INT32],
    z: pl.Tensor[[8, 8], pl.DT_INT32],
):
    src_type = pl.TileType(
        shape=[8, 8], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec, pad=pl.TilePad.zero, valid_shape=[-1, -1]
    )
    dst_type = pl.TileType(shape=[8, 8], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec, pad=pl.TilePad.zero)
    src = pl.make_tile_group(type=src_type, addrs=0x0000, mutex_ids=[0])
    dst = pl.make_tile_group(type=dst_type, addrs=0x0000, mutex_ids=[1])
    with pl.section_vector():
        cur_src = src.current()
        cur_dst = dst.current()
        pl.set_validshape(cur_src, [5, 7])
        pl.load(cur_src, x, [0, 0])
        pl.fillpad(cur_dst, cur_src, mode=pl.FillPadMode.INPLACE)
        pl.store(z, cur_dst, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_fillpad_inplace():
    device = ST_DEVICE
    _require_a5(device)
    x, z, z_ref = _make_fillpad_inputs(device, (8, 8))
    fillpad_inplace_kernel(x, z)
    torch.npu.synchronize()
    torch.testing.assert_close(z, z_ref)
    logging.info("fillpad_inplace result equal!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for fn in [test_fillpad, test_fillpad_expand, test_fillpad_inplace]:
        fn()
    logging.info("\nAll fillpad doc examples passed!")
