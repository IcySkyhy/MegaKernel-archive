# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Transpose and element read/write (subscript getval / setval / transpose).

Doc: docs/zh/pypto_pro/api/SIMD-API/计算API/Memory矢量计算/转置与元素读写

make_tile_group + auto_mutex style. Verifies scalar read/write
via S pipeline and transpose(dst, src) correctness.
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
# getval / setval —— tile 内元素读写，走 S（标量）流水
#   把 tile[0] 读出，写到 tile[1]，store 回 GM。golden: a[0,1] == a[0,0]
# ===========================================================================
@pl.jit(auto_mutex=True)
def getval_setval_kernel(
    a: pl.Tensor[[64, 128], pl.DT_FP16],
):
    tile_a_group = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids=[0],
    )
    with pl.section_vector():
        tile_a = tile_a_group.current()
        pl.load(tile_a, a, [0, 0])
        value = tile_a[0, 0]  # 读 tile 第 0 个元素
        tile_a[0, 1] = value  # 写到第 1 个位置
        pl.store(a, tile_a, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_getval_setval():
    device = ST_DEVICE
    _require_a5(device)
    torch.manual_seed(0)
    a = torch.rand(64, 128, device=device, dtype=torch.float16)
    a_orig_00 = a[0, 0].item()
    getval_setval_kernel(a)
    torch.npu.synchronize()
    # a[0,1] 应被写成原 a[0,0]
    assert abs(a[0, 1].item() - a_orig_00) < 1e-3, f"getval/setval failed: a[0,1]={a[0, 1].item()}, expect {a_orig_00}"
    logging.info("tile getval/setval result equal!")


# ===========================================================================
#   仓内无独立 block transpose 调用，按 transpose(dst, src) 推导。
# ===========================================================================
@pl.jit(auto_mutex=True)
def transpose_kernel(
    a: pl.Tensor[[64, 64], pl.DT_FP16],
    out: pl.Tensor[[64, 64], pl.DT_FP16],
):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_out = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.transpose(cur_out, cur_a)
        pl.store(out, cur_out, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_transpose():
    device = ST_DEVICE
    _require_a5(device)
    torch.manual_seed(0)
    a = torch.rand(64, 64, device=device, dtype=torch.float16)
    out = torch.zeros(64, 64, device=device, dtype=torch.float16)
    transpose_kernel(a, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a.t().contiguous(), rtol=1e-2, atol=1e-2)
    logging.info("transpose result equal!")


# ===========================================================================
# tensor getval / setval —— tensor 级元素读写
#   读 scale_tensor[0]，写到 scale_tensor[1]
# ===========================================================================
@pl.jit(auto_mutex=True)
def tensor_getval_setval_kernel(
    scale_tensor: pl.Tensor[[2], pl.DT_FP32],
):
    scale = scale_tensor[0]
    scale_tensor[1] = scale


# ===========================================================================
# tensor getval / setval (5D) —— 5 维 tensor 多下标元素读写
#   读 tensor5d[0,0,0,0,0]，写到 tensor5d[1,1,1,1,1]
#   行主序线性化: offset = i0*(2*2*2*2) + i1*(2*2*2) + i2*(2*2) + i3*2 + i4
#   [0,0,0,0,0] → 0,  [1,1,1,1,1] → 31
# ===========================================================================
@pl.jit(auto_mutex=True)
def tensor_getval_setval_5d_kernel(
    tensor5d: pl.Tensor[[2, 2, 2, 2, 2], pl.DT_FP32],
):
    value = tensor5d[0, 0, 0, 0, 0]
    tensor5d[1, 1, 1, 1, 1] = value


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_tensor_getval_setval():
    device = ST_DEVICE
    _require_a5(device)
    scale_val = 2.5
    scale_tensor = torch.zeros(2, device=device, dtype=torch.float32)
    scale_tensor[0] = scale_val

    tensor_getval_setval_kernel(scale_tensor)
    torch.npu.synchronize()

    assert abs(scale_tensor[1].item() - scale_val) < 1e-6, (
        f"tensor setval failed: scale_tensor[1]={scale_tensor[1].item()}, expect {scale_val}"
    )
    logging.info("tensor getval/setval result equal!")

    # 5D tensor getval/setval
    val_5d = 7.3
    tensor5d = torch.zeros(2, 2, 2, 2, 2, device=device, dtype=torch.float32)
    tensor5d[0, 0, 0, 0, 0] = val_5d

    tensor_getval_setval_5d_kernel(tensor5d)
    torch.npu.synchronize()

    assert abs(tensor5d[1, 1, 1, 1, 1].item() - val_5d) < 1e-6, (
        f"5D tensor setval failed: tensor5d[1,1,1,1,1]={tensor5d[1, 1, 1, 1, 1].item()}, expect {val_5d}"
    )
    logging.info("5D tensor getval/setval result equal!")


# ===========================================================================
# Ptr + make_tensor → getval / setval
#   通过 pl.Ptr 传入裸指针，内部 make_tensor 构造 tensor 视图后进行标量读写
#   读 tensor_view[0]，写到 tensor_view[1]
# ===========================================================================
@pl.jit()
def ptr_make_tensor_getval_setval_kernel(
    data_ptr: pl.Ptr[pl.DT_FP32],
):
    tensor_view = pl.make_tensor(data_ptr, [2])
    scale = tensor_view[0]
    tensor_view[1] = scale


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_ptr_make_tensor_getval_setval():
    device = ST_DEVICE
    _require_a5(device)
    scale_val = 3.14
    data = torch.zeros(2, device=device, dtype=torch.float32)
    data[0] = scale_val

    ptr_make_tensor_getval_setval_kernel(data)
    torch.npu.synchronize()

    assert abs(data[1].item() - scale_val) < 1e-6, (
        f"ptr make_tensor getval/setval failed: data[1]={data[1].item()}, expect {scale_val}"
    )
    logging.info("ptr make_tensor getval/setval result equal!")


def _doc_transpose(ctx, kernel):
    a = ctx.base_fp16((64, 64), 0.25, 1.0)
    out = torch.zeros((64, 64), device=ctx.device, dtype=torch.float16)
    kernel(a, out)
    ctx.synchronize()
    return ctx.snippet("transpose", {"a": a}, {"out": out})


DOC_OUTPUT_CASES = {"transpose": _doc_transpose}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    test_getval_setval()
    test_tensor_getval_setval()
    test_transpose()
    test_ptr_make_tensor_getval_setval()
    logging.info("\nAll transpose/getval/setval examples passed!")
