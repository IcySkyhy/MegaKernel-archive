# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Scalar comparison and element-wise selection.

Doc: docs/zh/pypto_pro/api/SIMD-API/计算API/Memory矢量计算/{比较, 选择}

Pure vector mode with make_tile_group and auto_mutex. Verifies gt producing
a bit-packed UINT8 predicate and select choosing lhs/rhs per element.
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
# gt + select —— 按 mask 选择：out[i,j] = a[i,j] if mask[i,j] > 0 else b[i,j]
#   gt 把 mask_fp16 与 0.0 比较生成 bit-packed 谓词，select 据此选 a/b。
# ===========================================================================
@pl.jit(auto_mutex=True)
def scalar_gt_select_kernel(
    a: pl.Tensor[[64, 128], pl.DT_FP32],
    b: pl.Tensor[[64, 128], pl.DT_FP32],
    mask_in: pl.Tensor[[64, 128], pl.DT_FP16],
    out: pl.Tensor[[64, 128], pl.DT_FP32],
):
    tt32 = pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a_group = pl.make_tile_group(type=tt32, addrs=0x0000, mutex_ids=[0])
    tile_b_group = pl.make_tile_group(type=tt32, addrs=0x8000, mutex_ids=[1])
    tile_out_group = pl.make_tile_group(type=tt32, addrs=0x10000, mutex_ids=[2])
    tmp_vec_group = pl.make_tile_group(type=tt32, addrs=0x18000, mutex_ids=[3])
    mask_fp16_group = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x20000,
        mutex_ids=[4],
    )
    mask_vec_group = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_UINT8, target_memory=pl.MemorySpace.Vec),
        addrs=0x24000,
        mutex_ids=[5],
    )
    with pl.section_vector():
        tile_a = tile_a_group.current()
        tile_b = tile_b_group.current()
        tile_out = tile_out_group.current()
        tmp_vec = tmp_vec_group.current()
        mask_fp16 = mask_fp16_group.current()
        mask_vec = mask_vec_group.current()
        pl.load(tile_a, a, [0, 0])
        pl.load(tile_b, b, [0, 0])
        pl.load(mask_fp16, mask_in, [0, 0])
        # mask_fp16 > 0 -> bit-packed 谓词 mask_vec（cmp_mode=4 为 gt）
        pl.gt(mask_vec, mask_fp16, 0.0)
        # 谓词为真取 lhs(=a)，否则取 rhs(=b)
        pl.select(tile_out, mask_vec, tile_a, tile_b, tmp_vec)
        pl.store(out, tile_out, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_scalar_gt_select():
    device = ST_DEVICE
    _require_a5(device)
    torch.manual_seed(0)
    a = torch.randn(64, 128, device=device, dtype=torch.float32)
    b = torch.randn(64, 128, device=device, dtype=torch.float32)
    # mask: 一半正一半负，确保两个分支都覆盖
    mask_in = torch.randn(64, 128, device=device, dtype=torch.float16)
    out = torch.zeros(64, 128, device=device, dtype=torch.float32)
    scalar_gt_select_kernel(a, b, mask_in, out)
    torch.npu.synchronize()
    # 【推导】mask>0 取 a，否则取 b
    cond = mask_in.float() > 0
    out_ref = torch.where(cond, a, b)
    torch.testing.assert_close(out, out_ref, rtol=1e-2, atol=1e-2)
    logging.info("scalar gt+select result equal!")


def _doc_scalar_select(ctx, kernel):
    a = ctx.base_fp32((64, 128), 0.25, 1.0)
    b = ctx.base_fp32((64, 128), -0.125, 8.0)
    mask = torch.ones((64, 128), device=ctx.device, dtype=torch.float16)
    mask[:, 1::2] = -1
    out = torch.zeros((64, 128), device=ctx.device)
    kernel(a, b, mask, out)
    ctx.synchronize()
    return ctx.snippet("gt (Tile-Scalar) + select", {"a": a, "b": b, "mask": mask}, {"out": out})


DOC_OUTPUT_CASES = {
    "comparison_scalar": _doc_scalar_select,
    "select": _doc_scalar_select,
}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    test_scalar_gt_select()
    logging.info("\nAll comparison/select examples passed!")
