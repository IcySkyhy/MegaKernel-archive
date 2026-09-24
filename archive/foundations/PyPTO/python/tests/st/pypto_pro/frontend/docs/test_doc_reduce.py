# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""UT for doc examples — reduction ops.

Uses make_tile_group + auto_mutex pattern.

Verifies kernel examples from:
  docs/zh/pypto_pro/api/SIMD-API/operation/memory_vector_computation/elementwise/maximum.md
  docs/zh/pypto_pro/api/SIMD-API/operation/memory_vector_computation/math_functions/sum.md
  docs/zh/pypto_pro/api/SIMD-API/operation/memory_vector_computation/math_functions/argmax.md
  docs/zh/pypto_pro/api/SIMD-API/operation/memory_vector_computation/math_functions/argmin.md
  docs/zh/pypto_pro/api/SIMD-API/operation/memory_vector_computation/math_functions/expand_max.md
  docs/zh/pypto_pro/api/SIMD-API/operation/memory_vector_computation/math_functions/expand_min.md
"""

import logging
import os

import pypto_pro.language as pl
import pytest
import torch

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

M = 64
N = 128


# ============================================================================
# row_reduce
# ============================================================================


@pl.jit(auto_mutex=True)
def row_reduce_sum_kernel(
    x: pl.Tensor[[M, N], pl.DT_FP16],
    z: pl.Tensor[[M, 1], pl.DT_FP16],
):
    tile_a = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids=[0],
    )
    tile_tmp = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x4000,
        mutex_ids=[1],
    )
    tile_out = pl.make_tile_group(
        type=pl.TileType(shape=[64, 1], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec), addrs=0x8000, mutex_ids=[2]
    )
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_tmp = tile_tmp.current()
        cur_out = tile_out.current()
        pl.load(cur_a, x, [0, 0])
        pl.sum(cur_out, cur_a, cur_tmp, dim=0)
        pl.store(z, cur_out, [0, 0])


@pl.jit(auto_mutex=True)
def row_reduce_max_kernel(
    x: pl.Tensor[[M, N], pl.DT_FP16],
    z: pl.Tensor[[M, 1], pl.DT_FP16],
):
    tile_a = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids=[0],
    )
    tile_tmp = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x4000,
        mutex_ids=[1],
    )
    tile_out = pl.make_tile_group(
        type=pl.TileType(shape=[64, 1], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec), addrs=0x8000, mutex_ids=[2]
    )
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_tmp = tile_tmp.current()
        cur_out = tile_out.current()
        pl.load(cur_a, x, [0, 0])
        pl.maximum(cur_out, cur_a, cur_tmp, dim=0)
        pl.store(z, cur_out, [0, 0])


# ============================================================================
# row_argmax / row_argmin
# ============================================================================


@pl.jit(auto_mutex=True)
def row_argmax_kernel(
    x: pl.Tensor[[M, N], pl.DT_FP16],
    z: pl.Tensor[[M, 1], pl.DT_INT32],
):
    tile_a = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids=[0],
    )
    tile_tmp = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x4000,
        mutex_ids=[1],
    )
    tile_out = pl.make_tile_group(
        type=pl.TileType(shape=[64, 1], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x8000,
        mutex_ids=[2],
    )
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_tmp = tile_tmp.current()
        cur_out = tile_out.current()
        pl.load(cur_a, x, [0, 0])
        pl.argmax(cur_out, cur_a, cur_tmp)
        pl.store(z, cur_out, [0, 0])


@pl.jit(auto_mutex=True)
def row_argmin_kernel(
    x: pl.Tensor[[M, N], pl.DT_FP16],
    z: pl.Tensor[[M, 1], pl.DT_INT32],
):
    tile_a = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids=[0],
    )
    tile_tmp = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x4000,
        mutex_ids=[1],
    )
    tile_out = pl.make_tile_group(
        type=pl.TileType(shape=[64, 1], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x8000,
        mutex_ids=[2],
    )
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_tmp = tile_tmp.current()
        cur_out = tile_out.current()
        pl.load(cur_a, x, [0, 0])
        pl.argmin(cur_out, cur_a, cur_tmp)
        pl.store(z, cur_out, [0, 0])


# ============================================================================
# row_expand_max / row_expand_min
# ============================================================================


@pl.jit(auto_mutex=True)
def row_expand_max_kernel(
    x: pl.Tensor[[M, N], pl.DT_FP16],
    y: pl.Tensor[[M, 1], pl.DT_FP16],
    z: pl.Tensor[[M, N], pl.DT_FP16],
):
    tile_a = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids=[0],
    )
    tile_row = pl.make_tile_group(
        type=pl.TileType(shape=[64, 1], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec), addrs=0x4000, mutex_ids=[1]
    )
    tile_out = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x8000,
        mutex_ids=[2],
    )
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_row = tile_row.current()
        cur_out = tile_out.current()
        pl.load(cur_a, x, [0, 0])
        pl.load(cur_row, y, [0, 0])
        pl.expand_max(cur_out, cur_a, cur_row)
        pl.store(z, cur_out, [0, 0])


@pl.jit(auto_mutex=True)
def row_expand_min_kernel(
    x: pl.Tensor[[M, N], pl.DT_FP16],
    y: pl.Tensor[[M, 1], pl.DT_FP16],
    z: pl.Tensor[[M, N], pl.DT_FP16],
):
    tile_a = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids=[0],
    )
    tile_row = pl.make_tile_group(
        type=pl.TileType(shape=[64, 1], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec), addrs=0x4000, mutex_ids=[1]
    )
    tile_out = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x8000,
        mutex_ids=[2],
    )
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_row = tile_row.current()
        cur_out = tile_out.current()
        pl.load(cur_a, x, [0, 0])
        pl.load(cur_row, y, [0, 0])
        pl.expand_min(cur_out, cur_a, cur_row)
        pl.store(z, cur_out, [0, 0])


# ============================================================================
# col_reduce
# ============================================================================


@pl.jit(auto_mutex=True)
def col_reduce_sum_kernel(
    x: pl.Tensor[[M, N], pl.DT_FP16],
    z: pl.Tensor[[1, N], pl.DT_FP16],
):
    tile_a = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids=[0],
    )
    tile_tmp = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x4000,
        mutex_ids=[1],
    )
    tile_out = pl.make_tile_group(
        type=pl.TileType(shape=[1, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x8000,
        mutex_ids=[2],
    )
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_tmp = tile_tmp.current()
        cur_out = tile_out.current()
        pl.load(cur_a, x, [0, 0])
        pl.sum(cur_out, cur_a, cur_tmp, dim=1)
        pl.store(z, cur_out, [0, 0])


# ============================================================================
# col_argmax / col_argmin
# ============================================================================


@pl.jit(auto_mutex=True)
def col_argmax_kernel(
    x: pl.Tensor[[M, N], pl.DT_FP16],
    z: pl.Tensor[[1, N], pl.DT_INT32],
):
    tile_a = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids=[0],
    )
    tile_tmp = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x4000,
        mutex_ids=[1],
    )
    tile_out = pl.make_tile_group(
        type=pl.TileType(shape=[1, 128], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x8000,
        mutex_ids=[2],
    )
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_tmp = tile_tmp.current()
        cur_out = tile_out.current()
        pl.load(cur_a, x, [0, 0])
        pl.argmax(cur_out, cur_a, cur_tmp, dim=1)
        pl.store(z, cur_out, [0, 0])


@pl.jit(auto_mutex=True)
def col_argmin_kernel(
    x: pl.Tensor[[M, N], pl.DT_FP16],
    z: pl.Tensor[[1, N], pl.DT_INT32],
):
    tile_a = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids=[0],
    )
    tile_tmp = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x4000,
        mutex_ids=[1],
    )
    tile_out = pl.make_tile_group(
        type=pl.TileType(shape=[1, 128], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x8000,
        mutex_ids=[2],
    )
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_tmp = tile_tmp.current()
        cur_out = tile_out.current()
        pl.load(cur_a, x, [0, 0])
        pl.argmin(cur_out, cur_a, cur_tmp, dim=1)
        pl.store(z, cur_out, [0, 0])


# ============================================================================
# col_expand_max / col_expand_min
# ============================================================================


@pl.jit(auto_mutex=True)
def col_expand_max_kernel(
    x: pl.Tensor[[M, N], pl.DT_FP16],
    y: pl.Tensor[[1, N], pl.DT_FP16],
    z: pl.Tensor[[M, N], pl.DT_FP16],
):
    tile_a = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids=[0],
    )
    tile_col = pl.make_tile_group(
        type=pl.TileType(shape=[1, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x4000,
        mutex_ids=[1],
    )
    tile_out = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x8000,
        mutex_ids=[2],
    )
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_col = tile_col.current()
        cur_out = tile_out.current()
        pl.load(cur_a, x, [0, 0])
        pl.load(cur_col, y, [0, 0])
        pl.expand_max(cur_out, cur_a, cur_col, dim=1)
        pl.store(z, cur_out, [0, 0])


@pl.jit(auto_mutex=True)
def col_expand_min_kernel(
    x: pl.Tensor[[M, N], pl.DT_FP16],
    y: pl.Tensor[[1, N], pl.DT_FP16],
    z: pl.Tensor[[M, N], pl.DT_FP16],
):
    tile_a = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids=[0],
    )
    tile_col = pl.make_tile_group(
        type=pl.TileType(shape=[1, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x4000,
        mutex_ids=[1],
    )
    tile_out = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x8000,
        mutex_ids=[2],
    )
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_col = tile_col.current()
        cur_out = tile_out.current()
        pl.load(cur_a, x, [0, 0])
        pl.load(cur_col, y, [0, 0])
        pl.expand_min(cur_out, cur_a, cur_col, dim=1)
        pl.store(z, cur_out, [0, 0])


# ============================================================================
# Test functions
# ============================================================================


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_row_reduce_sum():
    torch.npu.set_device(ST_DEVICE)
    x = torch.randn(M, N, device=ST_DEVICE, dtype=torch.float16)
    z = torch.empty(M, 1, device=ST_DEVICE, dtype=torch.float16)
    z_ref = x.sum(dim=1, keepdim=True)
    row_reduce_sum_kernel(x, z)
    torch.npu.synchronize()
    torch.testing.assert_close(z, z_ref, atol=1e-1, rtol=1e-1)
    logging.info("test_row_reduce_sum passed!")


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_row_reduce_max():
    torch.npu.set_device(ST_DEVICE)
    x = torch.randn(M, N, device=ST_DEVICE, dtype=torch.float16)
    z = torch.empty(M, 1, device=ST_DEVICE, dtype=torch.float16)
    z_ref = x.amax(dim=1, keepdim=True)
    row_reduce_max_kernel(x, z)
    torch.npu.synchronize()
    torch.testing.assert_close(z, z_ref, atol=1e-2, rtol=1e-2)
    logging.info("test_row_reduce_max passed!")


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_row_argmax():
    torch.npu.set_device(ST_DEVICE)
    x = torch.randn(M, N, device=ST_DEVICE, dtype=torch.float16)
    z = torch.empty(M, 1, device=ST_DEVICE, dtype=torch.int32)
    z_ref = x.argmax(dim=1, keepdim=True).to(torch.int32)
    row_argmax_kernel(x, z)
    torch.npu.synchronize()
    assert torch.equal(z, z_ref), "row_argmax mismatch"
    logging.info("test_row_argmax passed!")


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_row_argmin():
    torch.npu.set_device(ST_DEVICE)
    x = torch.randn(M, N, device=ST_DEVICE, dtype=torch.float16)
    z = torch.empty(M, 1, device=ST_DEVICE, dtype=torch.int32)
    z_ref = x.argmin(dim=1, keepdim=True).to(torch.int32)
    row_argmin_kernel(x, z)
    torch.npu.synchronize()
    assert torch.equal(z, z_ref), "row_argmin mismatch"
    logging.info("test_row_argmin passed!")


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_row_expand_max():
    torch.npu.set_device(ST_DEVICE)
    x = torch.randn(M, N, device=ST_DEVICE, dtype=torch.float16)
    y = torch.randn(M, 1, device=ST_DEVICE, dtype=torch.float16)
    z = torch.empty(M, N, device=ST_DEVICE, dtype=torch.float16)
    z_ref = torch.maximum(x, y)
    row_expand_max_kernel(x, y, z)
    torch.npu.synchronize()
    torch.testing.assert_close(z, z_ref, atol=1e-2, rtol=1e-2)
    logging.info("test_row_expand_max passed!")


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_row_expand_min():
    torch.npu.set_device(ST_DEVICE)
    x = torch.randn(M, N, device=ST_DEVICE, dtype=torch.float16)
    y = torch.randn(M, 1, device=ST_DEVICE, dtype=torch.float16)
    z = torch.empty(M, N, device=ST_DEVICE, dtype=torch.float16)
    z_ref = torch.minimum(x, y)
    row_expand_min_kernel(x, y, z)
    torch.npu.synchronize()
    torch.testing.assert_close(z, z_ref, atol=1e-2, rtol=1e-2)
    logging.info("test_row_expand_min passed!")


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_col_reduce_sum():
    torch.npu.set_device(ST_DEVICE)
    x = torch.randn(M, N, device=ST_DEVICE, dtype=torch.float16)
    z = torch.empty(1, N, device=ST_DEVICE, dtype=torch.float16)
    z_ref = x.sum(dim=0, keepdim=True)
    col_reduce_sum_kernel(x, z)
    torch.npu.synchronize()
    torch.testing.assert_close(z, z_ref, atol=1e-1, rtol=1e-1)
    logging.info("test_col_reduce_sum passed!")


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_col_argmax():
    torch.npu.set_device(ST_DEVICE)
    x = torch.randn(M, N, device=ST_DEVICE, dtype=torch.float16)
    z = torch.empty(1, N, device=ST_DEVICE, dtype=torch.int32)
    z_ref = x.argmax(dim=0, keepdim=True).to(torch.int32)
    col_argmax_kernel(x, z)
    torch.npu.synchronize()
    assert torch.equal(z, z_ref), "col_argmax mismatch"
    logging.info("test_col_argmax passed!")


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_col_argmin():
    torch.npu.set_device(ST_DEVICE)
    x = torch.randn(M, N, device=ST_DEVICE, dtype=torch.float16)
    z = torch.empty(1, N, device=ST_DEVICE, dtype=torch.int32)
    z_ref = x.argmin(dim=0, keepdim=True).to(torch.int32)
    col_argmin_kernel(x, z)
    torch.npu.synchronize()
    assert torch.equal(z, z_ref), "col_argmin mismatch"
    logging.info("test_col_argmin passed!")


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_col_expand_max():
    torch.npu.set_device(ST_DEVICE)
    x = torch.randn(M, N, device=ST_DEVICE, dtype=torch.float16)
    y = torch.randn(1, N, device=ST_DEVICE, dtype=torch.float16)
    z = torch.empty(M, N, device=ST_DEVICE, dtype=torch.float16)
    z_ref = torch.maximum(x, y)
    col_expand_max_kernel(x, y, z)
    torch.npu.synchronize()
    torch.testing.assert_close(z, z_ref, atol=1e-2, rtol=1e-2)
    logging.info("test_col_expand_max passed!")


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_col_expand_min():
    torch.npu.set_device(ST_DEVICE)
    x = torch.randn(M, N, device=ST_DEVICE, dtype=torch.float16)
    y = torch.randn(1, N, device=ST_DEVICE, dtype=torch.float16)
    z = torch.empty(M, N, device=ST_DEVICE, dtype=torch.float16)
    z_ref = torch.minimum(x, y)
    col_expand_min_kernel(x, y, z)
    torch.npu.synchronize()
    torch.testing.assert_close(z, z_ref, atol=1e-2, rtol=1e-2)
    logging.info("test_col_expand_min passed!")


def _doc_reduce(title, out_shape, output_dtype):
    def collect(ctx, kernel):
        a = ctx.base_fp32((M, N), 0.25, -8.0).to(torch.float16)
        z = torch.zeros(out_shape, device=ctx.device, dtype=output_dtype(torch))
        kernel(a, z)
        ctx.synchronize()
        return ctx.snippet(title, {"a": a}, {"z": z})

    return collect


def _doc_expand(title, v_shape):
    def collect(ctx, kernel):
        a = ctx.base_fp32((M, N), 0.125, -3.0).to(torch.float16)
        v = (ctx.base_fp32(v_shape, 0.25, 0.75).abs() + 0.5).to(torch.float16)
        z = torch.zeros((M, N), device=ctx.device, dtype=torch.float16)
        kernel(a, v, z)
        ctx.synchronize()
        return ctx.snippet(title, {"a": a, "v": v}, {"z": z})

    return collect


DOC_OUTPUT_CASES = {
    "row_argmax": _doc_reduce("argmax (dim=0)", (M, 1), lambda torch: torch.int32),
    "row_argmin": _doc_reduce("argmin (dim=0)", (M, 1), lambda torch: torch.int32),
    "col_argmax": _doc_reduce("argmax (dim=1)", (1, N), lambda torch: torch.int32),
    "col_argmin": _doc_reduce("argmin (dim=1)", (1, N), lambda torch: torch.int32),
    "col_reduce_sum": _doc_reduce("sum (dim=1, FP16)", (1, N), lambda torch: torch.float16),
    "row_expand_max": _doc_expand("expand_max (dim=0)", (M, 1)),
    "row_expand_min": _doc_expand("expand_min (dim=0)", (M, 1)),
    "col_expand_max": _doc_expand("expand_max (dim=1)", (1, N)),
    "col_expand_min": _doc_expand("expand_min (dim=1)", (1, N)),
}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cases = [
        test_row_reduce_sum,
        test_row_reduce_max,
        test_row_argmax,
        test_row_argmin,
        test_row_expand_max,
        test_row_expand_min,
        test_col_reduce_sum,
        test_col_argmax,
        test_col_argmin,
        test_col_expand_max,
        test_col_expand_min,
    ]
    for case in cases:
        case()
    logging.info("\nAll reduction tests passed!")
