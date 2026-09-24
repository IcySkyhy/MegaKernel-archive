# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Math functions: row/col reduce, argmax/argmin, expand variants.

Doc: docs/zh/pypto_pro/api/SIMD-API/计算API/数学函数

Vector mode with make_tile_group + auto_mutex (single buffer). Verifies reduction ops
(col_*/row_*) with tile_tmp and expand ops (max/min/add/sub/mul/div).
"""

import logging
import os

import pypto_pro.language as pl
import pytest
import torch

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

K_VALUE = 2.0


def _require_a5(device):
    try:
        torch.npu.set_device(device)
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
    name = torch.npu.get_device_name()
    if "Ascend950" not in name:
        pytest.skip(f"Current device is {name}, not A5 (Ascend950). Skip.")


# ===========================================================================
# 列向归约 col_*：out[0,j] = reduce_i(a[i,j])，输出 [1,N]，需 tile_tmp
# ===========================================================================
@pl.jit(auto_mutex=True)
def col_min_kernel(a: pl.Tensor[[64, 128], pl.DT_FP32], out: pl.Tensor[[1, 128], pl.DT_FP32]):
    tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tt_out = pl.TileType(shape=[1, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_tmp = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=tt_out, addrs=0x10000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_tmp = tile_tmp.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.minimum(cur_out, cur_a, cur_tmp, dim=1)
        pl.store(out, cur_out, [0, 0])


@pl.jit(auto_mutex=True)
def col_max_kernel(a: pl.Tensor[[64, 128], pl.DT_FP32], out: pl.Tensor[[1, 128], pl.DT_FP32]):
    tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tt_out = pl.TileType(shape=[1, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_tmp = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=tt_out, addrs=0x10000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_tmp = tile_tmp.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.maximum(cur_out, cur_a, cur_tmp, dim=1)
        pl.store(out, cur_out, [0, 0])


@pl.jit(auto_mutex=True)
def col_sum_kernel(a: pl.Tensor[[64, 128], pl.DT_FP32], out: pl.Tensor[[1, 128], pl.DT_FP32]):
    tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tt_out = pl.TileType(shape=[1, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_tmp = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=tt_out, addrs=0x10000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_tmp = tile_tmp.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.sum(cur_out, cur_a, cur_tmp, dim=1)
        pl.store(out, cur_out, [0, 0])


def _run_col_reduce(kernel, ref_fn):
    device = ST_DEVICE
    _require_a5(device)
    torch.manual_seed(0)
    a = torch.randn(64, 128, device=device, dtype=torch.float32)
    out = torch.zeros(1, 128, device=device, dtype=torch.float32)
    kernel(a, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, ref_fn(a), rtol=1e-2, atol=1e-2)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_col_min():
    _run_col_reduce(col_min_kernel, lambda a: a.min(dim=0, keepdim=True).values)
    logging.info("col_min result equal!")


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_col_max():
    _run_col_reduce(col_max_kernel, lambda a: a.max(dim=0, keepdim=True).values)
    logging.info("col_max result equal!")


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_col_sum():
    _run_col_reduce(col_sum_kernel, lambda a: a.sum(dim=0, keepdim=True))
    logging.info("col_sum result equal!")


# ===========================================================================
# 行向归约 row_*：out[i,0] = reduce_j(a[i,j])，输出 [M,1]（输出 tile layout=pl.DN）
# ===========================================================================
@pl.jit(auto_mutex=True)
def row_max_kernel(a: pl.Tensor[[64, 128], pl.DT_FP32], out: pl.Tensor[[64, 1], pl.DT_FP32]):
    tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tt_out = pl.TileType(shape=[64, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.DN)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_tmp = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=tt_out, addrs=0x10000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_tmp = tile_tmp.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.maximum(cur_out, cur_a, cur_tmp, dim=0)
        pl.store(out, cur_out, [0, 0])


@pl.jit(auto_mutex=True)
def row_sum_kernel(a: pl.Tensor[[64, 128], pl.DT_FP32], out: pl.Tensor[[64, 1], pl.DT_FP32]):
    tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tt_out = pl.TileType(shape=[64, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.DN)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_tmp = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=tt_out, addrs=0x10000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_tmp = tile_tmp.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.sum(cur_out, cur_a, cur_tmp)
        pl.store(out, cur_out, [0, 0])


def _run_row_reduce(kernel, ref_fn):
    device = ST_DEVICE
    _require_a5(device)
    torch.manual_seed(0)
    a = torch.randn(64, 128, device=device, dtype=torch.float32)
    out = torch.zeros(64, 1, device=device, dtype=torch.float32)
    kernel(a, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, ref_fn(a), rtol=1e-2, atol=1e-2)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_row_max():
    _run_row_reduce(row_max_kernel, lambda a: a.max(dim=1, keepdim=True).values)
    logging.info("row_max result equal!")


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_row_sum():
    _run_row_reduce(row_sum_kernel, lambda a: a.sum(dim=1, keepdim=True))
    logging.info("row_sum result equal!")


# ===========================================================================
# 列广播 col_expand_*：用 [1,N] 向量沿列广播，与 a 逐元素运算，输出 [M,N]
# ===========================================================================
@pl.jit(auto_mutex=True)
def col_expand_sub_kernel(
    a: pl.Tensor[[64, 128], pl.DT_FP32], v: pl.Tensor[[1, 128], pl.DT_FP32], out: pl.Tensor[[64, 128], pl.DT_FP32]
):
    tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tt_v = pl.TileType(shape=[1, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_v = pl.make_tile_group(type=tt_v, addrs=0x8000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=tt, addrs=0x10000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_v = tile_v.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_v, v, [0, 0])
        pl.expand_sub(cur_out, cur_a, cur_v, dim=1)
        pl.store(out, cur_out, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_col_expand_sub():
    device = ST_DEVICE
    _require_a5(device)
    torch.manual_seed(0)
    a = torch.randn(64, 128, device=device, dtype=torch.float32)
    v = torch.randn(1, 128, device=device, dtype=torch.float32)
    out = torch.zeros(64, 128, device=device, dtype=torch.float32)
    col_expand_sub_kernel(a, v, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a - v, rtol=1e-2, atol=1e-2)
    logging.info("col_expand_sub result equal!")


# ===========================================================================
# 行广播 row_expand_*：用 [M,1] 向量沿行广播，与 a 逐元素运算，输出 [M,N]
# ===========================================================================
@pl.jit(auto_mutex=True)
def row_expand_sub_kernel(
    a: pl.Tensor[[64, 128], pl.DT_FP32], v: pl.Tensor[[64, 1], pl.DT_FP32], out: pl.Tensor[[64, 128], pl.DT_FP32]
):
    tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tt_v = pl.TileType(shape=[64, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.DN)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_v = pl.make_tile_group(type=tt_v, addrs=0x8000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=tt, addrs=0x10000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_v = tile_v.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_v, v, [0, 0])
        pl.expand_sub(cur_out, cur_a, cur_v)
        pl.store(out, cur_out, [0, 0])


@pl.jit(auto_mutex=True)
def row_expand_mul_kernel(
    a: pl.Tensor[[64, 128], pl.DT_FP32], v: pl.Tensor[[64, 1], pl.DT_FP32], out: pl.Tensor[[64, 128], pl.DT_FP32]
):
    tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tt_v = pl.TileType(shape=[64, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.DN)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_v = pl.make_tile_group(type=tt_v, addrs=0x8000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=tt, addrs=0x10000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_v = tile_v.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_v, v, [0, 0])
        pl.expand_mul(cur_out, cur_a, cur_v)
        pl.store(out, cur_out, [0, 0])


@pl.jit(auto_mutex=True)
def row_expand_div_kernel(
    a: pl.Tensor[[64, 128], pl.DT_FP32], v: pl.Tensor[[64, 1], pl.DT_FP32], out: pl.Tensor[[64, 128], pl.DT_FP32]
):
    tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tt_v = pl.TileType(shape=[64, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.DN)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_v = pl.make_tile_group(type=tt_v, addrs=0x8000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=tt, addrs=0x10000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_v = tile_v.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_v, v, [0, 0])
        pl.expand_div(cur_out, cur_a, cur_v)
        pl.store(out, cur_out, [0, 0])


def _run_row_expand(kernel, ref_fn):
    device = ST_DEVICE
    _require_a5(device)
    torch.manual_seed(0)
    a = torch.randn(64, 128, device=device, dtype=torch.float32)
    v = torch.randn(64, 1, device=device, dtype=torch.float32).abs() + 0.5  # 避免除0
    out = torch.zeros(64, 128, device=device, dtype=torch.float32)
    kernel(a, v, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, ref_fn(a, v), rtol=1e-2, atol=1e-2)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_row_expand_sub():
    _run_row_expand(row_expand_sub_kernel, lambda a, v: a - v)
    logging.info("row_expand_sub result equal!")


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_row_expand_mul():
    _run_row_expand(row_expand_mul_kernel, lambda a, v: a * v)
    logging.info("row_expand_mul result equal!")


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_row_expand_div():
    _run_row_expand(row_expand_div_kernel, lambda a, v: a / v)
    logging.info("row_expand_div result equal!")


# ===========================================================================
# expands —— 把标量广播填入整块 tile：out[i,j] = K_VALUE
# ===========================================================================
@pl.jit(auto_mutex=True)
def expands_kernel(dummy: pl.Tensor[[64, 64], pl.DT_FP32], out: pl.Tensor[[64, 64], pl.DT_FP32]):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_out = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    with pl.section_vector():
        cur_out = tile_out.current()
        pl.expands(cur_out, K_VALUE)
        pl.store(out, cur_out, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_expands():
    device = ST_DEVICE
    _require_a5(device)
    torch.manual_seed(0)
    dummy = torch.zeros(64, 64, device=device, dtype=torch.float32)
    out = torch.zeros(64, 64, device=device, dtype=torch.float32)
    expands_kernel(dummy, out)
    torch.npu.synchronize()
    ref = torch.full((64, 64), K_VALUE, device=device, dtype=torch.float32)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
    logging.info("expands result equal!")


# ===========================================================================
# 行向归约 row_min：out[i,0] = min_j(a[i,j])，输出 [M,1]（输出 tile layout=pl.DN）
# ===========================================================================
@pl.jit(auto_mutex=True)
def row_min_kernel(a: pl.Tensor[[64, 128], pl.DT_FP32], out: pl.Tensor[[64, 1], pl.DT_FP32]):
    tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tt_out = pl.TileType(shape=[64, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.DN)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_tmp = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=tt_out, addrs=0x10000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_tmp = tile_tmp.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.minimum(cur_out, cur_a, cur_tmp, dim=0)
        pl.store(out, cur_out, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_row_min():
    _run_row_reduce(row_min_kernel, lambda a: a.min(dim=1, keepdim=True).values)
    logging.info("row_min result equal!")


# ===========================================================================
# 列广播 col_expand_mul / col_expand_div
# ===========================================================================
@pl.jit(auto_mutex=True)
def col_expand_mul_kernel(
    a: pl.Tensor[[64, 128], pl.DT_FP32], v: pl.Tensor[[1, 128], pl.DT_FP32], out: pl.Tensor[[64, 128], pl.DT_FP32]
):
    tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tt_v = pl.TileType(shape=[1, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_v = pl.make_tile_group(type=tt_v, addrs=0x8000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=tt, addrs=0x10000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_v = tile_v.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_v, v, [0, 0])
        pl.expand_mul(cur_out, cur_a, cur_v, dim=1)
        pl.store(out, cur_out, [0, 0])


@pl.jit(auto_mutex=True)
def col_expand_div_kernel(
    a: pl.Tensor[[64, 128], pl.DT_FP32], v: pl.Tensor[[1, 128], pl.DT_FP32], out: pl.Tensor[[64, 128], pl.DT_FP32]
):
    tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tt_v = pl.TileType(shape=[1, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_v = pl.make_tile_group(type=tt_v, addrs=0x8000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=tt, addrs=0x10000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_v = tile_v.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_v, v, [0, 0])
        pl.expand_div(cur_out, cur_a, cur_v, dim=1)
        pl.store(out, cur_out, [0, 0])


def _run_col_expand(kernel, ref_fn):
    device = ST_DEVICE
    _require_a5(device)
    torch.manual_seed(0)
    a = torch.randn(64, 128, device=device, dtype=torch.float32)
    v = torch.randn(1, 128, device=device, dtype=torch.float32).abs() + 0.5  # 避免除0
    out = torch.zeros(64, 128, device=device, dtype=torch.float32)
    kernel(a, v, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, ref_fn(a, v), rtol=1e-2, atol=1e-2)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_col_expand_mul():
    _run_col_expand(col_expand_mul_kernel, lambda a, v: a * v)
    logging.info("col_expand_mul result equal!")


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_col_expand_div():
    _run_col_expand(col_expand_div_kernel, lambda a, v: a / v)
    logging.info("col_expand_div result equal!")


def _doc_reduce(title, out_shape):
    def collect(ctx, kernel):
        a = ctx.base_fp32((64, 128), 0.25, -8.0)
        out = torch.zeros(out_shape, device=ctx.device, dtype=torch.float32)
        kernel(a, out)
        ctx.synchronize()
        return ctx.snippet(title, {"a": a}, {"out": out})

    return collect


def _doc_expand(title, v_shape):
    def collect(ctx, kernel):
        a = ctx.base_fp32((64, 128), 0.125, -3.0)
        v = ctx.base_fp32(v_shape, 0.25, 0.75).abs() + 0.5
        out = torch.zeros((64, 128), device=ctx.device, dtype=torch.float32)
        kernel(a, v, out)
        ctx.synchronize()
        return ctx.snippet(title, {"a": a, "v": v}, {"out": out})

    return collect


def _doc_expands(ctx, kernel):
    dummy = torch.zeros((64, 64), device=ctx.device, dtype=torch.float32)
    out = torch.zeros((64, 64), device=ctx.device, dtype=torch.float32)
    kernel(dummy, out)
    ctx.synchronize()
    return ctx.snippet("expands", {"value": torch.tensor(K_VALUE)}, {"out": out})


DOC_OUTPUT_CASES = {
    "row_min": _doc_reduce("minimum (dim=0)", (64, 1)),
    "row_max": _doc_reduce("maximum (dim=0)", (64, 1)),
    "row_sum": _doc_reduce("sum (dim=0)", (64, 1)),
    "col_min": _doc_reduce("minimum (dim=1)", (1, 128)),
    "col_max": _doc_reduce("maximum (dim=1)", (1, 128)),
    "col_sum": _doc_reduce("sum (dim=1)", (1, 128)),
    "row_expand_sub": _doc_expand("expand_sub (dim=0)", (64, 1)),
    "row_expand_mul": _doc_expand("expand_mul (dim=0)", (64, 1)),
    "row_expand_div": _doc_expand("expand_div (dim=0)", (64, 1)),
    "col_expand_sub": _doc_expand("expand_sub (dim=1)", (1, 128)),
    "col_expand_mul": _doc_expand("expand_mul (dim=1)", (1, 128)),
    "col_expand_div": _doc_expand("expand_div (dim=1)", (1, 128)),
    "expands": _doc_expands,
}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for fn in [
        test_col_min,
        test_col_max,
        test_col_sum,
        test_row_max,
        test_row_sum,
        test_row_min,
        test_col_expand_sub,
        test_col_expand_mul,
        test_col_expand_div,
        test_row_expand_sub,
        test_row_expand_mul,
        test_row_expand_div,
        test_expands,
    ]:
        fn()
    logging.info("\nAll batch-2 math-function examples passed!")
