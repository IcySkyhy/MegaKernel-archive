# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Elementwise, composite, and fused vector ops.

Doc: docs/zh/pypto_pro/api/SIMD-API/计算API/Memory矢量计算/{逐元素, 复合计算, 融合算子, 类型转换}

Pure vector mode with make_tile_group + auto_mutex style (single-buffer, no manual sync).
Verifies add/sub/mul/div/xor, maximum/minimum, neg/relu, and fused ops (axpy/mul_add/etc).
"""

import logging
import os

import pypto_pro.language as pl
import pytest
import torch

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

SCALE = 0.125
SUB_VALUE = 1.0
DIVISOR = 4.0
ADD_VALUE = 1.0
MAX_SCALAR = 0.0


def _require_a5(device):
    try:
        torch.npu.set_device(device)
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
    name = torch.npu.get_device_name()
    if "Ascend950" not in name:
        pytest.skip(f"Current device is {name}, not A5 (Ascend950). Skip.")


def _run_binary(kernel, ref_fn, dtype=torch.float32):
    device = ST_DEVICE
    _require_a5(device)
    torch.manual_seed(0)
    a = torch.randn(64, 64, device=device, dtype=dtype)
    b = torch.randn(64, 64, device=device, dtype=dtype)
    out = torch.zeros(64, 64, device=device, dtype=dtype)
    kernel(a, b, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, ref_fn(a, b), rtol=1e-2, atol=1e-2)


def _run_unary(kernel, ref_fn):
    device = ST_DEVICE
    _require_a5(device)
    torch.manual_seed(0)
    a = torch.randn(64, 64, device=device, dtype=torch.float32)
    out = torch.zeros(64, 64, device=device, dtype=torch.float32)
    kernel(a, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, ref_fn(a), rtol=1e-2, atol=1e-2)


def _run_positive_unary(kernel, ref_fn):
    device = ST_DEVICE
    _require_a5(device)
    torch.manual_seed(0)
    a = torch.rand(64, 64, device=device, dtype=torch.float32) + 1.0
    out = torch.zeros(64, 64, device=device, dtype=torch.float32)
    kernel(a, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, ref_fn(a), rtol=1e-2, atol=1e-2)


# ===========================================================================
# 逐元素二元：add / sub / mul / maximum / div / minimum
# ===========================================================================
@pl.jit(auto_mutex=True)
def add_kernel(
    a: pl.Tensor[[64, 64], pl.DT_FP32], b: pl.Tensor[[64, 64], pl.DT_FP32], out: pl.Tensor[[64, 64], pl.DT_FP32]
):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_b = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_b = tile_b.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.add(cur_out, cur_a, cur_b)
        pl.store(out, cur_out, [0, 0])


@pl.jit(auto_mutex=True)
def sub_kernel(
    a: pl.Tensor[[64, 64], pl.DT_FP32], b: pl.Tensor[[64, 64], pl.DT_FP32], out: pl.Tensor[[64, 64], pl.DT_FP32]
):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_b = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_b = tile_b.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.sub(cur_out, cur_a, cur_b)
        pl.store(out, cur_out, [0, 0])


@pl.jit(auto_mutex=True)
def mul_kernel(
    a: pl.Tensor[[64, 64], pl.DT_FP32], b: pl.Tensor[[64, 64], pl.DT_FP32], out: pl.Tensor[[64, 64], pl.DT_FP32]
):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_b = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_b = tile_b.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.mul(cur_out, cur_a, cur_b)
        pl.store(out, cur_out, [0, 0])


@pl.jit(auto_mutex=True)
def maximum_kernel(
    a: pl.Tensor[[64, 64], pl.DT_FP32], b: pl.Tensor[[64, 64], pl.DT_FP32], out: pl.Tensor[[64, 64], pl.DT_FP32]
):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_b = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_b = tile_b.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.maximum(cur_out, cur_a, cur_b)
        pl.store(out, cur_out, [0, 0])


@pl.jit(auto_mutex=True)
def div_kernel(
    a: pl.Tensor[[64, 64], pl.DT_FP32], b: pl.Tensor[[64, 64], pl.DT_FP32], out: pl.Tensor[[64, 64], pl.DT_FP32]
):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_b = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_b = tile_b.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.div(cur_out, cur_a, cur_b)
        pl.store(out, cur_out, [0, 0])


@pl.jit(auto_mutex=True)
def minimum_kernel(
    a: pl.Tensor[[64, 64], pl.DT_FP32], b: pl.Tensor[[64, 64], pl.DT_FP32], out: pl.Tensor[[64, 64], pl.DT_FP32]
):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_b = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_b = tile_b.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.minimum(cur_out, cur_a, cur_b)
        pl.store(out, cur_out, [0, 0])


MIN_SCALAR = 1.0


@pl.jit(auto_mutex=True)
def minimum_scalar_kernel(a: pl.Tensor[[64, 64], pl.DT_FP32], out: pl.Tensor[[64, 64], pl.DT_FP32]):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_out = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.minimum(cur_out, cur_a, MIN_SCALAR)
        pl.store(out, cur_out, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_add():
    _run_binary(add_kernel, lambda a, b: a + b)
    logging.info("add result equal!")


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_sub():
    _run_binary(sub_kernel, lambda a, b: a - b)
    logging.info("sub result equal!")


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_mul():
    _run_binary(mul_kernel, lambda a, b: a * b)
    logging.info("mul result equal!")


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_maximum():
    _run_binary(maximum_kernel, lambda a, b: torch.maximum(a, b))
    logging.info("maximum result equal!")


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_div():
    _run_binary(div_kernel, lambda a, b: a / b)
    logging.info("div result equal!")


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_minimum():
    _run_binary(minimum_kernel, lambda a, b: torch.minimum(a, b))
    logging.info("minimum result equal!")


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_minimum_scalar():
    _run_unary(minimum_scalar_kernel, lambda a: torch.minimum(a, torch.tensor(MIN_SCALAR)))
    logging.info("minimum_scalar result equal!")


# ===========================================================================
# 逐元素按位运算：xor
# ===========================================================================
@pl.jit(auto_mutex=True)
def xor_kernel(
    a: pl.Tensor[[64, 64], pl.DT_INT32],
    b: pl.Tensor[[64, 64], pl.DT_INT32],
    out: pl.Tensor[[64, 64], pl.DT_INT32],
):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_b = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    tile_tmp = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[2])
    tile_out = pl.make_tile_group(type=tt, addrs=0xC000, mutex_ids=[3])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_b = tile_b.current()
        cur_tmp = tile_tmp.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.xor(cur_out, cur_a, cur_b, cur_tmp)
        pl.store(out, cur_out, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_xor():
    device = ST_DEVICE
    _require_a5(device)
    a = torch.arange(64 * 64, device=device, dtype=torch.int32).reshape(64, 64) + 2
    b = (torch.arange(64 * 64, device=device, dtype=torch.int32).reshape(64, 64) % 16) + 1
    out = torch.zeros((64, 64), device=device, dtype=torch.int32)
    xor_kernel(a, b, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, torch.bitwise_xor(a, b), rtol=0, atol=0)
    logging.info("xor result equal!")


# ===========================================================================
# 逐元素标量：mul / sub / div (tile-scalar)
# ===========================================================================
@pl.jit(auto_mutex=True)
def mul_scalar_kernel(a: pl.Tensor[[64, 64], pl.DT_FP32], out: pl.Tensor[[64, 64], pl.DT_FP32]):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_out = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.mul(cur_out, cur_a, SCALE)
        pl.store(out, cur_out, [0, 0])


@pl.jit(auto_mutex=True)
def sub_scalar_kernel(a: pl.Tensor[[64, 64], pl.DT_FP32], out: pl.Tensor[[64, 64], pl.DT_FP32]):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_out = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.sub(cur_out, cur_a, SUB_VALUE)
        pl.store(out, cur_out, [0, 0])


@pl.jit(auto_mutex=True)
def div_scalar_kernel(a: pl.Tensor[[64, 64], pl.DT_FP32], out: pl.Tensor[[64, 64], pl.DT_FP32]):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_out = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.div(cur_out, cur_a, DIVISOR)
        pl.store(out, cur_out, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_mul_scalar():
    _run_unary(mul_scalar_kernel, lambda a: a * SCALE)
    logging.info("mul_scalar result equal!")


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_sub_scalar():
    _run_unary(sub_scalar_kernel, lambda a: a - SUB_VALUE)
    logging.info("sub_scalar result equal!")


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_div_scalar():
    _run_unary(div_scalar_kernel, lambda a: a / DIVISOR)
    logging.info("div_scalar result equal!")


@pl.jit(auto_mutex=True)
def add_scalar_kernel(a: pl.Tensor[[64, 64], pl.DT_FP32], out: pl.Tensor[[64, 64], pl.DT_FP32]):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_out = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.add(cur_out, cur_a, ADD_VALUE)
        pl.store(out, cur_out, [0, 0])


@pl.jit(auto_mutex=True)
def maximum_scalar_kernel(a: pl.Tensor[[64, 64], pl.DT_FP32], out: pl.Tensor[[64, 64], pl.DT_FP32]):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_out = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.maximum(cur_out, cur_a, MAX_SCALAR)
        pl.store(out, cur_out, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_add_scalar():
    _run_unary(add_scalar_kernel, lambda a: a + ADD_VALUE)
    logging.info("add_scalar result equal!")


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_maximum_scalar():
    _run_unary(maximum_scalar_kernel, lambda a: torch.maximum(a, torch.tensor(MAX_SCALAR)))
    logging.info("maximum_scalar result equal!")


# ===========================================================================
# 逐元素一元：exp / relu / neg
# ===========================================================================
@pl.jit(auto_mutex=True)
def exp_kernel(a: pl.Tensor[[64, 64], pl.DT_FP32], out: pl.Tensor[[64, 64], pl.DT_FP32]):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_out = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.exp(cur_out, cur_a)
        pl.store(out, cur_out, [0, 0])


@pl.jit(auto_mutex=True)
def relu_kernel(a: pl.Tensor[[64, 64], pl.DT_FP32], out: pl.Tensor[[64, 64], pl.DT_FP32]):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_out = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.relu(cur_out, cur_a)
        pl.store(out, cur_out, [0, 0])


@pl.jit(auto_mutex=True)
def neg_kernel(a: pl.Tensor[[64, 64], pl.DT_FP32], out: pl.Tensor[[64, 64], pl.DT_FP32]):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_out = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.neg(cur_out, cur_a)
        pl.store(out, cur_out, [0, 0])


@pl.jit(auto_mutex=True)
def rsqrt_kernel(a: pl.Tensor[[64, 64], pl.DT_FP32], out: pl.Tensor[[64, 64], pl.DT_FP32]):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_out = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.rsqrt(cur_out, cur_a)
        pl.store(out, cur_out, [0, 0])


@pl.jit(auto_mutex=True)
def recip_kernel(a: pl.Tensor[[64, 64], pl.DT_FP32], out: pl.Tensor[[64, 64], pl.DT_FP32]):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_out = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.recip(cur_out, cur_a)
        pl.store(out, cur_out, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_exp():
    _run_unary(exp_kernel, lambda a: torch.exp(a))
    logging.info("exp result equal!")


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_relu():
    _run_unary(relu_kernel, lambda a: torch.relu(a))
    logging.info("relu result equal!")


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_neg():
    _run_unary(neg_kernel, lambda a: -a)
    logging.info("neg result equal!")


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_rsqrt():
    _run_positive_unary(rsqrt_kernel, lambda a: torch.rsqrt(a))
    logging.info("rsqrt result equal!")


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_recip():
    _run_positive_unary(recip_kernel, lambda a: torch.reciprocal(a))
    logging.info("recip result equal!")


@pl.jit(auto_mutex=True)
def sqrt_kernel(a: pl.Tensor[[64, 64], pl.DT_FP32], out: pl.Tensor[[64, 64], pl.DT_FP32]):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_out = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.sqrt(cur_out, cur_a)
        pl.store(out, cur_out, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_sqrt():
    _run_positive_unary(sqrt_kernel, lambda a: torch.sqrt(a))
    logging.info("sqrt result equal!")


# ===========================================================================
# 复合/融合：fused_mul_add / add_relu / sub_relu
# ===========================================================================
@pl.jit(auto_mutex=True)
def fused_mul_add_kernel(
    a: pl.Tensor[[64, 64], pl.DT_FP32],
    b: pl.Tensor[[64, 64], pl.DT_FP32],
    c: pl.Tensor[[64, 64], pl.DT_FP32],
):
    # fused_mul_add 为 in-place 融合乘加：c = c * a + b
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_b = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    tile_c = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_b = tile_b.current()
        cur_c = tile_c.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.load(cur_c, c, [0, 0])
        pl.fused_mul_add(cur_c, cur_a, cur_b)
        pl.store(c, cur_c, [0, 0])


@pl.jit(auto_mutex=True)
def add_relu_kernel(
    a: pl.Tensor[[64, 64], pl.DT_FP32], b: pl.Tensor[[64, 64], pl.DT_FP32], out: pl.Tensor[[64, 64], pl.DT_FP32]
):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_b = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_b = tile_b.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.add_relu(cur_out, cur_a, cur_b)
        pl.store(out, cur_out, [0, 0])


@pl.jit(auto_mutex=True)
def sub_relu_kernel(
    a: pl.Tensor[[64, 64], pl.DT_FP32], b: pl.Tensor[[64, 64], pl.DT_FP32], out: pl.Tensor[[64, 64], pl.DT_FP32]
):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_b = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_b = tile_b.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.sub_relu(cur_out, cur_a, cur_b)
        pl.store(out, cur_out, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_fused_mul_add():
    device = ST_DEVICE
    _require_a5(device)
    torch.manual_seed(0)
    a = torch.randn(64, 64, device=device, dtype=torch.float32)
    b = torch.randn(64, 64, device=device, dtype=torch.float32)
    c = torch.randn(64, 64, device=device, dtype=torch.float32)
    c_ref = c * a + b
    fused_mul_add_kernel(a, b, c)
    torch.npu.synchronize()
    torch.testing.assert_close(c, c_ref, rtol=1e-2, atol=1e-2)
    logging.info("fused_mul_add result equal!")


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_add_relu():
    _run_binary(add_relu_kernel, lambda a, b: torch.relu(a + b))
    logging.info("add_relu result equal!")


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_sub_relu():
    _run_binary(sub_relu_kernel, lambda a, b: torch.relu(a - b))
    logging.info("sub_relu result equal!")


# ===========================================================================
# 类型转换融合：add_relu_cast / mul_cast / sub_relu_cast（输入 FP16，输出 FP32）
# ===========================================================================
@pl.jit(auto_mutex=True)
def add_relu_cast_kernel(
    a: pl.Tensor[[64, 64], pl.DT_FP16], b: pl.Tensor[[64, 64], pl.DT_FP16], out: pl.Tensor[[64, 64], pl.DT_FP32]
):
    tt_in = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tt_out = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt_in, addrs=0x0000, mutex_ids=[0])
    tile_b = pl.make_tile_group(type=tt_in, addrs=0x2000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=tt_out, addrs=0x4000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_b = tile_b.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.add_relu_cast(cur_out, cur_a, cur_b, target_type=pl.DT_FP32, mode=pl.RoundMode.CAST_ROUND)
        pl.store(out, cur_out, [0, 0])


@pl.jit(auto_mutex=True)
def mul_cast_kernel(
    a: pl.Tensor[[64, 64], pl.DT_FP16], b: pl.Tensor[[64, 64], pl.DT_FP16], out: pl.Tensor[[64, 64], pl.DT_FP32]
):
    tt_in = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tt_out = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt_in, addrs=0x0000, mutex_ids=[0])
    tile_b = pl.make_tile_group(type=tt_in, addrs=0x2000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=tt_out, addrs=0x4000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_b = tile_b.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.mul_cast(cur_out, cur_a, cur_b, target_type=pl.DT_FP32, mode=pl.RoundMode.CAST_ROUND)
        pl.store(out, cur_out, [0, 0])


@pl.jit(auto_mutex=True)
def sub_relu_cast_kernel(
    a: pl.Tensor[[64, 64], pl.DT_FP16], b: pl.Tensor[[64, 64], pl.DT_FP16], out: pl.Tensor[[64, 64], pl.DT_FP32]
):
    tt_in = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tt_out = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt_in, addrs=0x0000, mutex_ids=[0])
    tile_b = pl.make_tile_group(type=tt_in, addrs=0x2000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=tt_out, addrs=0x4000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_b = tile_b.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.sub_relu_cast(cur_out, cur_a, cur_b, target_type=pl.DT_FP32, mode=pl.RoundMode.CAST_ROUND)
        pl.store(out, cur_out, [0, 0])


def _run_cast_binary(kernel, ref_fn):
    device = ST_DEVICE
    _require_a5(device)
    torch.manual_seed(0)
    a = torch.randn(64, 64, device=device, dtype=torch.float16)
    b = torch.randn(64, 64, device=device, dtype=torch.float16)
    out = torch.zeros(64, 64, device=device, dtype=torch.float32)
    kernel(a, b, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, ref_fn(a.float(), b.float()), rtol=1e-2, atol=1e-2)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_add_relu_cast():
    _run_cast_binary(add_relu_cast_kernel, lambda a, b: torch.relu(a + b))
    logging.info("add_relu_cast result equal!")


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_mul_cast():
    _run_cast_binary(mul_cast_kernel, lambda a, b: a * b)
    logging.info("mul_cast result equal!")


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_sub_relu_cast():
    _run_cast_binary(sub_relu_cast_kernel, lambda a, b: torch.relu(a - b))
    logging.info("sub_relu_cast result equal!")


def _doc_binary(title):
    def collect(ctx, kernel):
        a = ctx.base_fp32((64, 64), 0.25, 1.0)
        b = ctx.base_fp32((64, 64), 0.5, 10.0)
        out = torch.zeros((64, 64), device=ctx.device, dtype=torch.float32)
        kernel(a, b, out)
        ctx.synchronize()
        return ctx.snippet(title, {"a": a, "b": b}, {"out": out})

    return collect


def _doc_unary(title, positive=False):
    def collect(ctx, kernel):
        a = ctx.base_fp32((64, 64), 0.125, 1.0 if positive else -4.0)
        out = torch.zeros((64, 64), device=ctx.device, dtype=torch.float32)
        kernel(a, out)
        ctx.synchronize()
        return ctx.snippet(title, {"a": a}, {"out": out})

    return collect


def _doc_cast_binary(title):
    def collect(ctx, kernel):
        a = ctx.base_fp16((64, 64), 0.25, -2.0)
        b = ctx.base_fp16((64, 64), -0.125, 3.0)
        out = torch.zeros((64, 64), device=ctx.device, dtype=torch.float32)
        kernel(a, b, out)
        ctx.synchronize()
        return ctx.snippet(title, {"a": a, "b": b}, {"out": out})

    return collect


def _doc_fused_mul_add(ctx, kernel):
    a = ctx.base_fp32((64, 64), 0.25, 1.0)
    b = ctx.base_fp32((64, 64), 0.5, -3.0)
    c = ctx.base_fp32((64, 64), -0.125, 2.0)
    c_original = c.clone()
    kernel(a, b, c)
    ctx.synchronize()
    return ctx.snippet("fused_mul_add", {"a": a, "b": b, "c原始值": c_original}, {"c": c})


def _doc_add_relu(ctx, kernel):
    a = ctx.base_fp32((64, 64), 0.25, -6.0)
    b = ctx.base_fp32((64, 64), 0.5, 1.0)
    out = torch.zeros((64, 64), device=ctx.device)
    kernel(a, b, out)
    ctx.synchronize()
    return ctx.snippet("add_relu", {"a": a, "b": b}, {"out": out})


def _doc_sub_relu(ctx, kernel):
    a = ctx.base_fp32((64, 64), 0.5, 2.0)
    b = ctx.base_fp32((64, 64), 0.25, 3.0)
    out = torch.zeros((64, 64), device=ctx.device)
    kernel(a, b, out)
    ctx.synchronize()
    return ctx.snippet("sub_relu", {"a": a, "b": b}, {"out": out})


def _doc_xor(ctx, kernel):
    a = torch.arange(64 * 64, device=ctx.device, dtype=torch.int32).reshape(64, 64) + 2
    b = (torch.arange(64 * 64, device=ctx.device, dtype=torch.int32).reshape(64, 64) % 16) + 1
    out = torch.zeros((64, 64), device=ctx.device, dtype=torch.int32)
    kernel(a, b, out)
    ctx.synchronize()
    return ctx.snippet("xor", {"a": a, "b": b}, {"out": out})


DOC_OUTPUT_CASES = {
    "add": _doc_binary("add"),
    "sub": _doc_binary("sub"),
    "mul": _doc_binary("mul"),
    "div": _doc_binary("div"),
    "maximum": _doc_binary("maximum"),
    "minimum": _doc_binary("minimum"),
    "neg": _doc_unary("neg"),
    "relu": _doc_unary("relu"),
    "exp": _doc_unary("exp"),
    "rsqrt": _doc_unary("rsqrt", positive=True),
    "recip": _doc_unary("recip", positive=True),
    "add_relu_cast": _doc_cast_binary("add_relu_cast"),
    "mul_cast": _doc_cast_binary("mul_cast"),
    "sub_relu_cast": _doc_cast_binary("sub_relu_cast"),
    "fused_mul_add": _doc_fused_mul_add,
    "add_relu": _doc_add_relu,
    "sub_relu": _doc_sub_relu,
    "xor": _doc_xor,
}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for fn in [
        test_add,
        test_sub,
        test_mul,
        test_maximum,
        test_div,
        test_minimum,
        test_xor,
        test_add_scalar,
        test_sub_scalar,
        test_mul_scalar,
        test_div_scalar,
        test_minimum_scalar,
        test_maximum_scalar,
        test_exp,
        test_relu,
        test_neg,
        test_rsqrt,
        test_recip,
        test_sqrt,
        test_fused_mul_add,
        test_add_relu,
        test_sub_relu,
        test_add_relu_cast,
        test_mul_cast,
        test_sub_relu_cast,
    ]:
        fn()
    logging.info("\nAll batch-1 element-wise / fused examples passed!")
