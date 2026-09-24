# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""if control_flow 前端测试。

测试覆盖场景:
  1. 覆盖 if control_flow 场景的前端语法、编译入口与运行验证。
  2. 覆盖 dtype 泛化：DT_FP16/DT_BF16/DT_FP32/DT_INT32; Tensor 维度/shape 泛化：2D。
  3. 覆盖控制流组合：for 循环、if/elif/else 分支、with section 上下文、无值 return 提前退出、constexpr 编译期条件。
  4. 覆盖接口组合：基础算术运算、激活/比较类运算、Tensor 与 Tile 数据搬运、section_vector/section_cube 上下文。
  5. 覆盖边界场景：constexpr、常量比较、truthiness、三元表达式和深层嵌套分支。
"""

import logging
import os

import pypto_pro.language as pl
import pytest
import torch

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"


TILE_M = 64
TILE_N = 64

DTYPES_ALL = [
    (pl.DT_FP16, torch.float16, "fp16", 1e-2, 1e-2),
    (pl.DT_BF16, torch.bfloat16, "bf16", 1e-2, 1e-2),
    (pl.DT_FP32, torch.float32, "fp32", 1e-4, 1e-4),
    (pl.DT_INT32, torch.int32, "int32", 0, 0),
]
DTYPES_FP16 = [DTYPES_ALL[0]]


def _gen(shape, tdt, device):
    if tdt == torch.int32:
        x = torch.randint(-100, 100, shape, device=device, dtype=tdt)
        y = torch.randint(-100, 100, shape, device=device, dtype=tdt)
    else:
        x = torch.rand(shape, device=device, dtype=tdt)
        y = torch.rand(shape, device=device, dtype=tdt)
    z = torch.zeros(shape, device=device, dtype=tdt)
    return x, y, z


def _ref(tdt, fn, x, y):
    if tdt == torch.int32:
        return fn(x, y)
    return fn(x.float(), y.float()).to(tdt)


# ===================================================================
# if_else: if/else with add/sub  (FP16 / BF16 / FP32 / INT32)
# ===================================================================


# =============================================================================
# Test 1: if/else 加减分支 - FP16
#         if/else add-sub branch - FP16
# =============================================================================
@pl.jit(auto_mutex=True)
def if_else_fp16_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    m = x.shape[0]
    n = x.shape[1]
    tile_type = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a_db = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0, 1])
    b_db = pl.make_tile_group(type=tile_type, addrs=0x4000, mutex_ids=[2, 3])
    c_db = pl.make_tile_group(type=tile_type, addrs=0x8000, mutex_ids=[30, 31])
    with pl.section_vector():
        for i in pl.range(0, m // TILE_M, 1):
            for j in pl.range(0, n // TILE_N, 1):
                tile_a = a_db.next()
                tile_b = b_db.next()
                tile_c = c_db.next()
                pl.load_tile(tile_a, x, [i, j])
                pl.load_tile(tile_b, y, [i, j])
                if i == 0:
                    pl.add(tile_c, tile_a, tile_b)
                else:
                    pl.sub(tile_c, tile_a, tile_b)
                pl.store_tile(z, tile_c, [i, j])


# =============================================================================
# Test 2: if/else 加减分支 - BF16
#         if/else add-sub branch - BF16
# =============================================================================


@pl.jit(auto_mutex=True)
def if_else_bf16_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
):
    m = x.shape[0]
    n = x.shape[1]
    tile_type = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec)
    a_db = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0, 1])
    b_db = pl.make_tile_group(type=tile_type, addrs=0x4000, mutex_ids=[2, 3])
    c_db = pl.make_tile_group(type=tile_type, addrs=0x8000, mutex_ids=[30, 31])
    with pl.section_vector():
        for i in pl.range(0, m // TILE_M, 1):
            for j in pl.range(0, n // TILE_N, 1):
                tile_a = a_db.next()
                tile_b = b_db.next()
                tile_c = c_db.next()
                pl.load_tile(tile_a, x, [i, j])
                pl.load_tile(tile_b, y, [i, j])
                if i == 0:
                    pl.add(tile_c, tile_a, tile_b)
                else:
                    pl.sub(tile_c, tile_a, tile_b)
                pl.store_tile(z, tile_c, [i, j])


# =============================================================================
# Test 3: if/else 加减分支 - FP32
#         if/else add-sub branch - FP32
# =============================================================================


@pl.jit(auto_mutex=True)
def if_else_fp32_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    m = x.shape[0]
    n = x.shape[1]
    tile_type = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    a_db = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0, 1])
    b_db = pl.make_tile_group(type=tile_type, addrs=0x8000, mutex_ids=[2, 3])
    c_db = pl.make_tile_group(type=tile_type, addrs=0x10000, mutex_ids=[30, 31])
    with pl.section_vector():
        for i in pl.range(0, m // TILE_M, 1):
            for j in pl.range(0, n // TILE_N, 1):
                tile_a = a_db.next()
                tile_b = b_db.next()
                tile_c = c_db.next()
                pl.load_tile(tile_a, x, [i, j])
                pl.load_tile(tile_b, y, [i, j])
                if i == 0:
                    pl.add(tile_c, tile_a, tile_b)
                else:
                    pl.sub(tile_c, tile_a, tile_b)
                pl.store_tile(z, tile_c, [i, j])


# =============================================================================
# Test 4: if/else 加减分支 - INT32
#         if/else add-sub branch - INT32
# =============================================================================


@pl.jit(auto_mutex=True)
def if_else_int32_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT32],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT32],
    z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT32],
):
    m = x.shape[0]
    n = x.shape[1]
    tile_type = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    a_db = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0, 1])
    b_db = pl.make_tile_group(type=tile_type, addrs=0x8000, mutex_ids=[2, 3])
    c_db = pl.make_tile_group(type=tile_type, addrs=0x10000, mutex_ids=[30, 31])
    with pl.section_vector():
        for i in pl.range(0, m // TILE_M, 1):
            for j in pl.range(0, n // TILE_N, 1):
                tile_a = a_db.next()
                tile_b = b_db.next()
                tile_c = c_db.next()
                pl.load_tile(tile_a, x, [i, j])
                pl.load_tile(tile_b, y, [i, j])
                if i == 0:
                    pl.add(tile_c, tile_a, tile_b)
                else:
                    pl.sub(tile_c, tile_a, tile_b)
                pl.store_tile(z, tile_c, [i, j])


IF_ELSE_KERNELS = {
    "fp16": if_else_fp16_kernel,
    "bf16": if_else_bf16_kernel,
    "fp32": if_else_fp32_kernel,
    "int32": if_else_int32_kernel,
}


# ===================================================================
# if_constexpr: constexpr(True) if/else with add/sub  (FP16)
# ===================================================================


# =============================================================================
# Test 5: if + constexpr 编译期条件 - FP16
#         if with constexpr compile-time condition - FP16
# =============================================================================
@pl.jit(auto_mutex=True)
def if_constexpr_fp16_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    m = x.shape[0]
    n = x.shape[1]
    tile_type = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a_db = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0, 1])
    b_db = pl.make_tile_group(type=tile_type, addrs=0x4000, mutex_ids=[2, 3])
    c_db = pl.make_tile_group(type=tile_type, addrs=0x8000, mutex_ids=[30, 31])
    with pl.section_vector():
        for i in pl.range(0, m // TILE_M, 1):
            for j in pl.range(0, n // TILE_N, 1):
                tile_a = a_db.next()
                tile_b = b_db.next()
                tile_c = c_db.next()
                pl.load_tile(tile_a, x, [i, j])
                pl.load_tile(tile_b, y, [i, j])
                if pl.constexpr(True):
                    pl.add(tile_c, tile_a, tile_b)
                else:
                    pl.sub(tile_c, tile_a, tile_b)
                pl.store_tile(z, tile_c, [i, j])


IF_CONSTEXPR_KERNELS = {
    "fp16": if_constexpr_fp16_kernel,
}


# ===================================================================
# if_elif_else: if/elif/else with add/sub/mul  (FP16)
# ===================================================================


# =============================================================================
# Test 6: if/elif/else 多分支 - FP16
#         if/elif/else multi-branch - FP16
# =============================================================================
@pl.jit(auto_mutex=True)
def if_elif_else_fp16_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    m = x.shape[0]
    n = x.shape[1]
    tile_type = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a_db = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0, 1])
    b_db = pl.make_tile_group(type=tile_type, addrs=0x4000, mutex_ids=[2, 3])
    c_db = pl.make_tile_group(type=tile_type, addrs=0x8000, mutex_ids=[30, 31])
    with pl.section_vector():
        for i in pl.range(0, m // TILE_M, 1):
            for j in pl.range(0, n // TILE_N, 1):
                tile_a = a_db.next()
                tile_b = b_db.next()
                tile_c = c_db.next()
                pl.load_tile(tile_a, x, [i, j])
                pl.load_tile(tile_b, y, [i, j])
                if i == 0:
                    pl.add(tile_c, tile_a, tile_b)
                elif i == 1:
                    pl.sub(tile_c, tile_a, tile_b)
                else:
                    pl.mul(tile_c, tile_a, tile_b)
                pl.store_tile(z, tile_c, [i, j])


IF_ELIF_ELSE_KERNELS = {
    "fp16": if_elif_else_fp16_kernel,
}


# ===================================================================
# nested_if: nested if/else with add/sub/mul  (FP16)
# ===================================================================


# =============================================================================
# Test 7: 嵌套 if 分支 - FP16
#         nested if branch - FP16
# =============================================================================
@pl.jit(auto_mutex=True)
def nested_if_fp16_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    m = x.shape[0]
    n = x.shape[1]
    tile_type = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a_db = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0, 1])
    b_db = pl.make_tile_group(type=tile_type, addrs=0x4000, mutex_ids=[2, 3])
    c_db = pl.make_tile_group(type=tile_type, addrs=0x8000, mutex_ids=[30, 31])
    with pl.section_vector():
        for i in pl.range(0, m // TILE_M, 1):
            for j in pl.range(0, n // TILE_N, 1):
                tile_a = a_db.next()
                tile_b = b_db.next()
                tile_c = c_db.next()
                pl.load_tile(tile_a, x, [i, j])
                pl.load_tile(tile_b, y, [i, j])
                if i == 0:
                    if j == 0:
                        pl.add(tile_c, tile_a, tile_b)
                    else:
                        pl.sub(tile_c, tile_a, tile_b)
                else:
                    pl.mul(tile_c, tile_a, tile_b)
                pl.store_tile(z, tile_c, [i, j])


NESTED_IF_KERNELS = {
    "fp16": nested_if_fp16_kernel,
}


# ===================================================================
# if_relu: if/else with relu/add  (FP16)
# ===================================================================


# =============================================================================
# Test 8: if 分支 ReLU - FP16
#         if branch ReLU - FP16
# =============================================================================
@pl.jit(auto_mutex=True)
def if_relu_fp16_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    m = x.shape[0]
    n = x.shape[1]
    tile_type = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a_db = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0, 1])
    b_db = pl.make_tile_group(type=tile_type, addrs=0x4000, mutex_ids=[2, 3])
    c_db = pl.make_tile_group(type=tile_type, addrs=0x8000, mutex_ids=[30, 31])
    with pl.section_vector():
        for i in pl.range(0, m // TILE_M, 1):
            for j in pl.range(0, n // TILE_N, 1):
                tile_a = a_db.next()
                tile_b = b_db.next()
                tile_c = c_db.next()
                pl.load_tile(tile_a, x, [i, j])
                pl.load_tile(tile_b, y, [i, j])
                if i == 0:
                    pl.relu(tile_c, tile_a)
                else:
                    pl.add(tile_c, tile_a, tile_b)
                pl.store_tile(z, tile_c, [i, j])


IF_RELU_KERNELS = {
    "fp16": if_relu_fp16_kernel,
}


# ===================================================================
# constexpr: constexpr(True) if/else with add/sub  (FP16)
# ===================================================================


# =============================================================================
# Test 9: constexpr 分支 - FP16
#         constexpr branch - FP16
# =============================================================================
@pl.jit(auto_mutex=True)
def constexpr_fp16_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    m = x.shape[0]
    n = x.shape[1]
    tile_type = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a_db = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0, 1])
    b_db = pl.make_tile_group(type=tile_type, addrs=0x4000, mutex_ids=[2, 3])
    c_db = pl.make_tile_group(type=tile_type, addrs=0x8000, mutex_ids=[30, 31])
    with pl.section_vector():
        for i in pl.range(0, m // TILE_M, 1):
            for j in pl.range(0, n // TILE_N, 1):
                tile_a = a_db.next()
                tile_b = b_db.next()
                tile_c = c_db.next()
                pl.load_tile(tile_a, x, [i, j])
                pl.load_tile(tile_b, y, [i, j])
                if pl.constexpr(True):
                    pl.add(tile_c, tile_a, tile_b)
                else:
                    pl.sub(tile_c, tile_a, tile_b)
                pl.store_tile(z, tile_c, [i, j])


CONSTEXPR_KERNELS = {
    "fp16": constexpr_fp16_kernel,
}


# ===================================================================
# if_always_false: if(i<0) never true with add/sub  (FP16)
# ===================================================================


# =============================================================================
# Test 10: 恒 False if 分支 - FP16
#         always-false if branch - FP16
# =============================================================================
@pl.jit(auto_mutex=True)
def if_always_false_fp16_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    m = x.shape[0]
    n = x.shape[1]
    tile_type = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a_db = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0, 1])
    b_db = pl.make_tile_group(type=tile_type, addrs=0x4000, mutex_ids=[2, 3])
    c_db = pl.make_tile_group(type=tile_type, addrs=0x8000, mutex_ids=[30, 31])
    with pl.section_vector():
        for i in pl.range(0, m // TILE_M, 1):
            for j in pl.range(0, n // TILE_N, 1):
                tile_a = a_db.next()
                tile_b = b_db.next()
                tile_c = c_db.next()
                pl.load_tile(tile_a, x, [i, j])
                pl.load_tile(tile_b, y, [i, j])
                if i < 0:
                    pl.add(tile_c, tile_a, tile_b)
                else:
                    pl.sub(tile_c, tile_a, tile_b)
                pl.store_tile(z, tile_c, [i, j])


IF_ALWAYS_FALSE_KERNELS = {
    "fp16": if_always_false_fp16_kernel,
}


# ===================================================================
# deeply_nested_if: deeply nested if/else with relu/mul/sub/neg  (FP16)
# ===================================================================


# =============================================================================
# Test 11: 深层嵌套 if 分支 - FP16
#         deeply nested if branch - FP16
# =============================================================================
@pl.jit(auto_mutex=True)
def deeply_nested_if_fp16_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    m = x.shape[0]
    n = x.shape[1]
    tile_type = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a_db = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0, 1])
    b_db = pl.make_tile_group(type=tile_type, addrs=0x4000, mutex_ids=[2, 3])
    c_db = pl.make_tile_group(type=tile_type, addrs=0x8000, mutex_ids=[30, 31])
    with pl.section_vector():
        for i in pl.range(0, m // TILE_M, 1):
            for j in pl.range(0, n // TILE_N, 1):
                tile_a = a_db.next()
                tile_b = b_db.next()
                tile_c = c_db.next()
                pl.load_tile(tile_a, x, [i, j])
                pl.load_tile(tile_b, y, [i, j])
                if i == 0:
                    if j == 0:
                        pl.add(tile_c, tile_a, tile_b)
                        pl.relu(tile_c, tile_c)
                    else:
                        pl.mul(tile_c, tile_a, tile_b)
                else:
                    if j == 0:
                        pl.sub(tile_c, tile_a, tile_b)
                    else:
                        pl.neg(tile_c, tile_a)
                pl.store_tile(z, tile_c, [i, j])


DEEPLY_NESTED_IF_KERNELS = {
    "fp16": deeply_nested_if_fp16_kernel,
}


# ===================================================================
# Test functions
# ===================================================================


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_if_else():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shape = [128, 64]
    for _pl_dt, tdt, label, atol, rtol in DTYPES_ALL:
        kernel = IF_ELSE_KERNELS[label]
        x, y, z = _gen(shape, tdt, device)
        kernel(x, y, z)
        torch.npu.synchronize()
        z_ref = torch.zeros(shape, device=device, dtype=tdt)
        z_ref[:64, :] = _ref(tdt, lambda a, b: a + b, x[:64, :], y[:64, :])
        z_ref[64:, :] = _ref(tdt, lambda a, b: a - b, x[64:, :], y[64:, :])
        torch.testing.assert_close(z, z_ref, atol=atol, rtol=rtol)
        logging.info("test_if_else [%s] passed! shape=%s", label, shape)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_if_constexpr():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shape = [128, 64]
    for _pl_dt, tdt, label, atol, rtol in DTYPES_FP16:
        kernel = IF_CONSTEXPR_KERNELS[label]
        x, y, z = _gen(shape, tdt, device)
        kernel(x, y, z)
        torch.npu.synchronize()
        z_ref = _ref(tdt, lambda a, b: a + b, x, y)
        torch.testing.assert_close(z, z_ref, atol=atol, rtol=rtol)
        logging.info("test_if_constexpr [%s] passed! shape=%s", label, shape)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_if_elif_else():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shape = [192, 128]
    for _pl_dt, tdt, label, atol, rtol in DTYPES_FP16:
        kernel = IF_ELIF_ELSE_KERNELS[label]
        x, y, z = _gen(shape, tdt, device)
        kernel(x, y, z)
        torch.npu.synchronize()
        z_ref = torch.zeros(shape, device=device, dtype=tdt)
        z_ref[:64, :] = _ref(tdt, lambda a, b: a + b, x[:64, :], y[:64, :])
        z_ref[64:128, :] = _ref(tdt, lambda a, b: a - b, x[64:128, :], y[64:128, :])
        z_ref[128:, :] = _ref(tdt, lambda a, b: a * b, x[128:, :], y[128:, :])
        torch.testing.assert_close(z, z_ref, atol=atol, rtol=rtol)
        logging.info("test_if_elif_else [%s] passed! shape=%s", label, shape)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_nested_if():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shape = [128, 128]
    for _pl_dt, tdt, label, atol, rtol in DTYPES_FP16:
        kernel = NESTED_IF_KERNELS[label]
        x, y, z = _gen(shape, tdt, device)
        kernel(x, y, z)
        torch.npu.synchronize()
        z_ref = torch.zeros(shape, device=device, dtype=tdt)
        z_ref[:64, :64] = _ref(tdt, lambda a, b: a + b, x[:64, :64], y[:64, :64])
        z_ref[:64, 64:] = _ref(tdt, lambda a, b: a - b, x[:64, 64:], y[:64, 64:])
        z_ref[64:, :] = _ref(tdt, lambda a, b: a * b, x[64:, :], y[64:, :])
        torch.testing.assert_close(z, z_ref, atol=atol, rtol=rtol)
        logging.info("test_nested_if [%s] passed! shape=%s", label, shape)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_if_relu():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shape = [128, 64]
    for _pl_dt, tdt, label, atol, rtol in DTYPES_FP16:
        kernel = IF_RELU_KERNELS[label]
        x, y, z = _gen(shape, tdt, device)
        kernel(x, y, z)
        torch.npu.synchronize()
        z_ref = torch.zeros(shape, device=device, dtype=tdt)
        z_ref[:64, :] = _ref(tdt, lambda a, b: torch.relu(a), x[:64, :], y[:64, :])
        z_ref[64:, :] = _ref(tdt, lambda a, b: a + b, x[64:, :], y[64:, :])
        torch.testing.assert_close(z, z_ref, atol=atol, rtol=rtol)
        logging.info("test_if_relu [%s] passed! shape=%s", label, shape)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_constexpr():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shape = [128, 64]
    for _pl_dt, tdt, label, atol, rtol in DTYPES_FP16:
        kernel = CONSTEXPR_KERNELS[label]
        x, y, z = _gen(shape, tdt, device)
        kernel(x, y, z)
        torch.npu.synchronize()
        z_ref = _ref(tdt, lambda a, b: a + b, x, y)
        torch.testing.assert_close(z, z_ref, atol=atol, rtol=rtol)
        logging.info("test_constexpr [%s] passed! shape=%s", label, shape)


# ===================================================================
# if_lt_const: if(i < threshold) add/sub with numeric threshold
# ===================================================================


# =============================================================================
# Test 12: if < 常量比较 - FP16
#         if less-than constant comparison - FP16
# =============================================================================
@pl.jit(auto_mutex=True)
def if_lt_const_fp16_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    m = x.shape[0]
    n = x.shape[1]
    tile_type = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a_db = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0, 1])
    b_db = pl.make_tile_group(type=tile_type, addrs=0x4000, mutex_ids=[2, 3])
    c_db = pl.make_tile_group(type=tile_type, addrs=0x8000, mutex_ids=[30, 31])
    with pl.section_vector():
        for i in pl.range(0, m // TILE_M, 1):
            for j in pl.range(0, n // TILE_N, 1):
                tile_a = a_db.next()
                tile_b = b_db.next()
                tile_c = c_db.next()
                pl.load_tile(tile_a, x, [i, j])
                pl.load_tile(tile_b, y, [i, j])
                if i < 2:
                    pl.add(tile_c, tile_a, tile_b)
                else:
                    pl.sub(tile_c, tile_a, tile_b)
                pl.store_tile(z, tile_c, [i, j])


IF_LT_CONST_KERNELS = {
    "fp16": if_lt_const_fp16_kernel,
}


# ===================================================================
# if_ge_const_mul: if(i >= threshold) add/mul with numeric threshold
# ===================================================================


# =============================================================================
# Test 13: if >= 常量后乘法 - FP16
#         if greater-or-equal constant then multiply - FP16
# =============================================================================
@pl.jit(auto_mutex=True)
def if_ge_const_mul_fp16_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    m = x.shape[0]
    n = x.shape[1]
    tile_type = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a_db = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0, 1])
    b_db = pl.make_tile_group(type=tile_type, addrs=0x4000, mutex_ids=[2, 3])
    c_db = pl.make_tile_group(type=tile_type, addrs=0x8000, mutex_ids=[30, 31])
    with pl.section_vector():
        for i in pl.range(0, m // TILE_M, 1):
            for j in pl.range(0, n // TILE_N, 1):
                tile_a = a_db.next()
                tile_b = b_db.next()
                tile_c = c_db.next()
                pl.load_tile(tile_a, x, [i, j])
                pl.load_tile(tile_b, y, [i, j])
                if i >= 1:
                    pl.mul(tile_c, tile_a, tile_b)
                else:
                    pl.add(tile_c, tile_a, tile_b)
                pl.store_tile(z, tile_c, [i, j])


IF_GE_CONST_MUL_KERNELS = {
    "fp16": if_ge_const_mul_fp16_kernel,
}


# ===================================================================
# if_jt_const: if(j < threshold) add/sub with j numeric threshold
# ===================================================================


# =============================================================================
# Test 14: if j/t 索引常量比较 - FP16
#         if j/t index constant comparison - FP16
# =============================================================================
@pl.jit(auto_mutex=True)
def if_jt_const_fp16_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    m = x.shape[0]
    n = x.shape[1]
    tile_type = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a_db = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0, 1])
    b_db = pl.make_tile_group(type=tile_type, addrs=0x4000, mutex_ids=[2, 3])
    c_db = pl.make_tile_group(type=tile_type, addrs=0x8000, mutex_ids=[30, 31])
    with pl.section_vector():
        for i in pl.range(0, m // TILE_M, 1):
            for j in pl.range(0, n // TILE_N, 1):
                tile_a = a_db.next()
                tile_b = b_db.next()
                tile_c = c_db.next()
                pl.load_tile(tile_a, x, [i, j])
                pl.load_tile(tile_b, y, [i, j])
                if j < 1:
                    pl.add(tile_c, tile_a, tile_b)
                else:
                    pl.sub(tile_c, tile_a, tile_b)
                pl.store_tile(z, tile_c, [i, j])


IF_JT_CONST_KERNELS = {
    "fp16": if_jt_const_fp16_kernel,
}


# ===================================================================
# if_ij_cmp: if(i < j) add/sub comparing two loop vars
# ===================================================================


# =============================================================================
# Test 15: if i/j 索引比较 - FP16
#         if i/j index comparison - FP16
# =============================================================================
@pl.jit(auto_mutex=True)
def if_ij_cmp_fp16_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    m = x.shape[0]
    n = x.shape[1]
    tile_type = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a_db = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0, 1])
    b_db = pl.make_tile_group(type=tile_type, addrs=0x4000, mutex_ids=[2, 3])
    c_db = pl.make_tile_group(type=tile_type, addrs=0x8000, mutex_ids=[30, 31])
    with pl.section_vector():
        for i in pl.range(0, m // TILE_M, 1):
            for j in pl.range(0, n // TILE_N, 1):
                tile_a = a_db.next()
                tile_b = b_db.next()
                tile_c = c_db.next()
                pl.load_tile(tile_a, x, [i, j])
                pl.load_tile(tile_b, y, [i, j])
                if i < j:
                    pl.add(tile_c, tile_a, tile_b)
                else:
                    pl.sub(tile_c, tile_a, tile_b)
                pl.store_tile(z, tile_c, [i, j])


IF_IJ_CMP_KERNELS = {
    "fp16": if_ij_cmp_fp16_kernel,
}


# ===================================================================
# if_elif_const: if/elif/else with numeric thresholds on i
# ===================================================================


# =============================================================================
# Test 16: if/elif 常量分支 - FP16
#         if/elif constant branch - FP16
# =============================================================================
@pl.jit(auto_mutex=True)
def if_elif_const_fp16_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    m = x.shape[0]
    n = x.shape[1]
    tile_type = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a_db = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0, 1])
    b_db = pl.make_tile_group(type=tile_type, addrs=0x4000, mutex_ids=[2, 3])
    c_db = pl.make_tile_group(type=tile_type, addrs=0x8000, mutex_ids=[30, 31])
    with pl.section_vector():
        for i in pl.range(0, m // TILE_M, 1):
            for j in pl.range(0, n // TILE_N, 1):
                tile_a = a_db.next()
                tile_b = b_db.next()
                tile_c = c_db.next()
                pl.load_tile(tile_a, x, [i, j])
                pl.load_tile(tile_b, y, [i, j])
                if i < 1:
                    pl.add(tile_c, tile_a, tile_b)
                elif i < 3:
                    pl.sub(tile_c, tile_a, tile_b)
                else:
                    pl.mul(tile_c, tile_a, tile_b)
                pl.store_tile(z, tile_c, [i, j])


IF_ELIF_CONST_KERNELS = {
    "fp16": if_elif_const_fp16_kernel,
}


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_if_always_false():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shape = [128, 64]
    for _pl_dt, tdt, label, atol, rtol in DTYPES_FP16:
        kernel = IF_ALWAYS_FALSE_KERNELS[label]
        x, y, z = _gen(shape, tdt, device)
        kernel(x, y, z)
        torch.npu.synchronize()
        z_ref = _ref(tdt, lambda a, b: a - b, x, y)
        torch.testing.assert_close(z, z_ref, atol=atol, rtol=rtol)
        logging.info("test_if_always_false [%s] passed! shape=%s", label, shape)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_deeply_nested_if():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shape = [128, 128]
    for _pl_dt, tdt, label, atol, rtol in DTYPES_FP16:
        kernel = DEEPLY_NESTED_IF_KERNELS[label]
        x, y, z = _gen(shape, tdt, device)
        kernel(x, y, z)
        torch.npu.synchronize()
        z_ref = torch.zeros(shape, device=device, dtype=tdt)
        z_ref[:64, :64] = _ref(tdt, lambda a, b: torch.relu(a + b), x[:64, :64], y[:64, :64])
        z_ref[:64, 64:] = _ref(tdt, lambda a, b: a * b, x[:64, 64:], y[:64, 64:])
        z_ref[64:, :64] = _ref(tdt, lambda a, b: a - b, x[64:, :64], y[64:, :64])
        z_ref[64:, 64:] = _ref(tdt, lambda a, b: -a, x[64:, 64:], y[64:, 64:])
        torch.testing.assert_close(z, z_ref, atol=atol, rtol=rtol)
        logging.info("test_deeply_nested_if [%s] passed! shape=%s", label, shape)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_if_lt_const():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shape = [256, 128]
    for _pl_dt, tdt, label, atol, rtol in DTYPES_FP16:
        kernel = IF_LT_CONST_KERNELS[label]
        x, y, z = _gen(shape, tdt, device)
        kernel(x, y, z)
        torch.npu.synchronize()
        z_ref = torch.zeros(shape, device=device, dtype=tdt)
        z_ref[:128, :] = _ref(tdt, lambda a, b: a + b, x[:128, :], y[:128, :])
        z_ref[128:, :] = _ref(tdt, lambda a, b: a - b, x[128:, :], y[128:, :])
        torch.testing.assert_close(z, z_ref, atol=atol, rtol=rtol)
        logging.info("test_if_lt_const [%s] passed! shape=%s", label, shape)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_if_ge_const_mul():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shape = [128, 128]
    for _pl_dt, tdt, label, atol, rtol in DTYPES_FP16:
        kernel = IF_GE_CONST_MUL_KERNELS[label]
        x, y, z = _gen(shape, tdt, device)
        kernel(x, y, z)
        torch.npu.synchronize()
        z_ref = torch.zeros(shape, device=device, dtype=tdt)
        z_ref[:64, :] = _ref(tdt, lambda a, b: a + b, x[:64, :], y[:64, :])
        z_ref[64:, :] = _ref(tdt, lambda a, b: a * b, x[64:, :], y[64:, :])
        torch.testing.assert_close(z, z_ref, atol=atol, rtol=rtol)
        logging.info("test_if_ge_const_mul [%s] passed! shape=%s", label, shape)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_if_jt_const():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shape = [128, 128]
    for _pl_dt, tdt, label, atol, rtol in DTYPES_FP16:
        kernel = IF_JT_CONST_KERNELS[label]
        x, y, z = _gen(shape, tdt, device)
        kernel(x, y, z)
        torch.npu.synchronize()
        z_ref = torch.zeros(shape, device=device, dtype=tdt)
        z_ref[:, :64] = _ref(tdt, lambda a, b: a + b, x[:, :64], y[:, :64])
        z_ref[:, 64:] = _ref(tdt, lambda a, b: a - b, x[:, 64:], y[:, 64:])
        torch.testing.assert_close(z, z_ref, atol=atol, rtol=rtol)
        logging.info("test_if_jt_const [%s] passed! shape=%s", label, shape)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_if_ij_cmp():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shape = [128, 128]
    for _pl_dt, tdt, label, atol, rtol in DTYPES_FP16:
        kernel = IF_IJ_CMP_KERNELS[label]
        x, y, z = _gen(shape, tdt, device)
        kernel(x, y, z)
        torch.npu.synchronize()
        z_ref = torch.zeros(shape, device=device, dtype=tdt)
        for ti in range(shape[0] // TILE_M):
            for tj in range(shape[1] // TILE_N):
                if ti < tj:
                    sl = (slice(ti * TILE_M, (ti + 1) * TILE_M), slice(tj * TILE_N, (tj + 1) * TILE_N))
                    z_ref[sl] = _ref(tdt, lambda a, b: a + b, x[sl], y[sl])
                else:
                    sl = (slice(ti * TILE_M, (ti + 1) * TILE_M), slice(tj * TILE_N, (tj + 1) * TILE_N))
                    z_ref[sl] = _ref(tdt, lambda a, b: a - b, x[sl], y[sl])
        torch.testing.assert_close(z, z_ref, atol=atol, rtol=rtol)
        logging.info("test_if_ij_cmp [%s] passed! shape=%s", label, shape)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_if_elif_const():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shape = [256, 64]
    for _pl_dt, tdt, label, atol, rtol in DTYPES_FP16:
        kernel = IF_ELIF_CONST_KERNELS[label]
        x, y, z = _gen(shape, tdt, device)
        kernel(x, y, z)
        torch.npu.synchronize()
        z_ref = torch.zeros(shape, device=device, dtype=tdt)
        z_ref[:64, :] = _ref(tdt, lambda a, b: a + b, x[:64, :], y[:64, :])
        z_ref[64:192, :] = _ref(tdt, lambda a, b: a - b, x[64:192, :], y[64:192, :])
        z_ref[192:, :] = _ref(tdt, lambda a, b: a * b, x[192:, :], y[192:, :])
        torch.testing.assert_close(z, z_ref, atol=atol, rtol=rtol)
        logging.info("test_if_elif_const [%s] passed! shape=%s", label, shape)


# ===================================================================
# if_compound_bool: compound boolean (and/or) is not supported  (FP16)
# ===================================================================
# if_and / if_or / if_and_three: compound boolean expressions
# ===================================================================


# =============================================================================
# Test 17: if 布尔 and 条件
#         if boolean and condition
# =============================================================================
@pl.jit(auto_mutex=True)
def if_and_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    m = x.shape[0]
    n = x.shape[1]
    tile_type = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a_db = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0, 1])
    b_db = pl.make_tile_group(type=tile_type, addrs=0x4000, mutex_ids=[2, 3])
    c_db = pl.make_tile_group(type=tile_type, addrs=0x8000, mutex_ids=[30, 31])
    with pl.section_vector():
        for i in pl.range(0, m // TILE_M, 1):
            for j in pl.range(0, n // TILE_N, 1):
                tile_a = a_db.next()
                tile_b = b_db.next()
                tile_c = c_db.next()
                pl.load_tile(tile_a, x, [i, j])
                pl.load_tile(tile_b, y, [i, j])
                if i < 2 and j < 1:
                    pl.add(tile_c, tile_a, tile_b)
                else:
                    pl.sub(tile_c, tile_a, tile_b)
                pl.store_tile(z, tile_c, [i, j])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_if_and():
    device = ST_DEVICE
    torch.npu.set_device(device)
    x = torch.rand(128, 64, device=device, dtype=torch.float16)
    y = torch.rand(128, 64, device=device, dtype=torch.float16)
    z = torch.zeros(128, 64, device=device, dtype=torch.float16)
    if_and_kernel(x, y, z)
    torch.npu.synchronize()
    z_ref = torch.zeros(128, 64, device=device, dtype=torch.float16)
    x_f = x.float()
    y_f = y.float()
    for i in range(2):
        sl = (slice(i * TILE_M, (i + 1) * TILE_M), slice(0, TILE_N))
        z_ref[sl] = (x_f[sl] + y_f[sl]).to(torch.float16)
    z_ref[2 * TILE_M:, :] = (x_f[2 * TILE_M:, :] - y_f[2 * TILE_M:, :]).to(torch.float16)
    z_ref[:2 * TILE_M, TILE_N:] = (x_f[:2 * TILE_M, TILE_N:] - y_f[:2 * TILE_M, TILE_N:]).to(torch.float16)
    torch.testing.assert_close(z, z_ref, atol=1e-2, rtol=1e-2)
    logging.info("test_if_and passed")


# =============================================================================
# Test 18: 三元表达式
#         ternary expression
# =============================================================================
@pl.jit(auto_mutex=True)
def ternary_expr_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    m = x.shape[0]
    n = x.shape[1]
    tile_type = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a_db = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0, 1])
    b_db = pl.make_tile_group(type=tile_type, addrs=0x4000, mutex_ids=[2, 3])
    c_db = pl.make_tile_group(type=tile_type, addrs=0x8000, mutex_ids=[30, 31])
    with pl.section_vector():
        for i in pl.range(0, m // TILE_M, 1):
            for j in pl.range(0, n // TILE_N, 1):
                tile_a = a_db.next()
                tile_b = b_db.next()
                tile_c = c_db.next()
                pl.load_tile(tile_a, x, [i, j])
                pl.load_tile(tile_b, y, [i, j])
                sel = 1 if i < 2 else -1
                if sel == 1:
                    pl.add(tile_c, tile_a, tile_b)
                else:
                    pl.sub(tile_c, tile_a, tile_b)
                pl.store_tile(z, tile_c, [i, j])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_ternary_expr():
    device = ST_DEVICE
    torch.npu.set_device(device)
    x = torch.rand(128, 64, device=device, dtype=torch.float16)
    y = torch.rand(128, 64, device=device, dtype=torch.float16)
    z = torch.zeros(128, 64, device=device, dtype=torch.float16)
    ternary_expr_kernel(x, y, z)
    torch.npu.synchronize()
    z_ref = torch.zeros(128, 64, device=device, dtype=torch.float16)
    x_f = x.float()
    y_f = y.float()
    z_ref[:2 * TILE_M, :] = (x_f[:2 * TILE_M, :] + y_f[:2 * TILE_M, :]).to(torch.float16)
    z_ref[2 * TILE_M:, :] = (x_f[2 * TILE_M:, :] - y_f[2 * TILE_M:, :]).to(torch.float16)
    torch.testing.assert_close(z, z_ref, atol=1e-2, rtol=1e-2)
    logging.info("test_ternary_expr passed")


# =============================================================================
# Test 19: if 12 组复合条件
#         if with 12 compound conditions
# =============================================================================
@pl.jit(auto_mutex=True)
def if_compound_12_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    m = x.shape[0]
    n = x.shape[1]
    tile_type = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a_db = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0, 1])
    b_db = pl.make_tile_group(type=tile_type, addrs=0x4000, mutex_ids=[2, 3])
    c_db = pl.make_tile_group(type=tile_type, addrs=0x8000, mutex_ids=[30, 31])
    with pl.section_vector():
        for i in pl.range(0, m // TILE_M, 1):
            for j in pl.range(0, n // TILE_N, 1):
                tile_a = a_db.next()
                tile_b = b_db.next()
                tile_c = c_db.next()
                pl.load_tile(tile_a, x, [i, j])
                pl.load_tile(tile_b, y, [i, j])
                if (
                    i < 1
                    and j < 1
                    or i < 2
                    and j < 1
                    or not i >= 2
                    and j >= 0
                    or i < 3
                    and j >= 0
                    or not j >= 1
                    and i >= 0
                    or i >= 0
                ):
                    pl.add(tile_c, tile_a, tile_b)
                else:
                    pl.sub(tile_c, tile_a, tile_b)
                pl.store_tile(z, tile_c, [i, j])


def _compound_bool_ref(x_f, y_f, i_tiles, j_tiles):
    z_ref = torch.zeros_like(x_f, dtype=torch.float16)
    for ti in range(i_tiles):
        for tj in range(j_tiles):
            i, j = ti, tj
            cond = (
                i < 1
                and j < 1
                or i < 2
                and j < 1
                or not i >= 2
                and j >= 0
                or i < 3
                and j >= 0
                or not j >= 1
                and i >= 0
                or i >= 0
            )
            sl = (slice(ti * TILE_M, (ti + 1) * TILE_M), slice(tj * TILE_N, (tj + 1) * TILE_N))
            if cond:
                z_ref[sl] = (x_f[sl] + y_f[sl]).to(torch.float16)
            else:
                z_ref[sl] = (x_f[sl] - y_f[sl]).to(torch.float16)
    return z_ref


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_if_compound_12():
    device = ST_DEVICE
    torch.npu.set_device(device)
    x = torch.rand(192, 128, device=device, dtype=torch.float16)
    y = torch.rand(192, 128, device=device, dtype=torch.float16)
    z = torch.zeros(192, 128, device=device, dtype=torch.float16)
    if_compound_12_kernel(x, y, z)
    torch.npu.synchronize()
    z_ref = _compound_bool_ref(x.float(), y.float(), 3, 2)
    torch.testing.assert_close(z, z_ref, atol=1e-2, rtol=1e-2)
    logging.info("test_if_compound_12 passed")


# =============================================================================
# Test 20: 12 组三元表达式
#         12 ternary-expression cases
# =============================================================================
@pl.jit(auto_mutex=True)
def ternary_12_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    m = x.shape[0]
    n = x.shape[1]
    tile_type = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a_db = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0, 1])
    b_db = pl.make_tile_group(type=tile_type, addrs=0x4000, mutex_ids=[2, 3])
    c_db = pl.make_tile_group(type=tile_type, addrs=0x8000, mutex_ids=[30, 31])
    with pl.section_vector():
        for i in pl.range(0, m // TILE_M, 1):
            for j in pl.range(0, n // TILE_N, 1):
                tile_a = a_db.next()
                tile_b = b_db.next()
                tile_c = c_db.next()
                pl.load_tile(tile_a, x, [i, j])
                pl.load_tile(tile_b, y, [i, j])
                sel = (
                    (
                        0
                        if i < 1 and j < 1
                        else (
                            1
                            if i < 2 and j < 1
                            else (
                                2
                                if not i >= 2
                                else (
                                    3
                                    if j < 1
                                    else (
                                        4 if i < 3 else (5 if not j >= 1 else (6 if i >= 0 else (7 if j >= 0 else 8)))
                                    )
                                )
                            )
                        )
                    )
                    if i >= 0 and j >= 0
                    else 9
                )
                if sel == 0:
                    pl.add(tile_c, tile_a, tile_b)
                elif sel == 1:
                    pl.sub(tile_c, tile_a, tile_b)
                elif sel == 2:
                    pl.mul(tile_c, tile_a, tile_b)
                else:
                    pl.div(tile_c, tile_a, tile_b)
                pl.store_tile(z, tile_c, [i, j])


def _ternary_deep_ref(x_f, y_f, i_tiles, j_tiles):
    z_ref = torch.zeros_like(x_f, dtype=torch.float16)
    for ti in range(i_tiles):
        for tj in range(j_tiles):
            i, j = ti, tj
            sel = (
                (
                    0
                    if i < 1 and j < 1
                    else (
                        1
                        if i < 2 and j < 1
                        else (
                            2
                            if not i >= 2
                            else (
                                3
                                if j < 1
                                else (4 if i < 3 else (5 if not j >= 1 else (6 if i >= 0 else (7 if j >= 0 else 8))))
                            )
                        )
                    )
                )
                if i >= 0 and j >= 0
                else 9
            )
            sl = (slice(ti * TILE_M, (ti + 1) * TILE_M), slice(tj * TILE_N, (tj + 1) * TILE_N))
            if sel == 0:
                z_ref[sl] = (x_f[sl] + y_f[sl]).to(torch.float16)
            elif sel == 1:
                z_ref[sl] = (x_f[sl] - y_f[sl]).to(torch.float16)
            elif sel == 2:
                z_ref[sl] = (x_f[sl] * y_f[sl]).to(torch.float16)
            else:
                z_ref[sl] = (x_f[sl] / y_f[sl]).to(torch.float16)
    return z_ref


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_ternary_12():
    device = ST_DEVICE
    torch.npu.set_device(device)
    x = torch.rand(192, 128, device=device, dtype=torch.float16)
    y = torch.rand(192, 128, device=device, dtype=torch.float16)
    z = torch.zeros(192, 128, device=device, dtype=torch.float16)
    ternary_12_kernel(x, y, z)
    torch.npu.synchronize()
    z_ref = _ternary_deep_ref(x.float(), y.float(), 3, 2)
    torch.testing.assert_close(z, z_ref, atol=1e-2, rtol=1e-2)
    logging.info("test_ternary_12 passed")


# ===================================================================
# if_while_compound_mixed: if/elif/else inside bounded while  (FP16)
# ===================================================================


@pl.jit(auto_mutex=True)
def if_while_compound_mixed_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    m = x.shape[0]
    n = x.shape[1]
    tile_type = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a_db = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0, 1])
    b_db = pl.make_tile_group(type=tile_type, addrs=0x4000, mutex_ids=[2, 3])
    c_db = pl.make_tile_group(type=tile_type, addrs=0x8000, mutex_ids=[30, 31])
    with pl.section_vector():
        for i in pl.range(0, m // TILE_M, 1):
            j: pl.DT_INT64 = 0
            while j < n // TILE_N and j < 2:
                tile_a = a_db.next()
                tile_b = b_db.next()
                tile_c = c_db.next()
                pl.load_tile(tile_a, x, [i, j])
                pl.load_tile(tile_b, y, [i, j])
                if i == 0 and j == 0:
                    pl.add(tile_c, tile_a, tile_b)
                elif i == 0 or j == 0:
                    pl.sub(tile_c, tile_a, tile_b)
                else:
                    pl.mul(tile_c, tile_a, tile_b)
                pl.store_tile(z, tile_c, [i, j])
                j = j + 1


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_if_while_compound_mixed():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shape = [256, 128]
    x = torch.rand(shape, device=device, dtype=torch.float16)
    y = torch.rand(shape, device=device, dtype=torch.float16)
    z = torch.zeros(shape, device=device, dtype=torch.float16)
    if_while_compound_mixed_kernel(x, y, z)
    torch.npu.synchronize()
    z_ref = torch.zeros(shape, device=device, dtype=torch.float16)
    x_f = x.float()
    y_f = y.float()
    z_ref[:TILE_M, :TILE_N] = (x_f[:TILE_M, :TILE_N] + y_f[:TILE_M, :TILE_N]).to(torch.float16)
    z_ref[:TILE_M, TILE_N:2 * TILE_N] = (x_f[:TILE_M, TILE_N:2 * TILE_N] - y_f[:TILE_M, TILE_N:2 * TILE_N]).to(
        torch.float16
    )
    z_ref[TILE_M:, :TILE_N] = (x_f[TILE_M:, :TILE_N] - y_f[TILE_M:, :TILE_N]).to(torch.float16)
    z_ref[TILE_M:, TILE_N:2 * TILE_N] = (x_f[TILE_M:, TILE_N:2 * TILE_N] * y_f[TILE_M:, TILE_N:2 * TILE_N]).to(
        torch.float16
    )
    torch.testing.assert_close(z, z_ref, atol=1e-2, rtol=1e-2)
    logging.info("test_if_while_compound_mixed passed! shape=%s", shape)


# ===================================================================
# truthiness: if with non-bool scalar conditions (INT truthiness)
# ===================================================================


# =============================================================================
# Test 21: if 整数真值条件
#         if integer truthiness condition
# =============================================================================
@pl.jit(auto_mutex=True)
def if_truthiness_int_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    m = x.shape[0]
    n = x.shape[1]
    tile_type = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a_db = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0, 1])
    b_db = pl.make_tile_group(type=tile_type, addrs=0x4000, mutex_ids=[2, 3])
    c_db = pl.make_tile_group(type=tile_type, addrs=0x8000, mutex_ids=[30, 31])
    with pl.section_vector():
        for i in pl.range(0, m // TILE_M, 1):
            for j in pl.range(0, n // TILE_N, 1):
                tile_a = a_db.next()
                tile_b = b_db.next()
                tile_c = c_db.next()
                pl.load_tile(tile_a, x, [i, j])
                pl.load_tile(tile_b, y, [i, j])
                if i:
                    pl.add(tile_c, tile_a, tile_b)
                else:
                    pl.sub(tile_c, tile_a, tile_b)
                pl.store_tile(z, tile_c, [i, j])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_if_truthiness_int():
    device = ST_DEVICE
    torch.npu.set_device(device)
    x = torch.rand(128, 64, device=device, dtype=torch.float16)
    y = torch.rand(128, 64, device=device, dtype=torch.float16)
    z = torch.zeros(128, 64, device=device, dtype=torch.float16)
    if_truthiness_int_kernel(x, y, z)
    torch.npu.synchronize()
    z_ref = torch.zeros(128, 64, device=device, dtype=torch.float16)
    x_f = x.float()
    y_f = y.float()
    z_ref[TILE_M:, :] = (x_f[TILE_M:, :] + y_f[TILE_M:, :]).to(torch.float16)
    z_ref[:TILE_M, :] = (x_f[:TILE_M, :] - y_f[:TILE_M, :]).to(torch.float16)
    torch.testing.assert_close(z, z_ref, atol=1e-2, rtol=1e-2)
    logging.info("test_if_truthiness_int passed")


# ===================================================================
# if_in_operator: if x in (tuple) / pl.range() / closure list  (FP16)
# ===================================================================

_IN_CLOSURE_LIST = [0, 4]


# =============================================================================
# Test 22: in 操作 - 字面量 tuple / pl.range / 闭包列表
#         in operator - literal tuple / pl.range / closure list
# =============================================================================
@pl.jit(auto_mutex=True)
def if_in_operator_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    m = x.shape[0]
    n = x.shape[1]
    tile_type = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a_db = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0, 1])
    b_db = pl.make_tile_group(type=tile_type, addrs=0x4000, mutex_ids=[2, 3])
    c_db = pl.make_tile_group(type=tile_type, addrs=0x8000, mutex_ids=[30, 31])
    with pl.section_vector():
        num_m = m // TILE_M
        num_n = n // TILE_N
        for i in pl.range(0, num_m, 1):
            for j in pl.range(0, num_n, 1):
                tile_a = a_db.next()
                tile_b = b_db.next()
                tile_c = c_db.next()
                pl.load_tile(tile_a, x, [i, j])
                pl.load_tile(tile_b, y, [i, j])
                if i in (0, 2, 4):
                    pl.add(tile_c, tile_a, tile_b)
                elif i in pl.range(6, num_m, 2):
                    pl.add(tile_c, tile_a, tile_b)
                elif j in pl.range(2):
                    pl.add(tile_c, tile_a, tile_b)
                else:
                    pl.sub(tile_c, tile_a, tile_b)
                pl.store_tile(z, tile_c, [i, j])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_if_in_operator():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shape = [512, 512]
    x = torch.rand(shape, device=device, dtype=torch.float16)
    y = torch.rand(shape, device=device, dtype=torch.float16)
    z = torch.zeros(shape, device=device, dtype=torch.float16)
    if_in_operator_kernel(x, y, z)
    torch.npu.synchronize()
    x_f = x.float()
    y_f = y.float()
    z_ref = torch.zeros(shape, device=device, dtype=torch.float16)
    num_m = shape[0] // TILE_M
    num_n = shape[1] // TILE_N
    for ti in range(num_m):
        for tj in range(num_n):
            sl = (slice(ti * TILE_M, (ti + 1) * TILE_M), slice(tj * TILE_N, (tj + 1) * TILE_N))
            if ti in (0, 2, 4):
                z_ref[sl] = (x_f[sl] + y_f[sl]).to(torch.float16)
            elif ti in range(6, num_m, 2):
                z_ref[sl] = (x_f[sl] + y_f[sl]).to(torch.float16)
            elif tj in range(2):
                z_ref[sl] = (x_f[sl] + y_f[sl]).to(torch.float16)
            else:
                z_ref[sl] = (x_f[sl] - y_f[sl]).to(torch.float16)
    torch.testing.assert_close(z, z_ref, atol=1e-2, rtol=1e-2)
    logging.info("test_if_in_operator passed! shape=%s", shape)


# =============================================================================
# Test 23: not in 操作 - 字面量 / pl.range / 闭包列表
#         not-in operator - literal / pl.range / closure list
# =============================================================================
@pl.jit(auto_mutex=True)
def if_not_in_operator_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    m = x.shape[0]
    n = x.shape[1]
    tile_type = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a_db = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0, 1])
    b_db = pl.make_tile_group(type=tile_type, addrs=0x4000, mutex_ids=[2, 3])
    c_db = pl.make_tile_group(type=tile_type, addrs=0x8000, mutex_ids=[30, 31])
    with pl.section_vector():
        num_m = m // TILE_M
        num_n = n // TILE_N
        for i in pl.range(0, num_m, 1):
            for j in pl.range(0, num_n, 1):
                tile_a = a_db.next()
                tile_b = b_db.next()
                tile_c = c_db.next()
                pl.load_tile(tile_a, x, [i, j])
                pl.load_tile(tile_b, y, [i, j])
                if i not in (0, 1, 2):
                    pl.add(tile_c, tile_a, tile_b)
                elif i not in pl.range(0, num_m, 2):
                    pl.add(tile_c, tile_a, tile_b)
                elif i not in _IN_CLOSURE_LIST:
                    pl.add(tile_c, tile_a, tile_b)
                else:
                    pl.sub(tile_c, tile_a, tile_b)
                pl.store_tile(z, tile_c, [i, j])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_if_not_in_operator():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shape = [512, 512]
    x = torch.rand(shape, device=device, dtype=torch.float16)
    y = torch.rand(shape, device=device, dtype=torch.float16)
    z = torch.zeros(shape, device=device, dtype=torch.float16)
    if_not_in_operator_kernel(x, y, z)
    torch.npu.synchronize()
    x_f = x.float()
    y_f = y.float()
    z_ref = torch.zeros(shape, device=device, dtype=torch.float16)
    num_m = shape[0] // TILE_M
    num_n = shape[1] // TILE_N
    for ti in range(num_m):
        for tj in range(num_n):
            sl = (slice(ti * TILE_M, (ti + 1) * TILE_M), slice(tj * TILE_N, (tj + 1) * TILE_N))
            if ti not in (0, 1, 2):
                z_ref[sl] = (x_f[sl] + y_f[sl]).to(torch.float16)
            elif ti not in range(0, num_m, 2):
                z_ref[sl] = (x_f[sl] + y_f[sl]).to(torch.float16)
            elif ti not in _IN_CLOSURE_LIST:
                z_ref[sl] = (x_f[sl] + y_f[sl]).to(torch.float16)
            else:
                z_ref[sl] = (x_f[sl] - y_f[sl]).to(torch.float16)
    torch.testing.assert_close(z, z_ref, atol=1e-2, rtol=1e-2)
    logging.info("test_if_not_in_operator passed! shape=%s", shape)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    test_if_else()
    test_if_constexpr()
    test_if_elif_else()
    test_nested_if()
    test_if_relu()
    test_constexpr()
    test_if_always_false()
    test_deeply_nested_if()
    test_if_lt_const()
    test_if_ge_const_mul()
    test_if_jt_const()
    test_if_ij_cmp()
    test_if_elif_const()
    test_if_and()
    test_ternary_expr()
    test_if_compound_12()
    test_ternary_12()
    test_if_while_compound_mixed()
    test_if_truthiness_int()
    test_if_in_operator()
    test_if_not_in_operator()
    logging.info("\nAll tests passed!")
