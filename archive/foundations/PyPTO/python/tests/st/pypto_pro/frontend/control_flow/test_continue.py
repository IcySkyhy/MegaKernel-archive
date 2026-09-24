# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""continue control_flow 前端测试。

测试覆盖场景:
  1. 覆盖 continue control_flow 场景的前端语法、编译入口与运行验证。
  2. 覆盖 dtype 泛化：DT_FP16/DT_BF16/DT_FP32/DT_INT32; Tensor 维度/shape 泛化：2D。
  3. 覆盖控制流组合：for 循环、while 循环、if/elif/else 分支、with section 上下文、continue 跳转、无值 return 提前退出。
  4. 覆盖接口组合：基础算术运算、激活/比较类运算、Tensor 与 Tile 数据搬运、section_vector/section_cube 上下文。
  5. 覆盖边界场景：for/while 内的中间 continue、末尾 continue、多层嵌套 continue 和非对齐尾块。
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
# for_continue: for-loop add, skip j==0 (only later tile cols)  (FP16/BF16/FP32/INT32)
# ===================================================================


# =============================================================================
# Test 1: for 循环 continue 跳过 - FP16
#         for-loop continue - FP16
# =============================================================================
@pl.jit(auto_mutex=True)
def for_continue_fp16_kernel(
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
                if j == 0:
                    continue
                tile_a = a_db.next()
                tile_b = b_db.next()
                tile_c = c_db.next()
                pl.load_tile(tile_a, x, [i, j])
                pl.load_tile(tile_b, y, [i, j])
                pl.add(tile_c, tile_a, tile_b)
                pl.store_tile(z, tile_c, [i, j])


# =============================================================================
# Test 2: for 循环 continue 跳过 - BF16
#         for-loop continue - BF16
# =============================================================================
@pl.jit(auto_mutex=True)
def for_continue_bf16_kernel(
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
                if j == 0:
                    continue
                tile_a = a_db.next()
                tile_b = b_db.next()
                tile_c = c_db.next()
                pl.load_tile(tile_a, x, [i, j])
                pl.load_tile(tile_b, y, [i, j])
                pl.add(tile_c, tile_a, tile_b)
                pl.store_tile(z, tile_c, [i, j])


# =============================================================================
# Test 3: for 循环 continue 跳过 - FP32
#         for-loop continue - FP32
# =============================================================================
@pl.jit(auto_mutex=True)
def for_continue_fp32_kernel(
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
                if j == 0:
                    continue
                tile_a = a_db.next()
                tile_b = b_db.next()
                tile_c = c_db.next()
                pl.load_tile(tile_a, x, [i, j])
                pl.load_tile(tile_b, y, [i, j])
                pl.add(tile_c, tile_a, tile_b)
                pl.store_tile(z, tile_c, [i, j])


# =============================================================================
# Test 4: for 循环 continue 跳过 - INT32
#         for-loop continue - INT32
# =============================================================================
@pl.jit(auto_mutex=True)
def for_continue_int32_kernel(
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
                if j == 0:
                    continue
                tile_a = a_db.next()
                tile_b = b_db.next()
                tile_c = c_db.next()
                pl.load_tile(tile_a, x, [i, j])
                pl.load_tile(tile_b, y, [i, j])
                pl.add(tile_c, tile_a, tile_b)
                pl.store_tile(z, tile_c, [i, j])


FOR_CONTINUE_KERNELS = {
    "fp16": for_continue_fp16_kernel,
    "bf16": for_continue_bf16_kernel,
    "fp32": for_continue_fp32_kernel,
    "int32": for_continue_int32_kernel,
}


# ===================================================================
# while_continue: while-loop add, skip j==0 (only later tile cols)  (FP16)
# ===================================================================


# =============================================================================
# Test 5: while 循环 continue 跳过 - FP16
#         while-loop continue - FP16
# =============================================================================
@pl.jit(auto_mutex=True)
def while_continue_fp16_kernel(
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
        i: pl.DT_INT64 = 0
        while i < m // TILE_M:
            j: pl.DT_INT64 = 0
            while j < n // TILE_N:
                if j == 0:
                    j = j + 1
                    continue
                tile_a = a_db.next()
                tile_b = b_db.next()
                tile_c = c_db.next()
                pl.load_tile(tile_a, x, [i, j])
                pl.load_tile(tile_b, y, [i, j])
                pl.add(tile_c, tile_a, tile_b)
                pl.store_tile(z, tile_c, [i, j])
                j = j + 1
            i = i + 1


WHILE_CONTINUE_KERNELS = {
    "fp16": while_continue_fp16_kernel,
}


# ===================================================================
# for_while_continue: outer for + inner while, skip j==0  (FP16)
# ===================================================================


# =============================================================================
# Test 6: for + while 嵌套 continue - FP16
#         nested for + while continue - FP16
# =============================================================================
@pl.jit(auto_mutex=True)
def for_while_continue_fp16_kernel(
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
            while j < n // TILE_N:
                if j == 0:
                    j = j + 1
                    continue
                tile_a = a_db.next()
                tile_b = b_db.next()
                tile_c = c_db.next()
                pl.load_tile(tile_a, x, [i, j])
                pl.load_tile(tile_b, y, [i, j])
                pl.add(tile_c, tile_a, tile_b)
                pl.store_tile(z, tile_c, [i, j])
                j = j + 1


FOR_WHILE_CONTINUE_KERNELS = {
    "fp16": for_while_continue_fp16_kernel,
}


# ===================================================================
# for_3layer_continue: 3-layer for with continue at all 3 layers
#   L1 outer: skip i==0
#   L2 mid:   skip j==0
#   L3 inner: skip k==0 (k range 0..1, so only k==1 executes)
#   (innermost k loop is a dummy single-iteration loop to test 3-layer continue)
# ===================================================================


# =============================================================================
# Test 7: for 3 层嵌套 continue - FP16
#         3-layer nested for continue - FP16
# =============================================================================
@pl.jit(auto_mutex=True)
def for_3layer_continue_fp16_kernel(
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
            if i == 0:
                continue
            for j in pl.range(0, n // TILE_N, 1):
                if j == 0:
                    continue
                for k in pl.range(0, 2, 1):
                    if k == 0:
                        continue
                    tile_a = a_db.next()
                    tile_b = b_db.next()
                    tile_c = c_db.next()
                    pl.load_tile(tile_a, x, [i, j])
                    pl.load_tile(tile_b, y, [i, j])
                    pl.add(tile_c, tile_a, tile_b)
                    pl.store_tile(z, tile_c, [i, j])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_for_continue():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shapes = [
        [128, 128],
        [256, 128],
        [128, 256],
        [256, 256],
        [192, 128],
        [512, 256],
        [256, 512],
        [512, 512],
        [384, 256],
    ]
    for _pl_dt, tdt, label, atol, rtol in DTYPES_ALL:
        kernel = FOR_CONTINUE_KERNELS[label]
        for shape in shapes:
            x, y, z = _gen(shape, tdt, device)
            kernel(x, y, z)
            torch.npu.synchronize()
            z_ref = torch.zeros(shape, device=device, dtype=tdt)
            z_ref[:, TILE_N:] = _ref(tdt, lambda a, b: a + b, x[:, TILE_N:], y[:, TILE_N:])
            torch.testing.assert_close(z, z_ref, atol=atol, rtol=rtol)
            logging.info("test_for_continue [%s] passed! shape=%s", label, shape)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_while_continue():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shapes = [[256, 256]]
    for _pl_dt, tdt, label, atol, rtol in DTYPES_FP16:
        kernel = WHILE_CONTINUE_KERNELS[label]
        for shape in shapes:
            x, y, z = _gen(shape, tdt, device)
            kernel(x, y, z)
            torch.npu.synchronize()
            z_ref = torch.zeros(shape, device=device, dtype=tdt)
            z_ref[:, TILE_N:] = _ref(tdt, lambda a, b: a + b, x[:, TILE_N:], y[:, TILE_N:])
            torch.testing.assert_close(z, z_ref, atol=atol, rtol=rtol)
            logging.info("test_while_continue [%s] passed! shape=%s", label, shape)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_for_while_continue():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shapes = [[256, 256]]
    for _pl_dt, tdt, label, atol, rtol in DTYPES_FP16:
        kernel = FOR_WHILE_CONTINUE_KERNELS[label]
        for shape in shapes:
            x, y, z = _gen(shape, tdt, device)
            kernel(x, y, z)
            torch.npu.synchronize()
            z_ref = torch.zeros(shape, device=device, dtype=tdt)
            z_ref[:, TILE_N:] = _ref(tdt, lambda a, b: a + b, x[:, TILE_N:], y[:, TILE_N:])
            torch.testing.assert_close(z, z_ref, atol=atol, rtol=rtol)
            logging.info("test_for_while_continue [%s] passed! shape=%s", label, shape)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_for_3layer_continue():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shapes = [[256, 256]]
    for shape in shapes:
        x = torch.rand(shape, device=device, dtype=torch.float16)
        y = torch.rand(shape, device=device, dtype=torch.float16)
        z = torch.zeros(shape, device=device, dtype=torch.float16)
        for_3layer_continue_fp16_kernel(x, y, z)
        torch.npu.synchronize()
        z_ref = torch.zeros(shape, device=device, dtype=torch.float16)
        z_ref[TILE_M:, TILE_N:] = (x[TILE_M:, TILE_N:].float() + y[TILE_M:, TILE_N:].float()).to(torch.float16)
        torch.testing.assert_close(z, z_ref, atol=1e-2, rtol=1e-2)
        logging.info("test_for_3layer_continue passed! shape=%s", shape)


# =============================================================================
# Test 8: continue 非对齐尾块 - FP16
#         continue with unaligned tail block - FP16
# =============================================================================
@pl.jit(auto_mutex=True)
def continue_unaligned_fp16_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    m = x.shape[0]
    n = x.shape[1]
    tile_type = pl.TileType(
        shape=[TILE_M, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]
    )
    a_db = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0, 1])
    b_db = pl.make_tile_group(type=tile_type, addrs=0x4000, mutex_ids=[2, 3])
    c_db = pl.make_tile_group(type=tile_type, addrs=0x8000, mutex_ids=[30, 31])
    with pl.section_vector():
        for i in pl.range(0, m, TILE_M):
            for j in pl.range(0, n, TILE_N):
                if j == 0:
                    continue
                tile_a = a_db.next()
                tile_b = b_db.next()
                tile_c = c_db.next()
                valid_r = pl.min(m - i, TILE_M)
                valid_c = pl.min(n - j, TILE_N)
                pl.set_validshape(tile_a, [valid_r, valid_c])
                pl.set_validshape(tile_b, [valid_r, valid_c])
                pl.set_validshape(tile_c, [valid_r, valid_c])
                pl.load(tile_a, x, [i, j])
                pl.load(tile_b, y, [i, j])
                pl.add(tile_c, tile_a, tile_b)
                pl.store(z, tile_c, [i, j])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_continue_unaligned_shape():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shapes = [[97, 65]]
    for shape in shapes:
        x = torch.rand(shape, device=device, dtype=torch.float16)
        y = torch.rand(shape, device=device, dtype=torch.float16)
        z = torch.zeros(shape, device=device, dtype=torch.float16)
        continue_unaligned_fp16_kernel(x, y, z)
        torch.npu.synchronize()
        z_ref = torch.zeros(shape, device=device, dtype=torch.float16)
        z_ref[:, TILE_N:] = (x[:, TILE_N:].float() + y[:, TILE_N:].float()).to(torch.float16)
        torch.testing.assert_close(z, z_ref, atol=1e-2, rtol=1e-2)
        logging.info("test_continue_unaligned_shape passed! shape=%s", shape)


# ===================================================================
# for_continue_mid: continue in the MIDDLE of inner for body
#   load → if j >= 1: continue → add → store (j>=1 skips add/store)
# ===================================================================


# =============================================================================
# Test 9: for 循环中段 continue - FP16
#         for-loop mid-body continue - FP16
# =============================================================================
@pl.jit(auto_mutex=True)
def for_continue_mid_fp16_kernel(
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
                pl.system.bar_mte2()
                pl.load_tile(tile_a, x, [i, j])
                pl.load_tile(tile_b, y, [i, j])
                if j >= 1:
                    continue
                pl.add(tile_c, tile_a, tile_b)
                pl.store_tile(z, tile_c, [i, j])


# ===================================================================
# for_continue_end: continue at the END of inner for body
#   load → add → store → if j >= 1: continue (no-op effect, but tests
#   continue placement at end of body before the implicit loop-back)
# ===================================================================


# =============================================================================
# Test 10: for 循环末段 continue - FP16
#         for-loop end-body continue - FP16
# =============================================================================
@pl.jit(auto_mutex=True)
def for_continue_end_fp16_kernel(
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
                pl.add(tile_c, tile_a, tile_b)
                pl.store_tile(z, tile_c, [i, j])
                if j >= 1:
                    continue


# ===================================================================
# while_continue_mid: continue in the MIDDLE of inner while body
#   load → if j >= 1: j+=1; continue → add → store (j>=1 skips add/store)
# ===================================================================


# =============================================================================
# Test 11: while 循环中段 continue - FP16
#         while-loop mid-body continue - FP16
# =============================================================================
@pl.jit(auto_mutex=True)
def while_continue_mid_fp16_kernel(
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
        i: pl.DT_INT64 = 0
        while i < m // TILE_M:
            j: pl.DT_INT64 = 0
            while j < n // TILE_N:
                tile_a = a_db.next()
                tile_b = b_db.next()
                tile_c = c_db.next()
                pl.system.bar_mte2()
                pl.load_tile(tile_a, x, [i, j])
                pl.load_tile(tile_b, y, [i, j])
                if j >= 1:
                    j = j + 1
                    continue
                pl.add(tile_c, tile_a, tile_b)
                pl.store_tile(z, tile_c, [i, j])
                j = j + 1
            i = i + 1


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_for_continue_mid():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shapes = [[256, 256]]
    for shape in shapes:
        x = torch.rand(shape, device=device, dtype=torch.float16)
        y = torch.rand(shape, device=device, dtype=torch.float16)
        z = torch.zeros(shape, device=device, dtype=torch.float16)
        for_continue_mid_fp16_kernel(x, y, z)
        torch.npu.synchronize()
        z_ref = torch.zeros(shape, device=device, dtype=torch.float16)
        z_ref[:, :TILE_N] = (x[:, :TILE_N].float() + y[:, :TILE_N].float()).to(torch.float16)
        torch.testing.assert_close(z, z_ref, atol=1e-2, rtol=1e-2)
        logging.info("test_for_continue_mid passed! shape=%s", shape)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_for_continue_end():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shapes = [[256, 256]]
    for shape in shapes:
        x = torch.rand(shape, device=device, dtype=torch.float16)
        y = torch.rand(shape, device=device, dtype=torch.float16)
        z = torch.zeros(shape, device=device, dtype=torch.float16)
        for_continue_end_fp16_kernel(x, y, z)
        torch.npu.synchronize()
        z_ref = (x.float() + y.float()).to(torch.float16)
        torch.testing.assert_close(z, z_ref, atol=1e-2, rtol=1e-2)
        logging.info("test_for_continue_end passed! shape=%s", shape)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_while_continue_mid():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shapes = [[256, 256]]
    for shape in shapes:
        x = torch.rand(shape, device=device, dtype=torch.float16)
        y = torch.rand(shape, device=device, dtype=torch.float16)
        z = torch.zeros(shape, device=device, dtype=torch.float16)
        while_continue_mid_fp16_kernel(x, y, z)
        torch.npu.synchronize()
        z_ref = torch.zeros(shape, device=device, dtype=torch.float16)
        z_ref[:, :TILE_N] = (x[:, :TILE_N].float() + y[:, :TILE_N].float()).to(torch.float16)
        torch.testing.assert_close(z, z_ref, atol=1e-2, rtol=1e-2)
        logging.info("test_while_continue_mid passed! shape=%s", shape)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    test_for_continue()
    test_while_continue()
    test_for_while_continue()
    test_for_3layer_continue()
    test_continue_unaligned_shape()
    test_for_continue_mid()
    test_for_continue_end()
    test_while_continue_mid()
    logging.info("\nAll tests passed!")
