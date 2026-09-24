# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""for control_flow 前端测试。

测试覆盖场景:
  1. 覆盖 for control_flow 场景的前端语法、编译入口与运行验证。
  2. 覆盖 dtype 泛化：DT_FP16/DT_BF16/DT_FP32/DT_INT32; Tensor 维度/shape 泛化：2D/4D/8D。
  3. 覆盖控制流组合：for 循环、if/elif/else 分支、with section 上下文、无值 return 提前退出。
  4. 覆盖接口组合：基础算术运算、激活/比较类运算、Tensor 与 Tile 数据搬运、section_vector/section_cube 上下文。
  5. 覆盖边界场景：pl.range 单层/多层遍历、step 步长、4D/8D shape 和非对齐尾块。
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
# for_add: 2D for-loop elementwise add  (FP16 / BF16 / FP32 / INT32)
# ===================================================================


# =============================================================================
# Test 1: for 循环二维加法 - FP16
#         for-loop 2D add - FP16
# =============================================================================
@pl.jit(auto_mutex=True)
def for_add_fp16_kernel(
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


# =============================================================================
# Test 2: for 循环二维加法 - BF16
#         for-loop 2D add - BF16
# =============================================================================
@pl.jit(auto_mutex=True)
def for_add_bf16_kernel(
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
                pl.add(tile_c, tile_a, tile_b)
                pl.store_tile(z, tile_c, [i, j])


# =============================================================================
# Test 3: for 循环二维加法 - FP32
#         for-loop 2D add - FP32
# =============================================================================
@pl.jit(auto_mutex=True)
def for_add_fp32_kernel(
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
                pl.add(tile_c, tile_a, tile_b)
                pl.store_tile(z, tile_c, [i, j])


# =============================================================================
# Test 4: for 循环二维加法 - INT32
#         for-loop 2D add - INT32
# =============================================================================
@pl.jit(auto_mutex=True)
def for_add_int32_kernel(
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
                pl.add(tile_c, tile_a, tile_b)
                pl.store_tile(z, tile_c, [i, j])


FOR_ADD_KERNELS = {
    "fp16": for_add_fp16_kernel,
    "bf16": for_add_bf16_kernel,
    "fp32": for_add_fp32_kernel,
    "int32": for_add_int32_kernel,
}


# ===================================================================
# for_range_step_ge_span: for-loop with pl.range(start, stop, step)
#   step >= stop - start, so only the start tile row executes.
# ===================================================================


# =============================================================================
# Test 5: pl.range step 等于跨度 - FP16
#         pl.range step equals span - FP16
# =============================================================================
@pl.jit(auto_mutex=True)
def for_range_step_eq_span_fp16_kernel(
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
        start = 1
        stop = m // TILE_M
        step = stop - start
        for i in pl.range(start, stop, step):
            for j in pl.range(0, n // TILE_N, 1):
                tile_a = a_db.next()
                tile_b = b_db.next()
                tile_c = c_db.next()
                pl.load_tile(tile_a, x, [i, j])
                pl.load_tile(tile_b, y, [i, j])
                pl.add(tile_c, tile_a, tile_b)
                pl.store_tile(z, tile_c, [i, j])


# =============================================================================
# Test 6: pl.range step 大于跨度 - FP16
#         pl.range step greater than span - FP16
# =============================================================================
@pl.jit(auto_mutex=True)
def for_range_step_gt_span_fp16_kernel(
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
        start = 1
        stop = m // TILE_M
        step = stop - start + 1
        for i in pl.range(start, stop, step):
            for j in pl.range(0, n // TILE_N, 1):
                tile_a = a_db.next()
                tile_b = b_db.next()
                tile_c = c_db.next()
                pl.load_tile(tile_a, x, [i, j])
                pl.load_tile(tile_b, y, [i, j])
                pl.add(tile_c, tile_a, tile_b)
                pl.store_tile(z, tile_c, [i, j])


FOR_RANGE_STEP_GE_SPAN_KERNELS = [
    ("eq_span", for_range_step_eq_span_fp16_kernel),
    ("gt_span", for_range_step_gt_span_fp16_kernel),
]


# ===================================================================
# for_4d_add: 4D nested for-loop elementwise add  (FP16)
# ===================================================================


# =============================================================================
# Test 7: 4D for 嵌套加法 - FP16
#         4D nested for add - FP16
# =============================================================================
@pl.jit(auto_mutex=True)
def for_4d_add_fp16_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    b = x.shape[0]
    h = x.shape[1]
    m = x.shape[2]
    n = x.shape[3]
    tile_type = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a_db = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0, 1])
    b_db = pl.make_tile_group(type=tile_type, addrs=0x4000, mutex_ids=[2, 3])
    c_db = pl.make_tile_group(type=tile_type, addrs=0x8000, mutex_ids=[30, 31])
    with pl.section_vector():
        for b_idx in pl.range(0, b, 1):
            for h_idx in pl.range(0, h, 1):
                for i in pl.range(0, m // TILE_M, 1):
                    for j in pl.range(0, n // TILE_N, 1):
                        tile_a = a_db.next()
                        tile_b = b_db.next()
                        tile_c = c_db.next()
                        pl.load_tile(tile_a, x, [b_idx, h_idx, i, j], order=[2, 3])
                        pl.load_tile(tile_b, y, [b_idx, h_idx, i, j], order=[2, 3])
                        pl.add(tile_c, tile_a, tile_b)
                        pl.store_tile(z, tile_c, [b_idx, h_idx, i, j], order=[2, 3])


FOR_4D_ADD_KERNELS = {
    "fp16": for_4d_add_fp16_kernel,
}


# ===================================================================
# for_4d_layout_add: 4D for-loop with M/N at different tensor axes (FP16)
# ===================================================================


# =============================================================================
# Test 8: 4D for shape M=2/3N 加法 - FP16
#         4D for add with shape M=2/3N - FP16
# =============================================================================
@pl.jit(auto_mutex=True)
def for_4d_add_m23n_fp16_kernel(
    x: pl.Tensor[[pl.DYNAMIC, 2, 3, pl.DYNAMIC], pl.DT_FP16],
    y: pl.Tensor[[pl.DYNAMIC, 2, 3, pl.DYNAMIC], pl.DT_FP16],
    z: pl.Tensor[[pl.DYNAMIC, 2, 3, pl.DYNAMIC], pl.DT_FP16],
):
    m = x.shape[0]
    n = x.shape[3]
    tile_type = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a_db = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0, 1])
    b_db = pl.make_tile_group(type=tile_type, addrs=0x4000, mutex_ids=[2, 3])
    c_db = pl.make_tile_group(type=tile_type, addrs=0x8000, mutex_ids=[30, 31])
    with pl.section_vector():
        for i in pl.range(0, m // TILE_M, 1):
            for d1 in pl.range(0, 2, 1):
                for d2 in pl.range(0, 3, 1):
                    for j in pl.range(0, n // TILE_N, 1):
                        tile_a = a_db.next()
                        tile_b = b_db.next()
                        tile_c = c_db.next()
                        pl.load_tile(tile_a, x, [i, d1, d2, j], order=[0, 3])
                        pl.load_tile(tile_b, y, [i, d1, d2, j], order=[0, 3])
                        pl.add(tile_c, tile_a, tile_b)
                        pl.store_tile(z, tile_c, [i, d1, d2, j], order=[0, 3])


# =============================================================================
# Test 9: 4D for shape 2M=3N 加法 - FP16
#         4D for add with shape 2M=3N - FP16
# =============================================================================
@pl.jit(auto_mutex=True)
def for_4d_add_2m3n_fp16_kernel(
    x: pl.Tensor[[2, pl.DYNAMIC, 3, pl.DYNAMIC], pl.DT_FP16],
    y: pl.Tensor[[2, pl.DYNAMIC, 3, pl.DYNAMIC], pl.DT_FP16],
    z: pl.Tensor[[2, pl.DYNAMIC, 3, pl.DYNAMIC], pl.DT_FP16],
):
    m = x.shape[1]
    n = x.shape[3]
    tile_type = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a_db = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0, 1])
    b_db = pl.make_tile_group(type=tile_type, addrs=0x4000, mutex_ids=[2, 3])
    c_db = pl.make_tile_group(type=tile_type, addrs=0x8000, mutex_ids=[30, 31])
    with pl.section_vector():
        for d0 in pl.range(0, 2, 1):
            for i in pl.range(0, m // TILE_M, 1):
                for d2 in pl.range(0, 3, 1):
                    for j in pl.range(0, n // TILE_N, 1):
                        tile_a = a_db.next()
                        tile_b = b_db.next()
                        tile_c = c_db.next()
                        pl.load_tile(tile_a, x, [d0, i, d2, j], order=[1, 3])
                        pl.load_tile(tile_b, y, [d0, i, d2, j], order=[1, 3])
                        pl.add(tile_c, tile_a, tile_b)
                        pl.store_tile(z, tile_c, [d0, i, d2, j], order=[1, 3])


FOR_4D_LAYOUT_ADD_KERNELS = [
    ("m23n", for_4d_add_m23n_fp16_kernel, [128, 2, 3, 64]),
    ("2m3n", for_4d_add_2m3n_fp16_kernel, [2, 128, 3, 64]),
]


# ===================================================================
# for_high_dim_add: 8D for-loop elementwise add  (FP16)
# ===================================================================


# =============================================================================
# Test 10: 8D for 嵌套加法 - FP16
#         8D nested for add - FP16
# =============================================================================
@pl.jit(auto_mutex=True)
def for_8d_add_fp16_kernel(
    x: pl.Tensor[[2, 2, 2, 2, 2, 2, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    y: pl.Tensor[[2, 2, 2, 2, 2, 2, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    z: pl.Tensor[[2, 2, 2, 2, 2, 2, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    m = x.shape[6]
    n = x.shape[7]
    tile_type = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a_db = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0, 1])
    b_db = pl.make_tile_group(type=tile_type, addrs=0x4000, mutex_ids=[2, 3])
    c_db = pl.make_tile_group(type=tile_type, addrs=0x8000, mutex_ids=[30, 31])
    with pl.section_vector():
        for d0 in pl.range(0, 2, 1):
            for d1 in pl.range(0, 2, 1):
                for d2 in pl.range(0, 2, 1):
                    for d3 in pl.range(0, 2, 1):
                        for d4 in pl.range(0, 2, 1):
                            for d5 in pl.range(0, 2, 1):
                                for i in pl.range(0, m // TILE_M, 1):
                                    for j in pl.range(0, n // TILE_N, 1):
                                        tile_a = a_db.next()
                                        tile_b = b_db.next()
                                        tile_c = c_db.next()
                                        pl.load_tile(tile_a, x, [d0, d1, d2, d3, d4, d5, i, j], order=[6, 7])
                                        pl.load_tile(tile_b, y, [d0, d1, d2, d3, d4, d5, i, j], order=[6, 7])
                                        pl.add(tile_c, tile_a, tile_b)
                                        pl.store_tile(z, tile_c, [d0, d1, d2, d3, d4, d5, i, j], order=[6, 7])


FOR_HIGH_DIM_ADD_KERNELS = [
    ("8d_tail", for_8d_add_fp16_kernel, [2, 2, 2, 2, 2, 2, 128, 128]),
]


# ===================================================================
# sub: 2D for-loop elementwise sub  (FP16)
# ===================================================================


# =============================================================================
# Test 11: for 循环减法 - FP16
#         for-loop subtraction - FP16
# =============================================================================
@pl.jit(auto_mutex=True)
def sub_fp16_kernel(
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
                pl.sub(tile_c, tile_a, tile_b)
                pl.store_tile(z, tile_c, [i, j])


SUB_KERNELS = {
    "fp16": sub_fp16_kernel,
}


# ===================================================================
# mul_add: 2D for-loop fused mul+add (z = x*y + x)  (FP16)
# ===================================================================


# =============================================================================
# Test 12: for 循环乘加融合 - FP16
#         for-loop fused multiply-add - FP16
# =============================================================================
@pl.jit(auto_mutex=True)
def mul_add_fp16_kernel(
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
                pl.mul(tile_c, tile_a, tile_b)
                pl.add(tile_c, tile_c, tile_a)
                pl.store_tile(z, tile_c, [i, j])


MUL_ADD_KERNELS = {
    "fp16": mul_add_fp16_kernel,
}


# ===================================================================
# Test functions
# ===================================================================


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_for_add():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shapes = [[128, 64], [192, 128], [256, 128], [128, 128]]
    for _pl_dt, tdt, label, atol, rtol in DTYPES_ALL:
        kernel = FOR_ADD_KERNELS[label]
        for shape in shapes:
            x, y, z = _gen(shape, tdt, device)
            kernel(x, y, z)
            torch.npu.synchronize()
            z_ref = _ref(tdt, lambda a, b: a + b, x, y)
            torch.testing.assert_close(z, z_ref, atol=atol, rtol=rtol)
            logging.info("test_for_add [%s] passed! shape=%s", label, shape)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_for_range_step_ge_span():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shapes = [[256, 256]]
    for case_name, kernel in FOR_RANGE_STEP_GE_SPAN_KERNELS:
        for shape in shapes:
            x, y, z = _gen(shape, torch.float16, device)
            kernel(x, y, z)
            torch.npu.synchronize()
            z_ref = torch.zeros(shape, device=device, dtype=torch.float16)
            z_ref[TILE_M:2 * TILE_M, :] = (x[TILE_M:2 * TILE_M, :].float() + y[TILE_M:2 * TILE_M, :].float()).to(
                torch.float16
            )
            torch.testing.assert_close(z, z_ref, atol=1e-2, rtol=1e-2)
            logging.info("test_for_range_step_ge_span [%s] passed! shape=%s", case_name, shape)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_for_4d_add():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shapes = [[2, 3, 128, 64]]
    for _pl_dt, tdt, label, atol, rtol in DTYPES_FP16:
        kernel = FOR_4D_ADD_KERNELS[label]
        for shape in shapes:
            x, y, z = _gen(shape, tdt, device)
            kernel(x, y, z)
            torch.npu.synchronize()
            z_ref = _ref(tdt, lambda a, b: a + b, x, y)
            torch.testing.assert_close(z, z_ref, atol=atol, rtol=rtol)
            logging.info("test_for_4d_add [%s] passed! shape=%s", label, shape)
    for case_name, kernel, shape in FOR_4D_LAYOUT_ADD_KERNELS:
        x, y, z = _gen(shape, torch.float16, device)
        kernel(x, y, z)
        torch.npu.synchronize()
        z_ref = (x.float() + y.float()).to(torch.float16)
        torch.testing.assert_close(z, z_ref, atol=1e-2, rtol=1e-2)
        logging.info("test_for_4d_add layout [%s] passed! shape=%s", case_name, shape)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_for_high_dim_add():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    for case_name, kernel, shape in FOR_HIGH_DIM_ADD_KERNELS:
        x, y, z = _gen(shape, torch.float16, device)
        kernel(x, y, z)
        torch.npu.synchronize()
        z_ref = (x.float() + y.float()).to(torch.float16)
        torch.testing.assert_close(z, z_ref, atol=1e-2, rtol=1e-2)
        logging.info("test_for_high_dim_add [%s] passed! shape=%s", case_name, shape)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_sub():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shapes = [[128, 64]]
    for _pl_dt, tdt, label, atol, rtol in DTYPES_FP16:
        kernel = SUB_KERNELS[label]
        for shape in shapes:
            x, y, z = _gen(shape, tdt, device)
            kernel(x, y, z)
            torch.npu.synchronize()
            z_ref = _ref(tdt, lambda a, b: a - b, x, y)
            torch.testing.assert_close(z, z_ref, atol=atol, rtol=rtol)
            logging.info("test_sub [%s] passed! shape=%s", label, shape)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_mul_add():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shapes = [[128, 64]]
    for _pl_dt, tdt, label, atol, rtol in DTYPES_FP16:
        kernel = MUL_ADD_KERNELS[label]
        for shape in shapes:
            x, y, z = _gen(shape, tdt, device)
            kernel(x, y, z)
            torch.npu.synchronize()
            z_ref = _ref(tdt, lambda a, b: a * b + a, x, y)
            torch.testing.assert_close(z, z_ref, atol=atol, rtol=rtol)
            logging.info("test_mul_add [%s] passed! shape=%s", label, shape)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_single_tile():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shapes = [[64, 64]]
    for _pl_dt, tdt, label, atol, rtol in DTYPES_FP16:
        kernel = FOR_ADD_KERNELS[label]
        for shape in shapes:
            x, y, z = _gen(shape, tdt, device)
            kernel(x, y, z)
            torch.npu.synchronize()
            z_ref = _ref(tdt, lambda a, b: a + b, x, y)
            torch.testing.assert_close(z, z_ref, atol=atol, rtol=rtol)
            logging.info("test_single_tile [%s] passed! shape=%s", label, shape)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_large_shape():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shapes = [[1024, 512]]
    for _pl_dt, tdt, label, atol, rtol in DTYPES_FP16:
        kernel = FOR_ADD_KERNELS[label]
        for shape in shapes:
            x, y, z = _gen(shape, tdt, device)
            kernel(x, y, z)
            torch.npu.synchronize()
            z_ref = _ref(tdt, lambda a, b: a + b, x, y)
            torch.testing.assert_close(z, z_ref, atol=atol, rtol=rtol)
            logging.info("test_large_shape [%s] passed! shape=%s", label, shape)


# ===================================================================
# residual_relu: residual connection with ReLU (FP16)
# ===================================================================


# =============================================================================
# Test 13: 残差加法 + ReLU - FP16
#         residual add + ReLU - FP16
# =============================================================================
@pl.jit(auto_mutex=True)
def residual_relu_fp16_kernel(
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
                pl.relu(tile_c, tile_c)
                pl.add(tile_c, tile_c, tile_a)
                pl.store_tile(z, tile_c, [i, j])


RESIDUAL_RELU_KERNELS = {
    "fp16": residual_relu_fp16_kernel,
}


# ===================================================================
# leaky_relu: LeakyReLU activation (FP16)
# ===================================================================


# =============================================================================
# Test 14: LeakyReLU 条件分支 - FP16
#         LeakyReLU conditional branch - FP16
# =============================================================================
@pl.jit(auto_mutex=True)
def leaky_relu_fp16_kernel(
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
    d_db = pl.make_tile_group(type=tile_type, addrs=0xC000, mutex_ids=[4, 5])
    with pl.section_vector():
        for i in pl.range(0, m // TILE_M, 1):
            for j in pl.range(0, n // TILE_N, 1):
                tile_a = a_db.next()
                tile_b = b_db.next()
                tile_c = c_db.next()
                tile_d = d_db.next()
                pl.load_tile(tile_a, x, [i, j])
                pl.load_tile(tile_b, y, [i, j])
                pl.muls(tile_d, tile_a, 0.01)
                pl.maximum(tile_c, tile_a, tile_d)
                pl.store_tile(z, tile_c, [i, j])


LEAKY_RELU_KERNELS = {
    "fp16": leaky_relu_fp16_kernel,
}


# ===================================================================
# fused_mul_add_relu: fused multiply-add with ReLU (FP16)
# ===================================================================


# =============================================================================
# Test 15: 乘加 + ReLU 融合 - FP16
#         fused multiply-add + ReLU - FP16
# =============================================================================
@pl.jit(auto_mutex=True)
def fused_mul_add_relu_fp16_kernel(
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
                pl.mul(tile_c, tile_a, tile_b)
                pl.add(tile_c, tile_c, tile_a)
                pl.relu(tile_c, tile_c)
                pl.store_tile(z, tile_c, [i, j])


FUSED_MUL_ADD_RELU_KERNELS = {
    "fp16": fused_mul_add_relu_fp16_kernel,
}


# ===================================================================
# three_way: three-way conditional branch (FP16)
# ===================================================================


# =============================================================================
# Test 16: 三路条件分支 - FP16
#         three-way conditional branch - FP16
# =============================================================================
@pl.jit(auto_mutex=True)
def three_way_fp16_kernel(
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
    d_db = pl.make_tile_group(type=tile_type, addrs=0xC000, mutex_ids=[4, 5])
    e_db = pl.make_tile_group(type=tile_type, addrs=0x10000, mutex_ids=[6, 7])
    with pl.section_vector():
        for i in pl.range(0, m // TILE_M, 1):
            for j in pl.range(0, n // TILE_N, 1):
                tile_a = a_db.next()
                tile_b = b_db.next()
                tile_c = c_db.next()
                tile_d = d_db.next()
                tile_e = e_db.next()
                pl.load_tile(tile_a, x, [i, j])
                pl.load_tile(tile_b, y, [i, j])
                pl.add(tile_c, tile_a, tile_b)
                pl.relu(tile_c, tile_c)
                pl.sub(tile_d, tile_a, tile_b)
                pl.relu(tile_d, tile_d)
                pl.sub(tile_e, tile_c, tile_d)
                pl.store_tile(z, tile_e, [i, j])


THREE_WAY_KERNELS = {
    "fp16": three_way_fp16_kernel,
}


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_residual_relu():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shapes = [[128, 64]]
    for _pl_dt, tdt, label, atol, rtol in DTYPES_FP16:
        kernel = RESIDUAL_RELU_KERNELS[label]
        for shape in shapes:
            x, y, z = _gen(shape, tdt, device)
            kernel(x, y, z)
            torch.npu.synchronize()
            z_ref = _ref(tdt, lambda a, b: torch.relu(a + b) + a, x, y)
            torch.testing.assert_close(z, z_ref, atol=atol, rtol=rtol)
            logging.info("test_residual_relu [%s] passed! shape=%s", label, shape)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_leaky_relu():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shape = [128, 64]
    for _pl_dt, tdt, label, atol, rtol in DTYPES_FP16:
        kernel = LEAKY_RELU_KERNELS[label]
        x, y, z = _gen(shape, tdt, device)
        kernel(x, y, z)
        torch.npu.synchronize()
        z_ref = _ref(tdt, lambda a, b: torch.maximum(a, 0.01 * a), x, y)
        torch.testing.assert_close(z, z_ref, atol=atol, rtol=rtol)
        logging.info("test_leaky_relu [%s] passed! shape=%s", label, shape)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_fused_mul_add_relu():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shape = [128, 64]
    for _pl_dt, tdt, label, atol, rtol in DTYPES_FP16:
        kernel = FUSED_MUL_ADD_RELU_KERNELS[label]
        x, y, z = _gen(shape, tdt, device)
        kernel(x, y, z)
        torch.npu.synchronize()
        z_ref = _ref(tdt, lambda a, b: torch.relu(a * b + a), x, y)
        torch.testing.assert_close(z, z_ref, atol=atol, rtol=rtol)
        logging.info("test_fused_mul_add_relu [%s] passed! shape=%s", label, shape)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_three_way():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shape = [128, 64]
    for _pl_dt, tdt, label, atol, rtol in DTYPES_FP16:
        kernel = THREE_WAY_KERNELS[label]
        x, y, z = _gen(shape, tdt, device)
        kernel(x, y, z)
        torch.npu.synchronize()
        z_ref = _ref(tdt, lambda a, b: torch.relu(a + b) - torch.relu(a - b), x, y)
        torch.testing.assert_close(z, z_ref, atol=atol, rtol=rtol)
        logging.info("test_three_way [%s] passed! shape=%s", label, shape)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_unaligned_shape():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shapes = [[97, 65]]

    # =============================================================================
    # Test 17: for 循环非对齐尾块 - FP16
    #         for-loop unaligned tail block - FP16
    # =============================================================================
    @pl.jit(auto_mutex=True)
    def for_unaligned_fp16_kernel(
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

    # =============================================================================
    # Test 18: for 循环非对齐尾块 - FP32
    #         for-loop unaligned tail block - FP32
    # =============================================================================
    @pl.jit(auto_mutex=True)
    def for_unaligned_fp32_kernel(
        x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
        y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
        z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    ):
        m = x.shape[0]
        n = x.shape[1]
        tile_type = pl.TileType(
            shape=[TILE_M, TILE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]
        )
        a_db = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0, 1])
        b_db = pl.make_tile_group(type=tile_type, addrs=0x8000, mutex_ids=[2, 3])
        c_db = pl.make_tile_group(type=tile_type, addrs=0x10000, mutex_ids=[30, 31])
        with pl.section_vector():
            for i in pl.range(0, m, TILE_M):
                for j in pl.range(0, n, TILE_N):
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

    # =============================================================================
    # Test 19: for 循环非对齐尾块 - BF16
    #         for-loop unaligned tail block - BF16
    # =============================================================================
    @pl.jit(auto_mutex=True)
    def for_unaligned_bf16_kernel(
        x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
        y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
        z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    ):
        m = x.shape[0]
        n = x.shape[1]
        tile_type = pl.TileType(
            shape=[TILE_M, TILE_N], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]
        )
        a_db = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0, 1])
        b_db = pl.make_tile_group(type=tile_type, addrs=0x4000, mutex_ids=[2, 3])
        c_db = pl.make_tile_group(type=tile_type, addrs=0x8000, mutex_ids=[30, 31])
        with pl.section_vector():
            for i in pl.range(0, m, TILE_M):
                for j in pl.range(0, n, TILE_N):
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

    # =============================================================================
    # Test 20: for 循环非对齐尾块 - INT32
    #         for-loop unaligned tail block - INT32
    # =============================================================================
    @pl.jit(auto_mutex=True)
    def for_unaligned_int32_kernel(
        x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT32],
        y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT32],
        z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT32],
    ):
        m = x.shape[0]
        n = x.shape[1]
        tile_type = pl.TileType(
            shape=[TILE_M, TILE_N], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]
        )
        a_db = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0, 1])
        b_db = pl.make_tile_group(type=tile_type, addrs=0x8000, mutex_ids=[2, 3])
        c_db = pl.make_tile_group(type=tile_type, addrs=0x10000, mutex_ids=[30, 31])
        with pl.section_vector():
            for i in pl.range(0, m, TILE_M):
                for j in pl.range(0, n, TILE_N):
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

    unaligned_kernels = {
        "fp16": (for_unaligned_fp16_kernel, torch.float16, 1e-2, 1e-2),
        "bf16": (for_unaligned_bf16_kernel, torch.bfloat16, 1e-2, 1e-2),
        "fp32": (for_unaligned_fp32_kernel, torch.float32, 1e-4, 1e-4),
        "int32": (for_unaligned_int32_kernel, torch.int32, 0, 0),
    }
    for label, (kernel, tdt, atol, rtol) in unaligned_kernels.items():
        for shape in shapes:
            if tdt == torch.int32:
                x = torch.randint(-100, 100, shape, device=device, dtype=tdt)
                y = torch.randint(-100, 100, shape, device=device, dtype=tdt)
            else:
                x = torch.rand(shape, device=device, dtype=tdt)
                y = torch.rand(shape, device=device, dtype=tdt)
            z = torch.zeros(shape, device=device, dtype=tdt)
            kernel(x, y, z)
            torch.npu.synchronize()
            z_ref = _ref(tdt, lambda a, b: a + b, x, y)
            torch.testing.assert_close(z, z_ref, atol=atol, rtol=rtol)
            logging.info("test_unaligned_shape [%s] passed! shape=%s", label, shape)


# ===================================================================
# for_range_one_arg: 2D for-loop with pl.range(stop) single-arg form (FP16)
# ===================================================================


# =============================================================================
# Test 21: for 循环 pl.range 单参数 - FP16
#         for-loop pl.range single argument - FP16
# =============================================================================
@pl.jit(auto_mutex=True)
def for_range_one_arg_fp16_kernel(
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
        num_m_tiles = (m + TILE_M - 1) // TILE_M
        num_n_tiles = (n + TILE_N - 1) // TILE_N
        for i in pl.range(num_m_tiles):
            for j in pl.range(num_n_tiles):
                tile_a = a_db.next()
                tile_b = b_db.next()
                tile_c = c_db.next()
                row_off = i * TILE_M
                col_off = j * TILE_N
                valid_r = pl.min(m - row_off, TILE_M)
                valid_c = pl.min(n - col_off, TILE_N)
                pl.set_validshape(tile_a, [valid_r, valid_c])
                pl.set_validshape(tile_b, [valid_r, valid_c])
                pl.set_validshape(tile_c, [valid_r, valid_c])
                pl.load(tile_a, x, [row_off, col_off])
                pl.load(tile_b, y, [row_off, col_off])
                pl.add(tile_c, tile_a, tile_b)
                pl.store(z, tile_c, [row_off, col_off])


# ===================================================================
# for_range_two_arg: 2D for-loop with pl.range(start, stop) two-arg form (FP16)
# ===================================================================


# =============================================================================
# Test 22: for 循环 pl.range 双参数 - FP16
#         for-loop pl.range two arguments - FP16
# =============================================================================
@pl.jit(auto_mutex=True)
def for_range_two_arg_fp16_kernel(
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
        num_m_tiles = (m + TILE_M - 1) // TILE_M
        num_n_tiles = (n + TILE_N - 1) // TILE_N
        for i in pl.range(0, num_m_tiles):
            for j in pl.range(0, num_n_tiles):
                tile_a = a_db.next()
                tile_b = b_db.next()
                tile_c = c_db.next()
                row_off = i * TILE_M
                col_off = j * TILE_N
                valid_r = pl.min(m - row_off, TILE_M)
                valid_c = pl.min(n - col_off, TILE_N)
                pl.set_validshape(tile_a, [valid_r, valid_c])
                pl.set_validshape(tile_b, [valid_r, valid_c])
                pl.set_validshape(tile_c, [valid_r, valid_c])
                pl.load(tile_a, x, [row_off, col_off])
                pl.load(tile_b, y, [row_off, col_off])
                pl.add(tile_c, tile_a, tile_b)
                pl.store(z, tile_c, [row_off, col_off])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_for_range_one_arg():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shape = [255, 255]
    x, y, z = _gen(shape, torch.float16, device)
    for_range_one_arg_fp16_kernel(x, y, z)
    torch.npu.synchronize()
    z_ref = _ref(torch.float16, lambda a, b: a + b, x, y)
    torch.testing.assert_close(z, z_ref, atol=1e-2, rtol=1e-2)
    logging.info("test_for_range_one_arg [fp16] passed! shape=%s", shape)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_for_range_two_arg():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shape = [255, 255]
    x, y, z = _gen(shape, torch.float16, device)
    for_range_two_arg_fp16_kernel(x, y, z)
    torch.npu.synchronize()
    z_ref = _ref(torch.float16, lambda a, b: a + b, x, y)
    torch.testing.assert_close(z, z_ref, atol=1e-2, rtol=1e-2)
    logging.info("test_for_range_two_arg [fp16] passed! shape=%s", shape)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    test_for_add()
    test_for_range_step_ge_span()
    test_for_4d_add()
    test_for_high_dim_add()
    test_sub()
    test_mul_add()
    test_single_tile()
    test_large_shape()
    test_residual_relu()
    test_leaky_relu()
    test_fused_mul_add_relu()
    test_three_way()
    test_unaligned_shape()
    test_for_range_one_arg()
    test_for_range_two_arg()
    logging.info("\nAll tests passed!")
