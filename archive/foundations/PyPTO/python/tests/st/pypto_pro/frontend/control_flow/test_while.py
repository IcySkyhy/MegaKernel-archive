# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""while control_flow 前端测试。

测试覆盖场景:
  1. 覆盖 while control_flow 场景的前端语法、编译入口与运行验证。
  2. 覆盖 dtype 泛化：DT_FP16/DT_BF16/DT_FP32/DT_INT32; Tensor 维度/shape 泛化：2D/4D/8D。
  3. 覆盖控制流组合：for 循环、while 循环、if/elif/else 分支、with section 上下文、无值 return 提前退出。
  4. 覆盖接口组合：基础算术运算、激活/比较类运算、Tensor 与 Tile 数据搬运、section_vector/section_cube 上下文。
  5. 覆盖边界场景：while 条件更新、布尔/三元条件、4D/8D shape 和非对齐尾块。
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
# while_add: 2D while-loop elementwise add  (FP16 / BF16 / FP32 / INT32)
# ===================================================================


# =============================================================================
# Test 1: while 循环二维加法 - FP16
#         while-loop 2D add - FP16
# =============================================================================
@pl.jit(auto_mutex=True)
def while_add_fp16_kernel(
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
                pl.load_tile(tile_a, x, [i, j])
                pl.load_tile(tile_b, y, [i, j])
                pl.add(tile_c, tile_a, tile_b)
                pl.store_tile(z, tile_c, [i, j])
                j = j + 1
            i = i + 1


# =============================================================================
# Test 2: while 循环二维加法 - BF16
#         while-loop 2D add - BF16
# =============================================================================
@pl.jit(auto_mutex=True)
def while_add_bf16_kernel(
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
        i: pl.DT_INT64 = 0
        while i < m // TILE_M:
            j: pl.DT_INT64 = 0
            while j < n // TILE_N:
                tile_a = a_db.next()
                tile_b = b_db.next()
                tile_c = c_db.next()
                pl.load_tile(tile_a, x, [i, j])
                pl.load_tile(tile_b, y, [i, j])
                pl.add(tile_c, tile_a, tile_b)
                pl.store_tile(z, tile_c, [i, j])
                j = j + 1
            i = i + 1


# =============================================================================
# Test 3: while 循环二维加法 - FP32
#         while-loop 2D add - FP32
# =============================================================================
@pl.jit(auto_mutex=True)
def while_add_fp32_kernel(
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
        i: pl.DT_INT64 = 0
        while i < m // TILE_M:
            j: pl.DT_INT64 = 0
            while j < n // TILE_N:
                tile_a = a_db.next()
                tile_b = b_db.next()
                tile_c = c_db.next()
                pl.load_tile(tile_a, x, [i, j])
                pl.load_tile(tile_b, y, [i, j])
                pl.add(tile_c, tile_a, tile_b)
                pl.store_tile(z, tile_c, [i, j])
                j = j + 1
            i = i + 1


# =============================================================================
# Test 4: while 循环二维加法 - INT32
#         while-loop 2D add - INT32
# =============================================================================
@pl.jit(auto_mutex=True)
def while_add_int32_kernel(
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
        i: pl.DT_INT64 = 0
        while i < m // TILE_M:
            j: pl.DT_INT64 = 0
            while j < n // TILE_N:
                tile_a = a_db.next()
                tile_b = b_db.next()
                tile_c = c_db.next()
                pl.load_tile(tile_a, x, [i, j])
                pl.load_tile(tile_b, y, [i, j])
                pl.add(tile_c, tile_a, tile_b)
                pl.store_tile(z, tile_c, [i, j])
                j = j + 1
            i = i + 1


WHILE_ADD_KERNELS = {
    "fp16": while_add_fp16_kernel,
    "bf16": while_add_bf16_kernel,
    "fp32": while_add_fp32_kernel,
    "int32": while_add_int32_kernel,
}


# ===================================================================
# while_sub: 2D while-loop elementwise sub  (FP16)
# ===================================================================


# =============================================================================
# Test 5: while 循环减法 - FP16
#         while-loop subtraction - FP16
# =============================================================================
@pl.jit(auto_mutex=True)
def while_sub_fp16_kernel(
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
                pl.load_tile(tile_a, x, [i, j])
                pl.load_tile(tile_b, y, [i, j])
                pl.sub(tile_c, tile_a, tile_b)
                pl.store_tile(z, tile_c, [i, j])
                j = j + 1
            i = i + 1


WHILE_SUB_KERNELS = {
    "fp16": while_sub_fp16_kernel,
}


# ===================================================================
# while_mul: 2D while-loop elementwise mul  (FP16)
# ===================================================================


# =============================================================================
# Test 6: while 循环乘法 - FP16
#         while-loop multiply - FP16
# =============================================================================
@pl.jit(auto_mutex=True)
def while_mul_fp16_kernel(
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
                pl.load_tile(tile_a, x, [i, j])
                pl.load_tile(tile_b, y, [i, j])
                pl.mul(tile_c, tile_a, tile_b)
                pl.store_tile(z, tile_c, [i, j])
                j = j + 1
            i = i + 1


WHILE_MUL_KERNELS = {
    "fp16": while_mul_fp16_kernel,
}


# ===================================================================
# while_mul_add: 2D while-loop fused mul+add (z = x*y + x)  (FP16)
# ===================================================================


# =============================================================================
# Test 7: while 循环乘加融合 - FP16
#         while-loop fused multiply-add - FP16
# =============================================================================
@pl.jit(auto_mutex=True)
def while_mul_add_fp16_kernel(
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
                pl.load_tile(tile_a, x, [i, j])
                pl.load_tile(tile_b, y, [i, j])
                pl.mul(tile_c, tile_a, tile_b)
                pl.add(tile_c, tile_c, tile_a)
                pl.store_tile(z, tile_c, [i, j])
                j = j + 1
            i = i + 1


WHILE_MUL_ADD_KERNELS = {
    "fp16": while_mul_add_fp16_kernel,
}


# ===================================================================
# while_tail: 2D while-loop with tail block (set_validshape, compact=1)
# ===================================================================


# =============================================================================
# Test 8: while 循环尾块处理 - FP16
#         while-loop tail block handling - FP16
# =============================================================================
@pl.jit(auto_mutex=True)
def while_tail_fp16_kernel(
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
        i: pl.DT_INT64 = 0
        while i * TILE_M < m:
            j: pl.DT_INT64 = 0
            while j * TILE_N < n:
                tile_a = a_db.next()
                tile_b = b_db.next()
                tile_c = c_db.next()
                valid_r = pl.min(m - i * TILE_M, TILE_M)
                valid_c = pl.min(n - j * TILE_N, TILE_N)
                pl.set_validshape(tile_a, [valid_r, valid_c])
                pl.set_validshape(tile_b, [valid_r, valid_c])
                pl.set_validshape(tile_c, [valid_r, valid_c])
                pl.load(tile_a, x, [i * TILE_M, j * TILE_N])
                pl.load(tile_b, y, [i * TILE_M, j * TILE_N])
                pl.add(tile_c, tile_a, tile_b)
                pl.store(z, tile_c, [i * TILE_M, j * TILE_N])
                j = j + 1
            i = i + 1


# ===================================================================
# while_4d_add: 4D while-loop elementwise add  (FP16)
# ===================================================================


# =============================================================================
# Test 9: 4D while 嵌套加法 - FP16
#         4D nested while add - FP16
# =============================================================================
@pl.jit(auto_mutex=True)
def while_4d_add_fp16_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    b = x.shape[0]
    d = x.shape[1]
    m = x.shape[2]
    n = x.shape[3]
    tile_type = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a_db = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0, 1])
    b_db = pl.make_tile_group(type=tile_type, addrs=0x4000, mutex_ids=[2, 3])
    c_db = pl.make_tile_group(type=tile_type, addrs=0x8000, mutex_ids=[30, 31])
    with pl.section_vector():
        b0: pl.DT_INT64 = 0
        while b0 < b:
            b1: pl.DT_INT64 = 0
            while b1 < d:
                i: pl.DT_INT64 = 0
                while i < m // TILE_M:
                    j: pl.DT_INT64 = 0
                    while j < n // TILE_N:
                        tile_a = a_db.next()
                        tile_b = b_db.next()
                        tile_c = c_db.next()
                        pl.load_tile(tile_a, x, [b0, b1, i, j], order=[2, 3])
                        pl.load_tile(tile_b, y, [b0, b1, i, j], order=[2, 3])
                        pl.add(tile_c, tile_a, tile_b)
                        pl.store_tile(z, tile_c, [b0, b1, i, j], order=[2, 3])
                        j = j + 1
                    i = i + 1
                b1 = b1 + 1
            b0 = b0 + 1


WHILE_4D_ADD_KERNELS = {
    "fp16": while_4d_add_fp16_kernel,
}


# ===================================================================
# while_4d_layout_add: 4D while-loop with M/N at different tensor axes (FP16)
# ===================================================================


# =============================================================================
# Test 10: 4D while shape M=2/3N 加法 - FP16
#         4D while add with shape M=2/3N - FP16
# =============================================================================
@pl.jit(auto_mutex=True)
def while_4d_add_m23n_fp16_kernel(
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
        i: pl.DT_INT64 = 0
        while i < m // TILE_M:
            for d1 in pl.range(0, 2, 1):
                for d2 in pl.range(0, 3, 1):
                    j: pl.DT_INT64 = 0
                    while j < n // TILE_N:
                        tile_a = a_db.next()
                        tile_b = b_db.next()
                        tile_c = c_db.next()
                        pl.load_tile(tile_a, x, [i, d1, d2, j], order=[0, 3])
                        pl.load_tile(tile_b, y, [i, d1, d2, j], order=[0, 3])
                        pl.add(tile_c, tile_a, tile_b)
                        pl.store_tile(z, tile_c, [i, d1, d2, j], order=[0, 3])
                        j = j + 1
            i = i + 1


# =============================================================================
# Test 11: 4D while shape 2M=3N 加法 - FP16
#         4D while add with shape 2M=3N - FP16
# =============================================================================
@pl.jit(auto_mutex=True)
def while_4d_add_2m3n_fp16_kernel(
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
            i: pl.DT_INT64 = 0
            while i < m // TILE_M:
                for d2 in pl.range(0, 3, 1):
                    j: pl.DT_INT64 = 0
                    while j < n // TILE_N:
                        tile_a = a_db.next()
                        tile_b = b_db.next()
                        tile_c = c_db.next()
                        pl.load_tile(tile_a, x, [d0, i, d2, j], order=[1, 3])
                        pl.load_tile(tile_b, y, [d0, i, d2, j], order=[1, 3])
                        pl.add(tile_c, tile_a, tile_b)
                        pl.store_tile(z, tile_c, [d0, i, d2, j], order=[1, 3])
                        j = j + 1
                i = i + 1


WHILE_4D_LAYOUT_ADD_KERNELS = [
    ("m23n", while_4d_add_m23n_fp16_kernel, [128, 2, 3, 64]),
    ("2m3n", while_4d_add_2m3n_fp16_kernel, [2, 128, 3, 64]),
]


# ===================================================================
# while_high_dim_add: 8D while-loop elementwise add  (FP16)
# ===================================================================


# =============================================================================
# Test 12: 8D while 嵌套加法 - FP16
#         8D nested while add - FP16
# =============================================================================
@pl.jit(auto_mutex=True)
def while_8d_add_fp16_kernel(
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
                                i: pl.DT_INT64 = 0
                                while i < m // TILE_M:
                                    j: pl.DT_INT64 = 0
                                    while j < n // TILE_N:
                                        tile_a = a_db.next()
                                        tile_b = b_db.next()
                                        tile_c = c_db.next()
                                        pl.load_tile(tile_a, x, [d0, d1, d2, d3, d4, d5, i, j], order=[6, 7])
                                        pl.load_tile(tile_b, y, [d0, d1, d2, d3, d4, d5, i, j], order=[6, 7])
                                        pl.add(tile_c, tile_a, tile_b)
                                        pl.store_tile(z, tile_c, [d0, d1, d2, d3, d4, d5, i, j], order=[6, 7])
                                        j = j + 1
                                    i = i + 1


WHILE_HIGH_DIM_ADD_KERNELS = [
    ("8d_tail", while_8d_add_fp16_kernel, [2, 2, 2, 2, 2, 2, 128, 128]),
]


# ===================================================================
# Test functions
# ===================================================================


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_while_add():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shapes = [[64, 64], [128, 64], [192, 128], [256, 128], [256, 256]]
    for _pl_dt, tdt, label, atol, rtol in DTYPES_ALL:
        kernel = WHILE_ADD_KERNELS[label]
        for shape in shapes:
            x, y, z = _gen(shape, tdt, device)
            kernel(x, y, z)
            torch.npu.synchronize()
            z_ref = _ref(tdt, lambda a, b: a + b, x, y)
            torch.testing.assert_close(z, z_ref, atol=atol, rtol=rtol)
            logging.info("test_while_add [%s] passed! shape=%s", label, shape)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_while_shape_generalization():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shapes = [[256, 256]]
    for _pl_dt, tdt, label, atol, rtol in DTYPES_FP16:
        kernel = WHILE_ADD_KERNELS[label]
        for shape in shapes:
            x, y, z = _gen(shape, tdt, device)
            kernel(x, y, z)
            torch.npu.synchronize()
            z_ref = _ref(tdt, lambda a, b: a + b, x, y)
            torch.testing.assert_close(z, z_ref, atol=atol, rtol=rtol)
            logging.info("test_while_shape_generalization [%s] passed! shape=%s", label, shape)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_while_sub():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shapes = [[64, 64]]
    for _pl_dt, tdt, label, atol, rtol in DTYPES_FP16:
        kernel = WHILE_SUB_KERNELS[label]
        for shape in shapes:
            x, y, z = _gen(shape, tdt, device)
            kernel(x, y, z)
            torch.npu.synchronize()
            z_ref = _ref(tdt, lambda a, b: a - b, x, y)
            torch.testing.assert_close(z, z_ref, atol=atol, rtol=rtol)
            logging.info("test_while_sub [%s] passed! shape=%s", label, shape)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_while_mul():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shapes = [[64, 64]]
    for _pl_dt, tdt, label, atol, rtol in DTYPES_FP16:
        kernel = WHILE_MUL_KERNELS[label]
        for shape in shapes:
            x, y, z = _gen(shape, tdt, device)
            kernel(x, y, z)
            torch.npu.synchronize()
            z_ref = _ref(tdt, lambda a, b: a * b, x, y)
            torch.testing.assert_close(z, z_ref, atol=atol, rtol=rtol)
            logging.info("test_while_mul [%s] passed! shape=%s", label, shape)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_while_mul_add():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shapes = [[64, 64]]
    for _pl_dt, tdt, label, atol, rtol in DTYPES_FP16:
        kernel = WHILE_MUL_ADD_KERNELS[label]
        for shape in shapes:
            x, y, z = _gen(shape, tdt, device)
            kernel(x, y, z)
            torch.npu.synchronize()
            z_ref = _ref(tdt, lambda a, b: a * b + a, x, y)
            torch.testing.assert_close(z, z_ref, atol=atol, rtol=rtol)
            logging.info("test_while_mul_add [%s] passed! shape=%s", label, shape)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_while_tail():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shapes = [[97, 65]]
    for shape in shapes:
        x = torch.rand(shape, device=device, dtype=torch.float16)
        y = torch.rand(shape, device=device, dtype=torch.float16)
        z = torch.zeros(shape, device=device, dtype=torch.float16)
        while_tail_fp16_kernel(x, y, z)
        torch.npu.synchronize()
        z_ref = (x.float() + y.float()).to(torch.float16)
        torch.testing.assert_close(z, z_ref, atol=1e-3, rtol=1e-3)
        logging.info("test_while_tail passed! shape=%s", shape)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_while_4d_add():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shapes = [[2, 2, 128, 64]]
    for _pl_dt, tdt, label, atol, rtol in DTYPES_FP16:
        kernel = WHILE_4D_ADD_KERNELS[label]
        for shape in shapes:
            x, y, z = _gen(shape, tdt, device)
            kernel(x, y, z)
            torch.npu.synchronize()
            z_ref = _ref(tdt, lambda a, b: a + b, x, y)
            torch.testing.assert_close(z, z_ref, atol=atol, rtol=rtol)
            logging.info("test_while_4d_add [%s] passed! shape=%s", label, shape)
    for case_name, kernel, shape in WHILE_4D_LAYOUT_ADD_KERNELS:
        x, y, z = _gen(shape, torch.float16, device)
        kernel(x, y, z)
        torch.npu.synchronize()
        z_ref = (x.float() + y.float()).to(torch.float16)
        torch.testing.assert_close(z, z_ref, atol=1e-2, rtol=1e-2)
        logging.info("test_while_4d_add layout [%s] passed! shape=%s", case_name, shape)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_while_high_dim_add():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    for case_name, kernel, shape in WHILE_HIGH_DIM_ADD_KERNELS:
        x, y, z = _gen(shape, torch.float16, device)
        kernel(x, y, z)
        torch.npu.synchronize()
        z_ref = (x.float() + y.float()).to(torch.float16)
        torch.testing.assert_close(z, z_ref, atol=1e-2, rtol=1e-2)
        logging.info("test_while_high_dim_add [%s] passed! shape=%s", case_name, shape)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_while_large_shape():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shapes = [[512, 1024]]
    for _pl_dt, tdt, label, atol, rtol in DTYPES_FP16:
        kernel = WHILE_ADD_KERNELS[label]
        for shape in shapes:
            x, y, z = _gen(shape, tdt, device)
            kernel(x, y, z)
            torch.npu.synchronize()
            z_ref = _ref(tdt, lambda a, b: a + b, x, y)
            torch.testing.assert_close(z, z_ref, atol=atol, rtol=rtol)
            logging.info("test_while_large_shape [%s] passed! shape=%s", label, shape)


# ===================================================================
# while_and / while_or / while_and_three: compound boolean in while condition
# ===================================================================


# =============================================================================
# Test 13: while 布尔 and 条件
#         while boolean and condition
# =============================================================================
@pl.jit(auto_mutex=True)
def while_and_kernel(
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
        while i < m // TILE_M and i < 3:
            j: pl.DT_INT64 = 0
            while j < n // TILE_N:
                tile_a = a_db.next()
                tile_b = b_db.next()
                tile_c = c_db.next()
                pl.load_tile(tile_a, x, [i, j])
                pl.load_tile(tile_b, y, [i, j])
                pl.add(tile_c, tile_a, tile_b)
                pl.store_tile(z, tile_c, [i, j])
                j = j + 1
            i = i + 1


# =============================================================================
# Test 14: while 三元表达式条件
#         while ternary-expression condition
# =============================================================================
@pl.jit(auto_mutex=True)
def while_ternary_expr_kernel(
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
                pl.load_tile(tile_a, x, [i, j])
                pl.load_tile(tile_b, y, [i, j])
                sel = 1 if i < 2 else -1
                if sel == 1:
                    pl.add(tile_c, tile_a, tile_b)
                else:
                    pl.sub(tile_c, tile_a, tile_b)
                pl.store_tile(z, tile_c, [i, j])
                j = j + 1
            i = i + 1


# ===================================================================
# while_truthiness: non-bool scalar in while condition
# ===================================================================


# =============================================================================
# Test 15: while 整数真值条件
#         while integer truthiness condition
# =============================================================================
@pl.jit(auto_mutex=True)
def while_truthiness_int_kernel(
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
            remaining: pl.DT_INT64 = n // TILE_N
            while remaining:
                tile_a = a_db.next()
                tile_b = b_db.next()
                tile_c = c_db.next()
                pl.load_tile(tile_a, x, [i, j])
                pl.load_tile(tile_b, y, [i, j])
                pl.add(tile_c, tile_a, tile_b)
                pl.store_tile(z, tile_c, [i, j])
                j = j + 1
                remaining = remaining - 1
            i = i + 1


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_while_and():
    device = ST_DEVICE
    torch.npu.set_device(device)
    x = torch.rand(256, 64, device=device, dtype=torch.float16)
    y = torch.rand(256, 64, device=device, dtype=torch.float16)
    z = torch.zeros(256, 64, device=device, dtype=torch.float16)
    while_and_kernel(x, y, z)
    torch.npu.synchronize()
    z_ref = torch.zeros(256, 64, device=device, dtype=torch.float16)
    x_f = x.float()
    y_f = y.float()
    z_ref[:3 * TILE_M, :] = (x_f[:3 * TILE_M, :] + y_f[:3 * TILE_M, :]).to(torch.float16)
    torch.testing.assert_close(z, z_ref, atol=1e-2, rtol=1e-2)
    logging.info("test_while_and passed")


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_while_ternary_expr():
    device = ST_DEVICE
    torch.npu.set_device(device)
    x = torch.rand(128, 64, device=device, dtype=torch.float16)
    y = torch.rand(128, 64, device=device, dtype=torch.float16)
    z = torch.zeros(128, 64, device=device, dtype=torch.float16)
    while_ternary_expr_kernel(x, y, z)
    torch.npu.synchronize()
    z_ref = torch.zeros(128, 64, device=device, dtype=torch.float16)
    x_f = x.float()
    y_f = y.float()
    z_ref[:2 * TILE_M, :] = (x_f[:2 * TILE_M, :] + y_f[:2 * TILE_M, :]).to(torch.float16)
    z_ref[2 * TILE_M:, :] = (x_f[2 * TILE_M:, :] - y_f[2 * TILE_M:, :]).to(torch.float16)
    torch.testing.assert_close(z, z_ref, atol=1e-2, rtol=1e-2)
    logging.info("test_while_ternary_expr passed")


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_while_truthiness_int():
    device = ST_DEVICE
    torch.npu.set_device(device)
    x = torch.rand(128, 64, device=device, dtype=torch.float16)
    y = torch.rand(128, 64, device=device, dtype=torch.float16)
    z = torch.zeros(128, 64, device=device, dtype=torch.float16)
    while_truthiness_int_kernel(x, y, z)
    torch.npu.synchronize()
    z_ref = (x.float() + y.float()).to(torch.float16)
    torch.testing.assert_close(z, z_ref, atol=1e-2, rtol=1e-2)
    logging.info("test_while_truthiness_int passed")


# ===================================================================
# while_not: while condition with not  (FP16)
# ===================================================================


# =============================================================================
# Test 16: while not >= 条件
#         while not-greater-or-equal condition
# =============================================================================
@pl.jit(auto_mutex=True)
def while_not_ge_kernel(
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
        while not i >= m // TILE_M:
            j: pl.DT_INT64 = 0
            while j < n // TILE_N:
                tile_a = a_db.next()
                tile_b = b_db.next()
                tile_c = c_db.next()
                pl.load_tile(tile_a, x, [i, j])
                pl.load_tile(tile_b, y, [i, j])
                pl.add(tile_c, tile_a, tile_b)
                pl.store_tile(z, tile_c, [i, j])
                j = j + 1
            i = i + 1


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_while_not_ge():
    device = ST_DEVICE
    torch.npu.set_device(device)
    x = torch.rand(128, 64, device=device, dtype=torch.float16)
    y = torch.rand(128, 64, device=device, dtype=torch.float16)
    z = torch.zeros(128, 64, device=device, dtype=torch.float16)
    while_not_ge_kernel(x, y, z)
    torch.npu.synchronize()
    z_ref = _ref(torch.float16, lambda a, b: a + b, x, y)
    torch.testing.assert_close(z, z_ref, atol=1e-2, rtol=1e-2)
    logging.info("test_while_not_ge passed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    test_while_add()
    test_while_shape_generalization()
    test_while_sub()
    test_while_mul()
    test_while_mul_add()
    test_while_tail()
    test_while_4d_add()
    test_while_high_dim_add()
    test_while_large_shape()
    test_while_and()
    test_while_ternary_expr()
    test_while_truthiness_int()
    test_while_not_ge()
    logging.info("\nAll tests passed!")
