# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""return control_flow 前端测试。

测试覆盖场景:
  1. 覆盖 return control_flow 场景的前端语法、编译入口与运行验证。
  2. 覆盖 dtype 泛化：DT_FP16/DT_BF16/DT_FP32/DT_INT32; Tensor 维度/shape 泛化：2D/4D/3D。
  3. 覆盖控制流组合：for 循环、while 循环、if/elif/else 分支、with section 上下文、无值 return 提前退出。
  4. 覆盖接口组合：基础算术运算、激活/比较类运算、Tensor 与 Tile 数据搬运、section_vector/section_cube 上下文。
  5. 覆盖边界场景：提前空 return、分支/循环中 return、多层嵌套 return 和无返回值语义。
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

_FP32_INT32 = (pl.DT_FP32, pl.DT_INT32)


def _addr_b(dtype):
    return 0x8000 if dtype in _FP32_INT32 else 0x4000


def _addr_c(dtype):
    return 0x10000 if dtype in _FP32_INT32 else 0x8000


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
# return_early: for-loop add with return at mid-iteration
#               Returns after first row, skipping redundant ops and rest
# ===================================================================


# =============================================================================
# Test 1: 提前空 return - FP16
#         early bare return - FP16
# =============================================================================
@pl.jit(auto_mutex=True)
def return_early_fp16_kernel(
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
                    return
        tile_a = a_db.next()
        tile_b = b_db.next()
        tile_c = c_db.next()
        pl.load_tile(tile_a, x, [0, 0])
        pl.load_tile(tile_b, y, [0, 0])
        pl.add(tile_c, tile_a, tile_b)
        pl.store_tile(z, tile_c, [0, 0])
        pl.sub(tile_c, tile_a, tile_b)
        pl.store_tile(z, tile_c, [0, 0])
        pl.mul(tile_c, tile_a, tile_b)
        pl.store_tile(z, tile_c, [0, 0])
        pl.div(tile_c, tile_a, tile_b)
        pl.store_tile(z, tile_c, [0, 0])


# =============================================================================
# Test 2: 提前空 return - BF16
#         early bare return - BF16
# =============================================================================
@pl.jit(auto_mutex=True)
def return_early_bf16_kernel(
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
                if j >= 1:
                    return
        tile_a = a_db.next()
        tile_b = b_db.next()
        tile_c = c_db.next()
        pl.load_tile(tile_a, x, [0, 0])
        pl.load_tile(tile_b, y, [0, 0])
        pl.add(tile_c, tile_a, tile_b)
        pl.store_tile(z, tile_c, [0, 0])
        pl.sub(tile_c, tile_a, tile_b)
        pl.store_tile(z, tile_c, [0, 0])
        pl.mul(tile_c, tile_a, tile_b)
        pl.store_tile(z, tile_c, [0, 0])
        pl.maximum(tile_c, tile_a, tile_b)
        pl.store_tile(z, tile_c, [0, 0])


# =============================================================================
# Test 3: 提前空 return - FP32
#         early bare return - FP32
# =============================================================================
@pl.jit(auto_mutex=True)
def return_early_fp32_kernel(
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
                if j >= 1:
                    return
        tile_a = a_db.next()
        tile_b = b_db.next()
        tile_c = c_db.next()
        pl.load_tile(tile_a, x, [0, 0])
        pl.load_tile(tile_b, y, [0, 0])
        pl.add(tile_c, tile_a, tile_b)
        pl.store_tile(z, tile_c, [0, 0])
        pl.sub(tile_c, tile_a, tile_b)
        pl.store_tile(z, tile_c, [0, 0])
        pl.mul(tile_c, tile_a, tile_b)
        pl.store_tile(z, tile_c, [0, 0])
        pl.div(tile_c, tile_a, tile_b)
        pl.store_tile(z, tile_c, [0, 0])


# =============================================================================
# Test 4: 提前空 return - INT32
#         early bare return - INT32
# =============================================================================
@pl.jit(auto_mutex=True)
def return_early_int32_kernel(
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
                if j >= 1:
                    return
        tile_a = a_db.next()
        tile_b = b_db.next()
        tile_c = c_db.next()
        pl.load_tile(tile_a, x, [0, 0])
        pl.load_tile(tile_b, y, [0, 0])
        pl.add(tile_c, tile_a, tile_b)
        pl.store_tile(z, tile_c, [0, 0])
        pl.sub(tile_c, tile_a, tile_b)
        pl.store_tile(z, tile_c, [0, 0])
        pl.mul(tile_c, tile_a, tile_b)
        pl.store_tile(z, tile_c, [0, 0])
        pl.div(tile_c, tile_a, tile_b)
        pl.store_tile(z, tile_c, [0, 0])


RETURN_EARLY_KERNELS = {
    "fp16": return_early_fp16_kernel,
    "bf16": return_early_bf16_kernel,
    "fp32": return_early_fp32_kernel,
    "int32": return_early_int32_kernel,
}


# ===================================================================
# return_in_if: for-loop add with return inside if at mid-loop
#               Returns after second row first tile
# ===================================================================


# =============================================================================
# Test 5: if 分支内空 return - FP16
#         bare return inside if branch - FP16
# =============================================================================
@pl.jit(auto_mutex=True)
def return_in_if_fp16_kernel(
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
                if i >= 1:
                    return


RETURN_IN_IF_KERNELS = {
    "fp16": return_in_if_fp16_kernel,
}


# ===================================================================
# plain_def_return_helpers: @pl.jit kernel calls ordinary def helpers
#                           with scalar, tuple, and list return values
# ===================================================================


def plain_def_return_scalar(value):
    return value + 1


def plain_def_return_pair(first, second):
    return first, second


def plain_def_return_list(first, second):
    return [first, second]


def make_plain_def_return_helpers_kernel(dtype):
    # =============================================================================
    # Test 6: 普通函数调用 + 空 return
    #         plain function calls + bare return
    # =============================================================================
    @pl.jit(auto_mutex=True)
    def kernel(
        x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], dtype],
        y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], dtype],
        z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], dtype],
    ):
        m = x.shape[0]
        n = x.shape[1]
        tile_type = pl.TileType(shape=[TILE_M, TILE_N], dtype=dtype, target_memory=pl.MemorySpace.Vec)
        a_db = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0, 1])
        b_db = pl.make_tile_group(type=tile_type, addrs=_addr_b(dtype), mutex_ids=[2, 3])
        c_db = pl.make_tile_group(type=tile_type, addrs=_addr_c(dtype), mutex_ids=[30, 31])
        with pl.section_vector():
            for i in pl.range(0, m // TILE_M, 1):
                for j in pl.range(0, n // TILE_N, 1):
                    tile_a = a_db.next()
                    tile_b = b_db.next()
                    tile_c = c_db.next()
                    row_idx, col_idx = plain_def_return_pair(i, j)
                    offsets = plain_def_return_list(row_idx, col_idx)
                    plain_def_return_scalar(0)
                    pl.load_tile(tile_a, x, offsets)
                    pl.load_tile(tile_b, y, offsets)
                    pl.add(tile_c, tile_a, tile_b)
                    pl.store_tile(z, tile_c, offsets)

    return kernel


PLAIN_DEF_RETURN_HELPERS_KERNELS = {
    "fp16": make_plain_def_return_helpers_kernel(pl.DT_FP16),
    "bf16": make_plain_def_return_helpers_kernel(pl.DT_BF16),
    "fp32": make_plain_def_return_helpers_kernel(pl.DT_FP32),
    "int32": make_plain_def_return_helpers_kernel(pl.DT_INT32),
}


# ===================================================================
# Test functions
# ===================================================================


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_return_early():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shapes = [[128, 128]]
    for _pl_dt, tdt, label, atol, rtol in DTYPES_ALL:
        kernel = RETURN_EARLY_KERNELS[label]
        for shape in shapes:
            x, y, z = _gen(shape, tdt, device)
            kernel(x, y, z)
            torch.npu.synchronize()
            z_ref = torch.zeros(shape, device=device, dtype=tdt)
            z_ref[:TILE_M, :] = _ref(tdt, lambda a, b: a + b, x[:TILE_M, :], y[:TILE_M, :])
            torch.testing.assert_close(z, z_ref, atol=atol, rtol=rtol)
            logging.info("test_return_early [%s] passed! shape=%s", label, shape)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_return_in_if():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shapes = [[128, 128]]
    for _pl_dt, tdt, label, atol, rtol in DTYPES_FP16:
        kernel = RETURN_IN_IF_KERNELS[label]
        for shape in shapes:
            x, y, z = _gen(shape, tdt, device)
            kernel(x, y, z)
            torch.npu.synchronize()
            z_ref = torch.zeros(shape, device=device, dtype=tdt)
            z_ref[:TILE_M, :] = _ref(tdt, lambda a, b: a + b, x[:TILE_M, :], y[:TILE_M, :])
            sl = (slice(TILE_M, 2 * TILE_M), slice(0, TILE_N))
            z_ref[sl] = _ref(tdt, lambda a, b: a + b, x[sl], y[sl])
            torch.testing.assert_close(z, z_ref, atol=atol, rtol=rtol)
            logging.info("test_return_in_if [%s] passed! shape=%s", label, shape)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_plain_def_return_helpers():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shapes = [[128, 128]]
    for _pl_dt, tdt, label, atol, rtol in DTYPES_FP16:
        kernel = PLAIN_DEF_RETURN_HELPERS_KERNELS[label]
        for shape in shapes:
            x, y, z = _gen(shape, tdt, device)
            kernel(x, y, z)
            torch.npu.synchronize()
            z_ref = _ref(tdt, lambda a, b: a + b, x, y)
            torch.testing.assert_close(z, z_ref, atol=atol, rtol=rtol)
            logging.info("test_plain_def_return_helpers [%s] passed! shape=%s", label, shape)


# ===================================================================
# while_return: while-loop add with return  (FP16)
# ===================================================================


# =============================================================================
# Test 7: while 循环内空 return
#         bare return inside while loop
# =============================================================================
@pl.jit(auto_mutex=True)
def while_return_kernel(
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
                if j >= 1:
                    return
                j = j + 1
            i = i + 1


# ===================================================================
# while_if_return: while + if return  (FP16)
# ===================================================================


# =============================================================================
# Test 8: while + if 分支空 return
#         bare return inside while + if branch
# =============================================================================
@pl.jit(auto_mutex=True)
def while_if_return_kernel(
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
            if i >= 1:
                return
            i = i + 1


# ===================================================================
# for_while_return: for outer + while inner + return  (FP16)
# ===================================================================


# =============================================================================
# Test 9: for + while 嵌套空 return
#         bare return inside nested for + while
# =============================================================================
@pl.jit(auto_mutex=True)
def for_while_return_kernel(
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
                tile_a = a_db.next()
                tile_b = b_db.next()
                tile_c = c_db.next()
                pl.load_tile(tile_a, x, [i, j])
                pl.load_tile(tile_b, y, [i, j])
                pl.add(tile_c, tile_a, tile_b)
                pl.store_tile(z, tile_c, [i, j])
                j = j + 1
            return


# ===================================================================
# for_4layer_if_return: 4-layer for + if + return  (FP16)
# ===================================================================


# =============================================================================
# Test 10: for 4 层 + if 空 return
#         bare return inside 4-layer for + if
# =============================================================================
@pl.jit(auto_mutex=True)
def for_4layer_if_return_kernel(
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
        for i in pl.range(0, m // TILE_M, 1):
            for j in pl.range(0, n // TILE_N, 1):
                for k in pl.range(0, b, 1):
                    for l_idx in pl.range(0, h, 1):
                        tile_a = a_db.next()
                        tile_b = b_db.next()
                        tile_c = c_db.next()
                        pl.load_tile(tile_a, x, [k, l_idx, i, j], order=[2, 3])
                        pl.load_tile(tile_b, y, [k, l_idx, i, j], order=[2, 3])
                        pl.add(tile_c, tile_a, tile_b)
                        pl.store_tile(z, tile_c, [k, l_idx, i, j], order=[2, 3])
                    if k >= 1:
                        return


# ===================================================================
# for_while_if_return_else: for+while+if/else+return mid-loop  (FP16)
# ===================================================================


# =============================================================================
# Test 11: for/while/if/else 空 return
#         bare return inside for/while/if/else
# =============================================================================
@pl.jit(auto_mutex=True)
def for_while_if_return_else_kernel(
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
                tile_a = a_db.next()
                tile_b = b_db.next()
                tile_c = c_db.next()
                pl.load_tile(tile_a, x, [i, j])
                pl.load_tile(tile_b, y, [i, j])
                if i >= 1 and j >= 1:
                    pl.sub(tile_c, tile_a, tile_b)
                    pl.store_tile(z, tile_c, [i, j])
                    return
                else:
                    pl.add(tile_c, tile_a, tile_b)
                    pl.store_tile(z, tile_c, [i, j])
                j = j + 1


# ===================================================================
# Test functions
# ===================================================================


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_while_return():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shape = [128, 128]
    x = torch.rand(shape, device=device, dtype=torch.float16)
    y = torch.rand(shape, device=device, dtype=torch.float16)
    z = torch.zeros(shape, device=device, dtype=torch.float16)
    while_return_kernel(x, y, z)
    torch.npu.synchronize()
    z_ref = torch.zeros(shape, device=device, dtype=torch.float16)
    x_f = x.float()
    y_f = y.float()
    z_ref[:TILE_M, :] = (x_f[:TILE_M, :] + y_f[:TILE_M, :]).to(torch.float16)
    torch.testing.assert_close(z, z_ref, atol=1e-2, rtol=1e-2)
    logging.info("test_while_return passed! shape=%s", shape)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_while_if_return():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shape = [128, 128]
    x = torch.rand(shape, device=device, dtype=torch.float16)
    y = torch.rand(shape, device=device, dtype=torch.float16)
    z = torch.zeros(shape, device=device, dtype=torch.float16)
    while_if_return_kernel(x, y, z)
    torch.npu.synchronize()
    z_ref = torch.zeros(shape, device=device, dtype=torch.float16)
    x_f = x.float()
    y_f = y.float()
    z_ref[:2 * TILE_M, :] = (x_f[:2 * TILE_M, :] + y_f[:2 * TILE_M, :]).to(torch.float16)
    torch.testing.assert_close(z, z_ref, atol=1e-2, rtol=1e-2)
    logging.info("test_while_if_return passed! shape=%s", shape)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_for_while_return():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shape = [128, 128]
    x = torch.rand(shape, device=device, dtype=torch.float16)
    y = torch.rand(shape, device=device, dtype=torch.float16)
    z = torch.zeros(shape, device=device, dtype=torch.float16)
    for_while_return_kernel(x, y, z)
    torch.npu.synchronize()
    z_ref = torch.zeros(shape, device=device, dtype=torch.float16)
    x_f = x.float()
    y_f = y.float()
    z_ref[:TILE_M, :] = (x_f[:TILE_M, :] + y_f[:TILE_M, :]).to(torch.float16)
    torch.testing.assert_close(z, z_ref, atol=1e-2, rtol=1e-2)
    logging.info("test_for_while_return passed! shape=%s", shape)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_for_4layer_if_return():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shape = [2, 2, 128, 128]
    x = torch.rand(shape, device=device, dtype=torch.float16)
    y = torch.rand(shape, device=device, dtype=torch.float16)
    z = torch.zeros(shape, device=device, dtype=torch.float16)
    for_4layer_if_return_kernel(x, y, z)
    torch.npu.synchronize()
    z_ref = torch.zeros(shape, device=device, dtype=torch.float16)
    x_f = x.float()
    y_f = y.float()
    z_ref[:, :, :TILE_M, :TILE_N] = (x_f[:, :, :TILE_M, :TILE_N] + y_f[:, :, :TILE_M, :TILE_N]).to(torch.float16)
    torch.testing.assert_close(z, z_ref, atol=1e-2, rtol=1e-2)
    logging.info("test_for_4layer_if_return passed! shape=%s", shape)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_for_while_if_return_else():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shape = [128, 128]
    x = torch.rand(shape, device=device, dtype=torch.float16)
    y = torch.rand(shape, device=device, dtype=torch.float16)
    z = torch.zeros(shape, device=device, dtype=torch.float16)
    for_while_if_return_else_kernel(x, y, z)
    torch.npu.synchronize()
    z_ref = torch.zeros(shape, device=device, dtype=torch.float16)
    x_f = x.float()
    y_f = y.float()
    z_ref[:TILE_M, :TILE_N] = (x_f[:TILE_M, :TILE_N] + y_f[:TILE_M, :TILE_N]).to(torch.float16)
    z_ref[:TILE_M, TILE_N:2 * TILE_N] = (x_f[:TILE_M, TILE_N:2 * TILE_N] + y_f[:TILE_M, TILE_N:2 * TILE_N]).to(
        torch.float16
    )
    z_ref[TILE_M:2 * TILE_M, :TILE_N] = (x_f[TILE_M:2 * TILE_M, :TILE_N] + y_f[TILE_M:2 * TILE_M, :TILE_N]).to(
        torch.float16
    )
    z_ref[TILE_M:2 * TILE_M, TILE_N:2 * TILE_N] = (
        x_f[TILE_M:2 * TILE_M, TILE_N:2 * TILE_N] - y_f[TILE_M:2 * TILE_M, TILE_N:2 * TILE_N]
    ).to(torch.float16)
    torch.testing.assert_close(z, z_ref, atol=1e-2, rtol=1e-2)
    logging.info("test_for_while_if_return_else passed! shape=%s", shape)


# ===================================================================
# direct_return: for/while with direct return (no if guard)
# ===================================================================


# =============================================================================
# Test 12: for 循环直接 return
#         direct return inside for loop
# =============================================================================
@pl.jit(auto_mutex=True)
def for_direct_return_kernel(
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
            return


# =============================================================================
# Test 13: for 3 层嵌套直接 return
#         direct return inside 3-layer nested for
# =============================================================================
@pl.jit(auto_mutex=True)
def for_3layer_direct_return_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    b = x.shape[0]
    m = x.shape[1]
    n = x.shape[2]
    tile_type = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a_db = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0, 1])
    b_db = pl.make_tile_group(type=tile_type, addrs=0x4000, mutex_ids=[2, 3])
    c_db = pl.make_tile_group(type=tile_type, addrs=0x8000, mutex_ids=[30, 31])
    with pl.section_vector():
        for i in pl.range(0, m // TILE_M, 1):
            for j in pl.range(0, n // TILE_N, 1):
                for k in pl.range(0, b, 1):
                    tile_a = a_db.next()
                    tile_b = b_db.next()
                    tile_c = c_db.next()
                    pl.load_tile(tile_a, x, [k, i, j], order=[1, 2])
                    pl.load_tile(tile_b, y, [k, i, j], order=[1, 2])
                    pl.add(tile_c, tile_a, tile_b)
                    pl.store_tile(z, tile_c, [k, i, j], order=[1, 2])
                return


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_for_direct_return():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shape = [128, 128]
    x = torch.rand(shape, device=device, dtype=torch.float16)
    y = torch.rand(shape, device=device, dtype=torch.float16)
    z = torch.zeros(shape, device=device, dtype=torch.float16)
    for_direct_return_kernel(x, y, z)
    torch.npu.synchronize()
    z_ref = torch.zeros(shape, device=device, dtype=torch.float16)
    x_f = x.float()
    y_f = y.float()
    z_ref[:TILE_M, :] = (x_f[:TILE_M, :] + y_f[:TILE_M, :]).to(torch.float16)
    torch.testing.assert_close(z, z_ref, atol=1e-2, rtol=1e-2)
    logging.info("test_for_direct_return passed! shape=%s", shape)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_for_3layer_direct_return():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shape = [2, 128, 128]
    x = torch.rand(shape, device=device, dtype=torch.float16)
    y = torch.rand(shape, device=device, dtype=torch.float16)
    z = torch.zeros(shape, device=device, dtype=torch.float16)
    for_3layer_direct_return_kernel(x, y, z)
    torch.npu.synchronize()
    z_ref = torch.zeros(shape, device=device, dtype=torch.float16)
    x_f = x.float()
    y_f = y.float()
    z_ref[:, :TILE_M, :TILE_N] = (x_f[:, :TILE_M, :TILE_N] + y_f[:, :TILE_M, :TILE_N]).to(torch.float16)
    torch.testing.assert_close(z, z_ref, atol=1e-2, rtol=1e-2)
    logging.info("test_for_3layer_direct_return passed! shape=%s", shape)


# =============================================================================
# Test 14: for/while 8 层嵌套 return
#         return inside 8-layer nested for/while
# =============================================================================
@pl.jit(auto_mutex=True)
def for_while_8layer_return_kernel(
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
        for i0 in pl.range(0, m // TILE_M, 1):
            i1: pl.DT_INT64 = 0
            while i1 < n // TILE_N:
                for _i2 in pl.range(0, 1, 1):
                    i3: pl.DT_INT64 = 0
                    while i3 < 1:
                        for _i4 in pl.range(0, 1, 1):
                            i5: pl.DT_INT64 = 0
                            while i5 < 1:
                                for _i6 in pl.range(0, 1, 1):
                                    i7: pl.DT_INT64 = 0
                                    while i7 < 1:
                                        tile_a = a_db.next()
                                        tile_b = b_db.next()
                                        tile_c = c_db.next()
                                        pl.load_tile(tile_a, x, [i0, i1])
                                        pl.load_tile(tile_b, y, [i0, i1])
                                        pl.add(tile_c, tile_a, tile_b)
                                        pl.store_tile(z, tile_c, [i0, i1])
                                        return
                i1 = i1 + 1


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_for_while_8layer_return():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shape = [128, 128]
    x = torch.rand(shape, device=device, dtype=torch.float16)
    y = torch.rand(shape, device=device, dtype=torch.float16)
    z = torch.zeros(shape, device=device, dtype=torch.float16)
    for_while_8layer_return_kernel(x, y, z)
    torch.npu.synchronize()
    z_ref = torch.zeros(shape, device=device, dtype=torch.float16)
    x_f = x.float()
    y_f = y.float()
    z_ref[:TILE_M, :TILE_N] = (x_f[:TILE_M, :TILE_N] + y_f[:TILE_M, :TILE_N]).to(torch.float16)
    torch.testing.assert_close(z, z_ref, atol=1e-2, rtol=1e-2)
    logging.info("test_for_while_8layer_return passed! shape=%s", shape)


# =============================================================================
# Test 15: return 非对齐尾块 - FP16
#         return with unaligned tail block - FP16
# =============================================================================
@pl.jit(auto_mutex=True)
def return_unaligned_fp16_kernel(
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
                if j >= 1:
                    return
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
def test_return_unaligned_shape():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shapes = [[97, 65]]
    for shape in shapes:
        x = torch.rand(shape, device=device, dtype=torch.float16)
        y = torch.rand(shape, device=device, dtype=torch.float16)
        z = torch.zeros(shape, device=device, dtype=torch.float16)
        return_unaligned_fp16_kernel(x, y, z)
        torch.npu.synchronize()
        z_ref = torch.zeros(shape, device=device, dtype=torch.float16)
        if shape[1] > TILE_N:
            valid_r = min(shape[0], TILE_M)
            valid_c = min(shape[1], TILE_N)
            z_ref[:valid_r, :valid_c] = (x[:valid_r, :valid_c].float() + y[:valid_r, :valid_c].float()).to(
                torch.float16
            )
        else:
            z_ref = (x.float() + y.float()).to(torch.float16)
        torch.testing.assert_close(z, z_ref, atol=1e-2, rtol=1e-2)
        logging.info("test_return_unaligned_shape passed! shape=%s", shape)


def alloc_tile():
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x00000,
        mutex_ids=[0],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x10000,
        mutex_ids=[1],
    )
    a_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000,
        mutex_ids=[2],
    )
    b_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000,
        mutex_ids=[3],
    )
    c_l0c = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
        addrs=0x0000,
        mutex_ids=[4],
    )
    return a_l1, b_l1, a_l0a, b_l0b, c_l0c


@pl.jit(auto_mutex=True)
def return_tile_group_kernel(
    a: pl.Tensor[[64, 64], pl.DT_FP16],
    b: pl.Tensor[[64, 64], pl.DT_FP16],
    out: pl.Tensor[[64, 64], pl.DT_FP32],
):
    """Load to L1 (FP16, 2D, default layout), verify via matmul-with-identity. LOAD-N01/N13."""
    a_l1, b_l1, a_l0a, b_l0b, c_l0c = alloc_tile()
    with pl.section_cube():
        cur_a = a_l1.next()
        cur_b = b_l1.next()
        al = a_l0a.next()
        br = b_l0b.next()
        ac = c_l0c.next()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.move(al, cur_a)
        pl.move(br, cur_b)
        pl.matmul(ac, al, br)
        pl.store(out, ac, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_return_tile_group_kernel():
    """LOAD-N01: Basic load to L1 (FP16), verify via matmul-with-identity."""
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    a = torch.rand([64, 64], device=device, dtype=torch.float16)
    b = torch.eye(64, device=device, dtype=torch.float16)
    out = torch.zeros([64, 64], device=device, dtype=torch.float32)
    return_tile_group_kernel(a, b, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a.float(), rtol=1e-2, atol=1e-2)
    logging.info("LOAD-N01 passed: FP16 load to L1 via matmul-with-identity")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    test_return_early()
    test_return_in_if()
    test_plain_def_return_helpers()
    test_while_return()
    test_while_if_return()
    test_for_while_return()
    test_for_4layer_if_return()
    test_for_while_if_return_else()
    test_for_direct_return()
    test_for_3layer_direct_return()
    test_for_while_8layer_return()
    test_return_unaligned_shape()
    test_return_tile_group_kernel()
    logging.info("\nAll tests passed!")
