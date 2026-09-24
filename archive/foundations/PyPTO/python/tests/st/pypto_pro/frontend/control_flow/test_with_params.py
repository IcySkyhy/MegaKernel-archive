# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""with_params control_flow 前端测试。

测试覆盖场景:
  1. 覆盖 with_params control_flow 场景的前端语法、编译入口与运行验证。
  2. 覆盖 dtype 泛化：DT_INT32; Tensor 维度/shape 泛化：无。
  3. 覆盖控制流组合：for 循环、while 循环、if/elif/else 分支、with section 上下文、break 跳转、
     无值 return 提前退出、constexpr 编译期条件。
  4. 覆盖接口组合：基础算术运算、Tensor 与 Tile 数据搬运、section_vector/section_cube 上下文。
  5. 覆盖边界场景：scalar 参数驱动 range stop/step、if flag、while condition、constexpr 和函数边界。
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
# for_scalar_stop: scalar stop condition splits add/sub regions
#   Two sequential while-loops: while i < stop → add, while i < M//TILE_M → sub
#   Scalar-scalar comparison only in while conditions (not in if).
# ===================================================================


def make_for_scalar_stop_kernel(dtype, name_suffix=""):
    # =============================================================================
    # Test 1: 标量 stop 参数驱动 for
    #         scalar stop parameter drives for loop
    # =============================================================================
    @pl.jit(auto_mutex=True, name=name_suffix or None)
    def kernel(
        x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], dtype],
        y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], dtype],
        z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], dtype],
        stop: pl.DT_INT32,
    ):
        m = x.shape[0]
        n = x.shape[1]
        tile_type = pl.TileType(shape=[TILE_M, TILE_N], dtype=dtype, target_memory=pl.MemorySpace.Vec)
        a_db = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0, 1])
        b_db = pl.make_tile_group(type=tile_type, addrs=_addr_b(dtype), mutex_ids=[2, 3])
        c_db = pl.make_tile_group(type=tile_type, addrs=_addr_c(dtype), mutex_ids=[30, 31])
        with pl.section_vector():
            i: pl.DT_INT64 = 0
            while i < stop:
                for j in pl.range(0, n // TILE_N, 1):
                    tile_a = a_db.next()
                    tile_b = b_db.next()
                    tile_c = c_db.next()
                    pl.load_tile(tile_a, x, [i, j])
                    pl.load_tile(tile_b, y, [i, j])
                    pl.add(tile_c, tile_a, tile_b)
                    pl.store_tile(z, tile_c, [i, j])
                i = i + 1
            while i < m // TILE_M:
                for j in pl.range(0, n // TILE_N, 1):
                    tile_a = a_db.next()
                    tile_b = b_db.next()
                    tile_c = c_db.next()
                    pl.load_tile(tile_a, x, [i, j])
                    pl.load_tile(tile_b, y, [i, j])
                    pl.sub(tile_c, tile_a, tile_b)
                    pl.store_tile(z, tile_c, [i, j])
                i = i + 1

    return kernel


# ===================================================================
# for_scalar_step: for-loop with scalar step params
# ===================================================================


def make_for_scalar_step_kernel(dtype, name_suffix=""):
    # =============================================================================
    # Test 2: 标量 step 参数驱动 for
    #         scalar step parameter drives for loop
    # =============================================================================
    @pl.jit(auto_mutex=True, name=name_suffix or None)
    def kernel(
        x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], dtype],
        y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], dtype],
        z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], dtype],
        stop_val: pl.DT_INT32,
        step_val: pl.DT_INT32,
    ):
        n = x.shape[1]
        tile_type = pl.TileType(shape=[TILE_M, TILE_N], dtype=dtype, target_memory=pl.MemorySpace.Vec)
        a_db = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0, 1])
        b_db = pl.make_tile_group(type=tile_type, addrs=_addr_b(dtype), mutex_ids=[2, 3])
        c_db = pl.make_tile_group(type=tile_type, addrs=_addr_c(dtype), mutex_ids=[30, 31])
        with pl.section_vector():
            for i in pl.range(0, stop_val, step_val):
                for j in pl.range(0, n // TILE_N, 1):
                    tile_a = a_db.next()
                    tile_b = b_db.next()
                    tile_c = c_db.next()
                    pl.load_tile(tile_a, x, [i, j])
                    pl.load_tile(tile_b, y, [i, j])
                    pl.add(tile_c, tile_a, tile_b)
                    pl.store_tile(z, tile_c, [i, j])

    return kernel


# ===================================================================
# if_scalar_flag: if-branch with scalar flag
# ===================================================================


def make_if_scalar_flag_kernel(dtype, name_suffix=""):
    # =============================================================================
    # Test 3: 标量 flag 驱动 if
    #         scalar flag drives if branch
    # =============================================================================
    @pl.jit(auto_mutex=True, name=name_suffix or None)
    def kernel(
        x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], dtype],
        y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], dtype],
        z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], dtype],
        flag: pl.DT_INT32,
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
                    pl.load_tile(tile_a, x, [i, j])
                    pl.load_tile(tile_b, y, [i, j])
                    if flag == 1:
                        pl.add(tile_c, tile_a, tile_b)
                    else:
                        pl.sub(tile_c, tile_a, tile_b)
                    pl.store_tile(z, tile_c, [i, j])

    return kernel


# ===================================================================
# while_scalar_cond: while-loop with scalar condition
# ===================================================================


def make_while_scalar_cond_kernel(dtype, name_suffix=""):
    # =============================================================================
    # Test 4: 标量条件驱动 while
    #         scalar condition drives while loop
    # =============================================================================
    @pl.jit(auto_mutex=True, name=name_suffix or None)
    def kernel(
        x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], dtype],
        y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], dtype],
        z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], dtype],
        max_iter: pl.DT_INT32,
    ):
        n = x.shape[1]
        tile_type = pl.TileType(shape=[TILE_M, TILE_N], dtype=dtype, target_memory=pl.MemorySpace.Vec)
        a_db = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0, 1])
        b_db = pl.make_tile_group(type=tile_type, addrs=_addr_b(dtype), mutex_ids=[2, 3])
        c_db = pl.make_tile_group(type=tile_type, addrs=_addr_c(dtype), mutex_ids=[30, 31])
        with pl.section_vector():
            i: pl.DT_INT64 = 0
            while i < max_iter:
                for j in pl.range(0, n // TILE_N, 1):
                    tile_a = a_db.next()
                    tile_b = b_db.next()
                    tile_c = c_db.next()
                    pl.load_tile(tile_a, x, [i, j])
                    pl.load_tile(tile_b, y, [i, j])
                    pl.add(tile_c, tile_a, tile_b)
                    pl.store_tile(z, tile_c, [i, j])
                i = i + 1

    return kernel


# ===================================================================
# for_if_break_scalar: for-loop with if-break on scalar
# ===================================================================


def make_for_if_break_scalar_kernel(dtype, name_suffix=""):
    # =============================================================================
    # Test 5: 标量参数 + for/if break
    #         scalar parameter + for/if break
    # =============================================================================
    @pl.jit(auto_mutex=True, name=name_suffix or None)
    def kernel(
        x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], dtype],
        y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], dtype],
        z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], dtype],
        break_col: pl.DT_INT32,
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
                    if j >= break_col:
                        break
                    tile_a = a_db.next()
                    tile_b = b_db.next()
                    tile_c = c_db.next()
                    pl.load_tile(tile_a, x, [i, j])
                    pl.load_tile(tile_b, y, [i, j])
                    pl.add(tile_c, tile_a, tile_b)
                    pl.store_tile(z, tile_c, [i, j])

    return kernel


# ===================================================================
# constexpr_scalar: constexpr branch with scalar
# ===================================================================


def make_constexpr_scalar_kernel(dtype, name_suffix=""):
    # =============================================================================
    # Test 6: 标量参数 + constexpr
    #         scalar parameter + constexpr
    # =============================================================================
    @pl.jit(auto_mutex=True, name=name_suffix or None)
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
                    pl.load_tile(tile_a, x, [i, j])
                    pl.load_tile(tile_b, y, [i, j])
                    if pl.constexpr(True):
                        pl.add(tile_c, tile_a, tile_b)
                    else:
                        pl.sub(tile_c, tile_a, tile_b)
                    pl.store_tile(z, tile_c, [i, j])

    return kernel


# ===================================================================
# func_range_bound: pl.range bound computed by plain python functions
#   div(m, n) -> m // n;  sub(m, n) -> m - n;  mul(m, n) -> m * n
#   All evaluated at compile-time to produce range stop/step.
# ===================================================================


def _div(a, b):
    return a // b


def _mul(a, b):
    return a * b


def make_func_range_bound_kernel(dtype, name_suffix=""):
    # =============================================================================
    # Test 7: 函数边界驱动 range
    #         function bound drives range
    # =============================================================================
    @pl.jit(auto_mutex=True, name=name_suffix or None)
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
            for i in pl.range(0, _div(m, TILE_M), 1):
                for j in pl.range(0, _div(n, TILE_N), _mul(1, 1)):
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

    return kernel


# ===================================================================
# if_func_bool_expr: if-condition uses a plain function returning bool
#   if _mul(i, j) < 3 → add, else → sub
# ===================================================================


def make_if_func_bool_expr_kernel(dtype, name_suffix=""):
    # =============================================================================
    # Test 8: 函数布尔表达式驱动 if
    #         function boolean expression drives if branch
    # =============================================================================
    @pl.jit(auto_mutex=True, name=name_suffix or None)
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
                    pl.load_tile(tile_a, x, [i, j])
                    pl.load_tile(tile_b, y, [i, j])
                    if _mul(i, j) < 3:
                        pl.add(tile_c, tile_a, tile_b)
                    else:
                        pl.sub(tile_c, tile_a, tile_b)
                    pl.store_tile(z, tile_c, [i, j])

    return kernel


# ===================================================================
# Test functions
# ===================================================================


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_for_scalar_stop():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shape = [128, 64]
    for pl_dt, tdt, label, atol, rtol in DTYPES_ALL:
        kernel = make_for_scalar_stop_kernel(pl_dt, f"for_scalar_stop_{label}")
        x, y, z = _gen(shape, tdt, device)
        kernel(x, y, z, 1)
        torch.npu.synchronize()
        z_ref = torch.zeros(shape, device=device, dtype=tdt)
        z_ref[:64, :] = _ref(tdt, lambda a, b: a + b, x[:64, :], y[:64, :])
        z_ref[64:, :] = _ref(tdt, lambda a, b: a - b, x[64:, :], y[64:, :])
        torch.testing.assert_close(z, z_ref, atol=atol, rtol=rtol)
        logging.info("test_for_scalar_stop [%s] passed! shape=%s", label, shape)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_for_scalar_step():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shape = [128, 64]
    for pl_dt, tdt, label, atol, rtol in DTYPES_FP16:
        kernel = make_for_scalar_step_kernel(pl_dt, f"for_scalar_step_{label}")
        x, y, z = _gen(shape, tdt, device)
        kernel(x, y, z, shape[0] // TILE_M, 1)
        torch.npu.synchronize()
        z_ref = _ref(tdt, lambda a, b: a + b, x, y)
        torch.testing.assert_close(z, z_ref, atol=atol, rtol=rtol)
        logging.info("test_for_scalar_step [%s] passed! shape=%s", label, shape)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_if_scalar_flag():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shape = [128, 64]
    for pl_dt, tdt, label, atol, rtol in DTYPES_FP16:
        kernel = make_if_scalar_flag_kernel(pl_dt, f"if_scalar_flag_{label}")
        x, y, z_add = _gen(shape, tdt, device)
        z_sub = torch.zeros(shape, device=device, dtype=tdt)
        kernel(x, y, z_add, 1)
        torch.npu.synchronize()
        z_ref_add = _ref(tdt, lambda a, b: a + b, x, y)
        torch.testing.assert_close(z_add, z_ref_add, atol=atol, rtol=rtol)
        kernel(x, y, z_sub, 0)
        torch.npu.synchronize()
        z_ref_sub = _ref(tdt, lambda a, b: a - b, x, y)
        torch.testing.assert_close(z_sub, z_ref_sub, atol=atol, rtol=rtol)
        logging.info("test_if_scalar_flag [%s] passed! shape=%s", label, shape)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_while_scalar_cond():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shape = [128, 64]
    for pl_dt, tdt, label, atol, rtol in DTYPES_FP16:
        kernel = make_while_scalar_cond_kernel(pl_dt, f"while_scalar_cond_{label}")
        x, y, z = _gen(shape, tdt, device)
        kernel(x, y, z, 2)
        torch.npu.synchronize()
        z_ref = _ref(tdt, lambda a, b: a + b, x, y)
        torch.testing.assert_close(z, z_ref, atol=atol, rtol=rtol)
        logging.info("test_while_scalar_cond [%s] passed! shape=%s", label, shape)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_for_if_break_scalar():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shape = [192, 128]
    for pl_dt, tdt, label, atol, rtol in DTYPES_FP16:
        kernel = make_for_if_break_scalar_kernel(pl_dt, f"for_if_break_scalar_{label}")
        x, y, z = _gen(shape, tdt, device)
        kernel(x, y, z, 1)
        torch.npu.synchronize()
        z_ref = torch.zeros(shape, device=device, dtype=tdt)
        z_ref[:, :TILE_N] = _ref(tdt, lambda a, b: a + b, x[:, :TILE_N], y[:, :TILE_N])
        torch.testing.assert_close(z, z_ref, atol=atol, rtol=rtol)
        logging.info("test_for_if_break_scalar [%s] passed! shape=%s", label, shape)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_constexpr_scalar():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shape = [128, 64]
    for pl_dt, tdt, label, atol, rtol in DTYPES_FP16:
        kernel = make_constexpr_scalar_kernel(pl_dt, f"constexpr_scalar_{label}")
        x, y, z = _gen(shape, tdt, device)
        kernel(x, y, z)
        torch.npu.synchronize()
        z_ref = _ref(tdt, lambda a, b: a + b, x, y)
        torch.testing.assert_close(z, z_ref, atol=atol, rtol=rtol)
        logging.info("test_constexpr_scalar [%s] passed! shape=%s", label, shape)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_func_range_bound():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shape = [128, 64]
    for pl_dt, tdt, label, atol, rtol in DTYPES_FP16:
        kernel = make_func_range_bound_kernel(pl_dt, f"func_range_bound_{label}")
        x, y, z = _gen(shape, tdt, device)
        kernel(x, y, z)
        torch.npu.synchronize()
        z_ref = _ref(tdt, lambda a, b: a + b, x, y)
        torch.testing.assert_close(z, z_ref, atol=atol, rtol=rtol)
        logging.info("test_func_range_bound [%s] passed! shape=%s", label, shape)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_if_func_bool_expr():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shape = [128, 64]
    for pl_dt, tdt, label, atol, rtol in DTYPES_FP16:
        kernel = make_if_func_bool_expr_kernel(pl_dt, f"if_func_bool_expr_{label}")
        x, y, z = _gen(shape, tdt, device)
        kernel(x, y, z)
        torch.npu.synchronize()
        z_ref = torch.zeros(shape, device=device, dtype=tdt)
        for ti in range(shape[0] // TILE_M):
            for tj in range(shape[1] // TILE_N):
                if ti * tj < 3:
                    z_ref[ti * TILE_M:(ti + 1) * TILE_M, tj * TILE_N:(tj + 1) * TILE_N] = _ref(
                        tdt,
                        lambda a, b: a + b,
                        x[ti * TILE_M:(ti + 1) * TILE_M, tj * TILE_N:(tj + 1) * TILE_N],
                        y[ti * TILE_M:(ti + 1) * TILE_M, tj * TILE_N:(tj + 1) * TILE_N],
                    )
                else:
                    z_ref[ti * TILE_M:(ti + 1) * TILE_M, tj * TILE_N:(tj + 1) * TILE_N] = _ref(
                        tdt,
                        lambda a, b: a - b,
                        x[ti * TILE_M:(ti + 1) * TILE_M, tj * TILE_N:(tj + 1) * TILE_N],
                        y[ti * TILE_M:(ti + 1) * TILE_M, tj * TILE_N:(tj + 1) * TILE_N],
                    )
        torch.testing.assert_close(z, z_ref, atol=atol, rtol=rtol)
        logging.info("test_if_func_bool_expr [%s] passed! shape=%s", label, shape)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    test_for_scalar_stop()
    test_for_scalar_step()
    test_if_scalar_flag()
    test_while_scalar_cond()
    test_for_if_break_scalar()
    test_constexpr_scalar()
    test_func_range_bound()
    test_if_func_bool_expr()
    logging.info("\nAll tests passed!")
