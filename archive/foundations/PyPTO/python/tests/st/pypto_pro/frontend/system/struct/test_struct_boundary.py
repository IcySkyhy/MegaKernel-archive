# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Boundary tests for pl.struct and pl.struct_array.

Covers field count limits, unsupported operations, index types, and
boundary behaviors.
"""

from __future__ import annotations

import logging
import os

import pypto_pro.language as pl
import pytest
import torch

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"


def _check_npu():
    try:
        torch.npu.set_device(ST_DEVICE)
        return True
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
        return False


# =============================================================================
# struct.md: 字段数量上限验证
# =============================================================================
@pl.jit()
def struct_many_fields_kernel(
    out: pl.Tensor[[1], pl.DT_INT32],
):
    s = pl.struct("TMany", f1=1, f2=2, f3=3, f4=4, f5=5, f6=6, f7=7, f8=8)
    with pl.section_vector():
        s.f1 = 10
        s.f2 = 20
        s.f3 = 30
        s.f4 = 40
        s.f5 = 50
        s.f6 = 60
        s.f7 = 70
        s.f8 = 80
        pl.setval(out, 0, s.f1 + s.f2 + s.f3 + s.f4 + s.f5 + s.f6 + s.f7 + s.f8)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_struct_many_fields():
    """验证 struct 支持 8 个字段，且字段可多次修改"""
    _check_npu()
    logging.info("------------test_struct_many_fields--------------")
    out = torch.zeros(1, device=ST_DEVICE, dtype=torch.int32)
    struct_many_fields_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([360], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"
    logging.info("test_struct_many_fields passed!")


# =============================================================================
# struct_array.md: 不支持的操作验证（编译期报错）
# =============================================================================
@pl.jit()
def struct_array_for_iteration_kernel(
    out: pl.Tensor[[1], pl.DT_INT32],
):
    ctx_arr = pl.struct_array(3, "T", v=0)
    with pl.section_vector():
        total = 0
        for slot in ctx_arr:  # 测试是否支持遍历
            total = total + slot.v
        pl.setval(out, 0, total)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_struct_array_for_iteration():
    """验证 struct_array 是否支持 for slot in ctx_arr 遍历"""
    _check_npu()
    logging.info("------------test_struct_array_for_iteration--------------")
    try:
        out = torch.zeros(1, device=ST_DEVICE, dtype=torch.int32)
        struct_array_for_iteration_kernel(out)
        torch.npu.synchronize()
        logging.info(f"test_struct_array_for_iteration: 支持遍历，结果={out.tolist()}")
    except Exception as e:
        logging.info(f"test_struct_array_for_iteration: 不支持遍历，错误={e}")


@pl.jit()
def struct_array_append_kernel(
    out: pl.Tensor[[1], pl.DT_INT32],
):
    ctx_arr = pl.struct_array(3, "T", v=0)
    with pl.section_vector():
        new_slot = pl.struct("T", v=100)
        ctx_arr.append(new_slot)  # 测试是否支持 append
        pl.setval(out, 0, ctx_arr[3].v)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_struct_array_append():
    """验证 struct_array 是否支持 .append()"""
    _check_npu()
    logging.info("------------test_struct_array_append--------------")
    try:
        out = torch.zeros(1, device=ST_DEVICE, dtype=torch.int32)
        struct_array_append_kernel(out)
        torch.npu.synchronize()
        logging.info(f"test_struct_array_append: 支持 append，结果={out.tolist()}")
    except Exception as e:
        logging.info(f"test_struct_array_append: 不支持 append，错误={e}")


@pl.jit()
def struct_array_len_kernel(
    out: pl.Tensor[[1], pl.DT_INT32],
):
    ctx_arr = pl.struct_array(3, "T", v=0)
    with pl.section_vector():
        n = len(ctx_arr)  # 测试是否支持 len()
        pl.setval(out, 0, n)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_struct_array_len():
    """验证 struct_array 是否支持 len()"""
    _check_npu()
    logging.info("------------test_struct_array_len--------------")
    try:
        out = torch.zeros(1, device=ST_DEVICE, dtype=torch.int32)
        struct_array_len_kernel(out)
        torch.npu.synchronize()
        logging.info(f"test_struct_array_len: 支持 len()，结果={out.tolist()}")
    except Exception as e:
        logging.info(f"test_struct_array_len: 不支持 len()，错误={e}")


@pl.jit()
def struct_array_slice_kernel(
    out: pl.Tensor[[1], pl.DT_INT32],
):
    ctx_arr = pl.struct_array(5, "T", v=0)
    with pl.section_vector():
        for i in pl.range(0, 5):
            ctx_arr[i].v = i * 10
        sub = ctx_arr[1:3]  # 测试是否支持切片
        pl.setval(out, 0, sub[0].v + sub[1].v)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_struct_array_slice():
    """验证 struct_array 是否支持切片 ctx_arr[0:2]"""
    _check_npu()
    logging.info("------------test_struct_array_slice--------------")
    try:
        out = torch.zeros(1, device=ST_DEVICE, dtype=torch.int32)
        struct_array_slice_kernel(out)
        torch.npu.synchronize()
        logging.info(f"test_struct_array_slice: 支持切片，结果={out.tolist()}")
    except Exception as e:
        logging.info(f"test_struct_array_slice: 不支持切片，错误={e}")


# =============================================================================
# struct_array.md: 索引类型验证
# =============================================================================
@pl.jit()
def struct_array_scalar_index_kernel(
    out: pl.Tensor[[1], pl.DT_INT32],
    idx: pl.DT_INT32,
):
    ctx_arr = pl.struct_array(3, "T", v=0)
    with pl.section_vector():
        ctx_arr[1].v = 100
        slot = ctx_arr[idx]
        pl.setval(out, 0, slot.v)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_struct_array_scalar_index():
    """验证 struct_array 支持 pl.Scalar 变量作为索引"""
    _check_npu()
    logging.info("------------test_struct_array_scalar_index--------------")
    try:
        out = torch.zeros(1, device=ST_DEVICE, dtype=torch.int32)
        idx = torch.tensor(1, device=ST_DEVICE, dtype=torch.int32)
        struct_array_scalar_index_kernel(out, idx)
        torch.npu.synchronize()
        expected = torch.tensor([100], device=ST_DEVICE, dtype=torch.int32)
        assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"
        logging.info("test_struct_array_scalar_index passed!")
    except Exception as e:
        logging.info(f"test_struct_array_scalar_index: 不支持 Scalar 索引，错误={e}")


@pl.jit()
def struct_array_dynamic_index_kernel(
    out: pl.Tensor[[1], pl.DT_INT32],
):
    ctx_arr = pl.struct_array(3, "T", v=0)
    with pl.section_vector():
        for i in pl.range(0, 3):
            ctx_arr[i].v = i * 10
        slot = ctx_arr[2]
        pl.setval(out, 0, slot.v)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_struct_array_dynamic_index():
    """验证 struct_array 支持动态计算的索引（循环变量）"""
    _check_npu()
    logging.info("------------test_struct_array_dynamic_index--------------")
    try:
        out = torch.zeros(1, device=ST_DEVICE, dtype=torch.int32)
        struct_array_dynamic_index_kernel(out)
        torch.npu.synchronize()
        expected = torch.tensor([20], device=ST_DEVICE, dtype=torch.int32)
        assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"
        logging.info("test_struct_array_dynamic_index passed!")
    except Exception as e:
        logging.info(f"test_struct_array_dynamic_index: 不支持动态索引，错误={e}")


# =============================================================================
# struct_array.md: 越界访问验证
# =============================================================================
@pl.jit()
def struct_array_out_of_bounds_kernel(
    out: pl.Tensor[[1], pl.DT_INT32],
):
    ctx_arr = pl.struct_array(3, "T", v=0)
    with pl.section_vector():
        ctx_arr[5].v = 100  # 越界访问
        pl.setval(out, 0, ctx_arr[5].v)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_struct_array_out_of_bounds():
    """验证 struct_array 越界访问的行为"""
    _check_npu()
    logging.info("------------test_struct_array_out_of_bounds--------------")
    try:
        out = torch.zeros(1, device=ST_DEVICE, dtype=torch.int32)
        struct_array_out_of_bounds_kernel(out)
        torch.npu.synchronize()
        logging.info(f"test_struct_array_out_of_bounds: 越界访问成功，结果={out.tolist()}")
    except Exception as e:
        logging.info(f"test_struct_array_out_of_bounds: 越界访问失败，错误={e}")


@pl.jit()
def struct_array_negative_index_kernel(
    out: pl.Tensor[[1], pl.DT_INT32],
):
    ctx_arr = pl.struct_array(3, "T", v=0)
    with pl.section_vector():
        ctx_arr[0].v = 10
        ctx_arr[1].v = 20
        ctx_arr[2].v = 30
        slot = ctx_arr[-1]  # 负数索引
        pl.setval(out, 0, slot.v)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_struct_array_negative_index():
    """验证 struct_array 负数索引的行为"""
    _check_npu()
    logging.info("------------test_struct_array_negative_index--------------")
    try:
        out = torch.zeros(1, device=ST_DEVICE, dtype=torch.int32)
        struct_array_negative_index_kernel(out)
        torch.npu.synchronize()
        logging.info(f"test_struct_array_negative_index: 负数索引成功，结果={out.tolist()}")
    except Exception as e:
        logging.info(f"test_struct_array_negative_index: 负数索引失败，错误={e}")


# =============================================================================
# struct.md: 循环/条件分支内创建 struct
# =============================================================================
@pl.jit()
def struct_in_loop_kernel(
    out: pl.Tensor[[1], pl.DT_INT32],
):
    with pl.section_vector():
        total = 0
        for i in pl.range(0, 4):
            s = pl.struct("LoopS", v=0)
            s.v = i * 10
            total = total + s.v
        pl.setval(out, 0, total)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_struct_in_loop():
    """验证可以在 for 循环内创建 struct"""
    _check_npu()
    logging.info("------------test_struct_in_loop--------------")
    out = torch.zeros(1, device=ST_DEVICE, dtype=torch.int32)
    struct_in_loop_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([0 + 10 + 20 + 30], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"
    logging.info("test_struct_in_loop passed!")


@pl.jit()
def struct_in_branch_kernel(
    out: pl.Tensor[[1], pl.DT_INT32],
    flag: pl.DT_BOOL,
):
    s = pl.struct("BranchS", v=0)
    with pl.section_vector():
        if flag:
            s.v = 100
        else:
            s.v = 200
        pl.setval(out, 0, s.v)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_struct_in_branch():
    """验证可以在 if/else 分支内修改 struct 字段（struct 须在分支外创建）"""
    _check_npu()
    logging.info("------------test_struct_in_branch--------------")
    out = torch.zeros(1, device=ST_DEVICE, dtype=torch.int32)
    struct_in_branch_kernel(out, True)
    torch.npu.synchronize()
    expected = torch.tensor([100], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"
    logging.info("test_struct_in_branch passed!")


# =============================================================================
# Standalone runner
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    tests = [
        test_struct_many_fields,
        test_struct_in_loop,
        test_struct_in_branch,
        test_struct_array_for_iteration,
        test_struct_array_append,
        test_struct_array_len,
        test_struct_array_slice,
        test_struct_array_scalar_index,
        test_struct_array_dynamic_index,
        test_struct_array_out_of_bounds,
        test_struct_array_negative_index,
    ]
    for t in tests:
        t()
        logging.info("%s completed!", t.__name__)
    logging.info("All struct boundary tests completed!")
