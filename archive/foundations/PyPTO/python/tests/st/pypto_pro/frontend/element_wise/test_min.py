#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""pl.min NPU 泛化测试 — 标量级取小语法糖，覆盖多种数据类型与控制流。

参考文献:
  docs/zh/pypto_pro/api/Utils-API/python_syntax_sugar/min.md

被测 API:
  pl.min(lhs, rhs)  — 取两个标量中的较小值，恰好 2 个参数，不接受关键字参数；
  lhs/rhs 为标量 Expr 或 Python int。标量级别（区别于 tile 级逐元素的 pl.minimum）。

数据类型覆盖:
  Python int 常量 / INT32 标量 / INT64 标量 / FP32 标量 (pl.const)

控制流覆盖:
  for 循环 / if-else / while 循环 / 作为循环边界
"""

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
# Test 1: Python int 常量 — 含正数、负数、相等三种
# =============================================================================
@pl.jit()
def min_const_kernel(
    out: pl.Tensor[[3], pl.DT_INT32],
):
    with pl.section_vector():
        out[0] = pl.min(3, 9)
        out[1] = pl.min(-5, -2)
        out[2] = pl.min(7, 7)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_min_const():
    """测试 pl.min 使用 Python int 常量作为参数的场景。

    测试目的：验证 pl.min(lhs, rhs) 对纯 Python int 常量（正数、负数、相等）取最小值的正确性。
    输入参数：无显式输入参数，kernel 内部使用 pl.min(3,9)、pl.min(-5,-2)、pl.min(7,7)；
             输出 tensor 为 [3] INT32，分配在 NPU 上，由 kernel 写入结果。
    预期行为：out = [3, -5, 7]（分别为 min(3,9)=3, min(-5,-2)=-5, min(7,7)=7）；
             通过 torch.equal 与预期 tensor 逐元素精确比较验证。
    """
    _check_npu()
    out = torch.zeros(3, device=ST_DEVICE, dtype=torch.int32)
    min_const_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([3, -5, 7], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# Test 2: INT32 标量 — struct 字段与标量算术表达式
# =============================================================================
@pl.jit()
def min_int32_kernel(
    out: pl.Tensor[[3], pl.DT_INT32],
):
    c = pl.struct("MinI32", a=10, b=25, c=-3)
    with pl.section_vector():
        out[0] = pl.min(c.a, c.b)
        out[1] = pl.min(c.a, c.c)
        out[2] = pl.min(c.b + 5, c.a * 3)


# =============================================================================
# Test 3: INT64 标量 — 大整数取小
# =============================================================================
@pl.jit()
def min_int64_kernel(
    out: pl.Tensor[[2], pl.DT_INT64],
):
    c = pl.struct("MinI64", a=1000000, b=7, c=999999)
    with pl.section_vector():
        out[0] = pl.min(c.a, c.b)
        out[1] = pl.min(c.a, c.c)


# =============================================================================
# Test 4: FP32 标量 — pl.const 构造浮点标量取小
# =============================================================================
@pl.jit()
def min_fp32_kernel(
    out: pl.Tensor[[2], pl.DT_FP32],
):
    with pl.section_vector():
        a = pl.const(2.5, pl.DT_FP32)
        b = pl.const(-1.5, pl.DT_FP32)
        c = pl.const(-1.5, pl.DT_FP32)
        out[0] = pl.min(a, b)
        out[1] = pl.min(b, c)


# =============================================================================
# Test 5: for 循环 — running min（逐迭代更新最小值）
# =============================================================================
@pl.jit()
def min_for_kernel(
    out: pl.Tensor[[1], pl.DT_INT32],
):
    m = pl.struct("RunMin", v=100)
    with pl.section_vector():
        for i in pl.range(0, 5):
            m.v = pl.min(m.v, i * i - 4)
        out[0] = m.v


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_min_for():
    """测试 pl.min 在 for 循环中作为 running min 累加器的场景。

    测试目的：验证 pl.min 在 for 循环体内逐迭代更新 struct 字段值、实现 running min 的正确性。
    输入参数：无显式输入参数，循环 i=0..4，每轮计算 m.v = pl.min(m.v, i*i-4)，初始 m.v=100；
             输出 tensor 为 [1] INT32。
    预期行为：out = [-4]（i*i-4 序列: -4,-3,0,5,12; running min 从 100: 100,-4,-4,-4,-4 -> 最终 -4）；
             通过 torch.equal 精确比较验证。
    """
    _check_npu()
    out = torch.zeros(1, device=ST_DEVICE, dtype=torch.int32)
    min_for_kernel(out)
    torch.npu.synchronize()
    # i*i-4: -4,-3,0,5,12; running min from 100 -> -4
    expected = torch.tensor([-4], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# Test 6: for + if/else — 两分支分别用 pl.min 更新
# =============================================================================
@pl.jit()
def min_if_else_kernel(
    out: pl.Tensor[[1], pl.DT_INT32],
):
    s = pl.struct("IfMin", r=100)
    with pl.section_vector():
        for i in pl.range(0, 6):
            if i < 3:
                s.r = pl.min(s.r, 20 - i * 2)
            else:
                s.r = pl.min(s.r, i + 5)
        out[0] = s.r


# =============================================================================
# Test 7: while 循环 — 裸标量循环变量 + struct 累计最小值
# =============================================================================
@pl.jit()
def min_while_kernel(
    out: pl.Tensor[[1], pl.DT_INT32],
):
    acc = pl.struct("WMin", best=100)
    with pl.section_vector():
        x: pl.DT_INT32 = 0
        while x < 5:
            x = x + 1
            acc.best = pl.min(acc.best, x * 3 - 8)
        out[0] = acc.best


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_min_while():
    """测试 pl.min 在 while 循环控制流中作为 running min 累加器的场景。

    测试目的：验证 pl.min 在 while 循环体内与裸标量循环变量 + struct 组合使用时的正确性。
    输入参数：无显式输入参数，while x<5，裸标量 x: INT32 自增，每轮 acc.best = pl.min(acc.best, x*3-8)；
             输出 tensor 为 [1] INT32。
    预期行为：out = [-5]（x=1..5: x*3-8 = -5,-2,1,4,7; running min 从 100 -> 最终 -5）；
             通过 torch.equal 精确比较验证。
    """
    _check_npu()
    out = torch.zeros(1, device=ST_DEVICE, dtype=torch.int32)
    min_while_kernel(out)
    torch.npu.synchronize()
    # x=1..5: x*3-8 = -5,-2,1,4,7; running min from 100 -> -5
    expected = torch.tensor([-5], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# Test 8: 作为 for 循环边界 — pl.min 决定迭代次数
# =============================================================================
@pl.jit()
def min_loop_bound_kernel(
    out: pl.Tensor[[1], pl.DT_INT32],
):
    cnt = pl.struct("Bnd", n=0)
    with pl.section_vector():
        for i in pl.range(0, pl.min(5, 3)):
            cnt.n = cnt.n + i
        out[0] = cnt.n


# =============================================================================
# Test 9: Python 内置 min() — 验证前端自动解析为 pl.min
# =============================================================================
@pl.jit()
def min_builtin_kernel(
    out: pl.Tensor[[3], pl.DT_INT32],
):
    with pl.section_vector():
        out[0] = min(3, 9)
        out[1] = min(-5, -2)
        out[2] = min(7, 7)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_min_builtin():
    """测试 Python 内置 min() 在前端解析时自动转换为 pl.min 的场景。

    测试目的：验证内置 min(lhs, rhs) 与 pl.min(lhs, rhs) 等价。
    输入参数：无显式输入参数，kernel 内部使用 min(3,9)、min(-5,-2)、min(7,7)；
             输出 tensor 为 [3] INT32。
    预期行为：out = [3, -5, 7]；通过 torch.equal 精确比较验证。
    """
    _check_npu()
    out = torch.zeros(3, device=ST_DEVICE, dtype=torch.int32)
    min_builtin_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([3, -5, 7], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# Standalone runner
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    tests = [
        test_min_const,
        test_min_for,
        test_min_while,
        test_min_builtin,
    ]
    for t in tests:
        t()
        logging.info("%s passed!", t.__name__)
    logging.info("All pl.min NPU tests passed!")
