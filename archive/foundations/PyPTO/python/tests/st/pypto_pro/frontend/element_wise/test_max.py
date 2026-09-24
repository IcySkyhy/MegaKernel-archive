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

"""pl.max NPU 泛化测试 — 标量级取大语法糖，覆盖多种数据类型与控制流。

参考文献:
  docs/zh/pypto_pro/api/Utils-API/python_syntax_sugar/max.md

被测 API:
  pl.max(lhs, rhs)  — 取两个标量中的较大值，恰好 2 个参数，不接受关键字参数；
  lhs/rhs 为标量 Expr 或 Python int。标量级别（区别于 tile 级逐元素的 pl.maximum）。

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
def max_const_kernel(
    out: pl.Tensor[[3], pl.DT_INT32],
):
    with pl.section_vector():
        out[0] = pl.max(3, 9)
        out[1] = pl.max(-5, -2)
        out[2] = pl.max(7, 7)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_max_const():
    """测试 pl.max 使用 Python int 常量作为参数的场景。

    测试目的：验证 pl.max(lhs, rhs) 对纯 Python int 常量（正数、负数、相等）取最大值的正确性。
    输入参数：无显式输入参数，kernel 内部使用 pl.max(3,9)、pl.max(-5,-2)、pl.max(7,7)；
             输出 tensor 为 [3] INT32，分配在 NPU 上，由 kernel 写入结果。
    预期行为：out = [9, -2, 7]（分别为 max(3,9)=9, max(-5,-2)=-2, max(7,7)=7）；
             通过 torch.equal 与预期 tensor 逐元素精确比较验证。
    """
    _check_npu()
    out = torch.zeros(3, device=ST_DEVICE, dtype=torch.int32)
    max_const_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([9, -2, 7], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# Test 2: INT32 标量 — struct 字段与标量算术表达式
# =============================================================================
@pl.jit()
def max_int32_kernel(
    out: pl.Tensor[[3], pl.DT_INT32],
):
    c = pl.struct("MaxI32", a=10, b=25, c=-3)
    with pl.section_vector():
        out[0] = pl.max(c.a, c.b)
        out[1] = pl.max(c.a, c.c)
        out[2] = pl.max(c.b + 5, c.a * 3)


# =============================================================================
# Test 3: INT64 标量 — 大整数取大
# =============================================================================
@pl.jit()
def max_int64_kernel(
    out: pl.Tensor[[2], pl.DT_INT64],
):
    c = pl.struct("MaxI64", a=1000000, b=7, c=999999)
    with pl.section_vector():
        out[0] = pl.max(c.a, c.b)
        out[1] = pl.max(c.b, c.c)


# =============================================================================
# Test 4: FP32 标量 — pl.const 构造浮点标量取大
# =============================================================================
@pl.jit()
def max_fp32_kernel(
    out: pl.Tensor[[2], pl.DT_FP32],
):
    with pl.section_vector():
        a = pl.const(2.5, pl.DT_FP32)
        b = pl.const(-1.5, pl.DT_FP32)
        c = pl.const(2.5, pl.DT_FP32)
        out[0] = pl.max(a, b)
        out[1] = pl.max(b, c)


# =============================================================================
# Test 5: for 循环 — running max（逐迭代更新最大值）
# =============================================================================
@pl.jit()
def max_for_kernel(
    out: pl.Tensor[[1], pl.DT_INT32],
):
    m = pl.struct("RunMax", v=0)
    with pl.section_vector():
        for i in pl.range(0, 5):
            m.v = pl.max(m.v, i * i - 4)
        out[0] = m.v


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_max_for():
    """测试 pl.max 在 for 循环中作为 running max 累加器的场景。

    测试目的：验证 pl.max 在 for 循环体内逐迭代更新 struct 字段值、实现 running max 的正确性。
    输入参数：无显式输入参数，循环 i=0..4，每轮计算 m.v = pl.max(m.v, i*i-4)，初始 m.v=0；
             输出 tensor 为 [1] INT32。
    预期行为：out = [12]（i*i-4 序列: -4,-3,0,5,12; running max 从 0: 0,0,0,5,12 -> 最终 12）；
             通过 torch.equal 精确比较验证。
    """
    _check_npu()
    out = torch.zeros(1, device=ST_DEVICE, dtype=torch.int32)
    max_for_kernel(out)
    torch.npu.synchronize()
    # i*i-4: -4,-3,0,5,12; running max from 0: 0,0,0,5,12 -> 12
    expected = torch.tensor([12], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# Test 6: for + if/else — 两分支分别用 pl.max 更新
# =============================================================================
@pl.jit()
def max_if_else_kernel(
    out: pl.Tensor[[1], pl.DT_INT32],
):
    s = pl.struct("IfMax", r=0)
    with pl.section_vector():
        for i in pl.range(0, 6):
            if i < 3:
                s.r = pl.max(s.r, i * 2)
            else:
                s.r = pl.max(s.r, 20 - i)
        out[0] = s.r


# =============================================================================
# Test 7: while 循环 — 裸标量循环变量 + struct 累计最大值
# =============================================================================
@pl.jit()
def max_while_kernel(
    out: pl.Tensor[[1], pl.DT_INT32],
):
    acc = pl.struct("WMax", best=0)
    with pl.section_vector():
        x: pl.DT_INT32 = 0
        while x < 5:
            x = x + 1
            acc.best = pl.max(acc.best, x * 3 - 8)
        out[0] = acc.best


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_max_while():
    """测试 pl.max 在 while 循环控制流中作为 running max 累加器的场景。

    测试目的：验证 pl.max 在 while 循环体内与裸标量循环变量 + struct 组合使用时的正确性。
    输入参数：无显式输入参数，while x<5，裸标量 x: INT32 自增，每轮 acc.best = pl.max(acc.best, x*3-8)；
             输出 tensor 为 [1] INT32。
    预期行为：out = [7]（x=1..5: x*3-8 = -5,-2,1,4,7; running max 从 0 -> 最终 7）；
             通过 torch.equal 精确比较验证。
    """
    _check_npu()
    out = torch.zeros(1, device=ST_DEVICE, dtype=torch.int32)
    max_while_kernel(out)
    torch.npu.synchronize()
    # x=1..5: x*3-8 = -5,-2,1,4,7; running max from 0 -> 7
    expected = torch.tensor([7], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# Test 8: 作为 for 循环边界 — pl.max 决定迭代次数
# =============================================================================
@pl.jit()
def max_loop_bound_kernel(
    out: pl.Tensor[[1], pl.DT_INT32],
):
    cnt = pl.struct("Bnd", n=0)
    with pl.section_vector():
        for i in pl.range(0, pl.max(2, 5)):
            cnt.n = cnt.n + i
        out[0] = cnt.n


# =============================================================================
# Test 9: Python 内置 max() — 验证前端自动解析为 pl.max
# =============================================================================
@pl.jit()
def max_builtin_kernel(
    out: pl.Tensor[[3], pl.DT_INT32],
):
    with pl.section_vector():
        out[0] = max(3, 9)
        out[1] = max(-5, -2)
        out[2] = max(7, 7)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_max_builtin():
    """测试 Python 内置 max() 在前端解析时自动转换为 pl.max 的场景。

    测试目的：验证内置 max(lhs, rhs) 与 pl.max(lhs, rhs) 等价。
    输入参数：无显式输入参数，kernel 内部使用 max(3,9)、max(-5,-2)、max(7,7)；
             输出 tensor 为 [3] INT32。
    预期行为：out = [9, -2, 7]；通过 torch.equal 精确比较验证。
    """
    _check_npu()
    out = torch.zeros(3, device=ST_DEVICE, dtype=torch.int32)
    max_builtin_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([9, -2, 7], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# Standalone runner
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    tests = [
        test_max_const,
        test_max_for,
        test_max_while,
        test_max_builtin,
    ]
    for t in tests:
        t()
        logging.info("%s passed!", t.__name__)
    logging.info("All pl.max NPU tests passed!")
