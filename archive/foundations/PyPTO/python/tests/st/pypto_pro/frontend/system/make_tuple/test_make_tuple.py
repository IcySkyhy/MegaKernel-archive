# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""NPU coverage for active ``pl.make_tuple`` scalar and control-flow scenarios.

Covers scalar fields, INT64 interaction with ``pl.max``, construction in a for
loop, if/else aggregation, and many-field access.
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
# Test 1: 标量字段打包 + .字段名 访问
# =============================================================================
@pl.jit()
def tuple_scalar_kernel(
    out: pl.Tensor[[2], pl.DT_INT32],
):
    s = pl.struct("TScalar", a=11, b=22)
    with pl.section_vector():
        t = pl.make_tuple(first=s.a, second=s.b)
        pl.setval(out, 0, t.first + t.second)
        pl.setval(out, 1, t.second - t.first)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_tuple_scalar():
    """测试 pl.make_tuple 打包标量字段并通过 .字段名 访问的场景。

    测试目的：验证 pl.make_tuple 将 INT32 标量字段打包为编译期元组后，通过 .字段名 访问并进行算术运算的正确性。
    输入参数：无显式输入参数，struct 字段 a=11, b=22，打包为 t = pl.make_tuple(first=s.a, second=s.b)；
             输出 tensor 为 [2] INT32。
    预期行为：out = [33, 11]（t.first+t.second=11+22=33, t.second-t.first=22-11=11）；
             通过 torch.equal 精确比较验证。
    """
    _check_npu()
    out = torch.zeros(2, device=ST_DEVICE, dtype=torch.int32)
    tuple_scalar_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([33, 11], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# Test 2: INT64 标量字段 + 与 pl.max 组合
# =============================================================================
@pl.jit()
def tuple_int64_kernel(
    out: pl.Tensor[[2], pl.DT_INT64],
):
    s = pl.struct("T64", a=100000, b=7)
    with pl.section_vector():
        t = pl.make_tuple(big=s.a, small=s.b)
        pl.setval(out, 0, pl.max(t.big, t.small))
        pl.setval(out, 1, t.big + t.small)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_tuple_int64():
    """测试 pl.make_tuple 使用 INT64 标量字段 + 与 pl.max 组合使用的场景。

    测试目的：验证 pl.make_tuple 打包 INT64 类型大整数字段后与 pl.max 等其他语法糖 API 组合使用的正确性。
    输入参数：无显式输入参数，struct 字段 a=100000, b=7，打包 t = pl.make_tuple(big=s.a, small=s.b)；
             输出 tensor 为 [2] INT64。
    预期行为：out = [100000, 100007]（pl.max(100000,7)=100000, 100000+7=100007）；
             通过 torch.equal 精确比较验证。
    """
    _check_npu()
    out = torch.zeros(2, device=ST_DEVICE, dtype=torch.int64)
    tuple_int64_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([100000, 100007], device=ST_DEVICE, dtype=torch.int64)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# Test 3: for 循环内打包 — 字段值经 struct INT32 字段中转（规避 uint64 narrowing）
# =============================================================================
@pl.jit()
def tuple_in_loop_kernel(
    out: pl.Tensor[[1], pl.DT_INT32],
):
    acc = pl.struct("LoopT", v=0, cur=0)
    with pl.section_vector():
        for i in pl.range(0, 4):
            acc.cur = i
            t = pl.make_tuple(x=acc.cur, y=acc.cur * 10)
            acc.v = acc.v + t.x + t.y
        pl.setval(out, 0, acc.v)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_tuple_in_loop():
    """测试 pl.make_tuple 在 for 循环内部打包并通过 struct 字段中转的场景。

    测试目的：验证在循环体内部每轮调用 pl.make_tuple 打包字段值（经 struct INT32 字段中转规避 u64 narrowing），
             然后解包访问参与累加的正确性。
    输入参数：无显式输入参数，循环 i=0..3，每轮 acc.cur=i，打包 t=pl.make_tuple(x=acc.cur, y=acc.cur*10)；
             acc.v 累加 t.x+t.y；输出 tensor 为 [1] INT32。
    预期行为：out = [66]（i*11: 0+11+22+33 = 66）；
             通过 torch.equal 精确比较验证。
    """
    _check_npu()
    out = torch.zeros(1, device=ST_DEVICE, dtype=torch.int32)
    tuple_in_loop_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([66], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# Test 4: if/else 分支结果聚合 — 分支写 struct，循环后 make_tuple 打包
# =============================================================================
@pl.jit()
def tuple_if_else_kernel(
    out: pl.Tensor[[3], pl.DT_INT32],
):
    st = pl.struct("Branch", lo=0, hi=0)
    with pl.section_vector():
        for i in pl.range(0, 6):
            if i < 3:
                st.lo = st.lo + i
            else:
                st.hi = st.hi + i
        t = pl.make_tuple(lo=st.lo, hi=st.hi)
        pl.setval(out, 0, t.lo)
        pl.setval(out, 1, t.hi)
        pl.setval(out, 2, t.lo + t.hi)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_tuple_if_else():
    """测试 pl.make_tuple 在 for + if/else 分支中聚合结果的场景。

    测试目的：验证分支控制流中 struct 字段分别累加（lo 和 hi），循环后用 pl.make_tuple 打包双路结果。
    输入参数：无显式输入参数，循环 i=0..5，i<3 累加到 st.lo，i>=3 累加到 st.hi；
             循环后用 pl.make_tuple(lo=st.lo, hi=st.hi) 打包；输出 tensor 为 [3] INT32。
    预期行为：out = [3, 12, 15]（lo=0+1+2=3, hi=3+4+5=12, lo+hi=15）；
             通过 torch.equal 精确比较验证。
    """
    _check_npu()
    out = torch.zeros(3, device=ST_DEVICE, dtype=torch.int32)
    tuple_if_else_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([3, 12, 15], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# Test 5: 多字段打包 — 验证字段数量上限
# =============================================================================


@pl.jit()
def tuple_many_fields_kernel(
    out: pl.Tensor[[1], pl.DT_INT32],
):
    s = pl.struct("TMany", a=1, b=2, c=3, d=4, e=5, f=6, g=7, h=8)
    with pl.section_vector():
        t = pl.make_tuple(f1=s.a, f2=s.b, f3=s.c, f4=s.d, f5=s.e, f6=s.f, f7=s.g, f8=s.h)
        pl.setval(out, 0, t.f1 + t.f2 + t.f3 + t.f4 + t.f5 + t.f6 + t.f7 + t.f8)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_tuple_many_fields():
    """测试 pl.make_tuple 打包 8 个字段的场景，验证字段数量上限。

    测试目的：验证 pl.make_tuple 可以打包多个字段（8 个），字段访问正确。
    输入参数：无显式输入参数，struct 字段 a=1..h=8，打包为 t = pl.make_tuple(f1=s.a..f8=s.h)；
             输出 tensor 为 [1] INT32。
    预期行为：out = [36]（1+2+3+4+5+6+7+8=36）；
             通过 torch.equal 精确比较验证。
    """
    _check_npu()
    logging.info("------------test_tuple_many_fields--------------")
    out = torch.zeros(1, device=ST_DEVICE, dtype=torch.int32)
    tuple_many_fields_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([36], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"
    logging.info("test_tuple_many_fields passed!")


# =============================================================================
# Standalone runner
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    tests = [
        test_tuple_scalar,
        test_tuple_int64,
        test_tuple_in_loop,
        test_tuple_if_else,
        test_tuple_many_fields,
    ]
    for t in tests:
        t()
        logging.info("%s passed!", t.__name__)
    logging.info("All pl.make_tuple NPU tests passed!")
