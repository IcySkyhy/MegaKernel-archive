# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""NPU coverage for active ``pl.struct`` field and control-flow scenarios.

Covers basic field access, loop accumulation, conditional updates, aliasing,
nested cross-assignment, and reference passing.
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
# Test 1: 基础赋值 + 两字段求和写回
# =============================================================================
@pl.jit()
def struct_basic_kernel(
    out: pl.Tensor[[1], pl.DT_INT32],
):
    ctx = pl.struct("Ctx1", val=0, base=0)
    with pl.section_vector():
        ctx.val = 100
        ctx.base = 200
        pl.setval(out, 0, ctx.val + ctx.base)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_struct_basic():
    _check_npu()
    out = torch.zeros(1, device=ST_DEVICE, dtype=torch.int32)
    struct_basic_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([300], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# Test 2: for 循环内字段累加 — total = sum(1..5) = 15
# =============================================================================
@pl.jit()
def struct_for_accum_kernel(
    out: pl.Tensor[[1], pl.DT_INT32],
):
    acc = pl.struct("Accum", total=0)
    with pl.section_vector():
        for i in pl.range(1, 6):
            acc.total = acc.total + i
        pl.setval(out, 0, acc.total)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_struct_for_accum():
    _check_npu()
    out = torch.zeros(1, device=ST_DEVICE, dtype=torch.int32)
    struct_for_accum_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([15], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# Test 3: for 循环 + if/else 条件分支 — 字段按条件分流
# =============================================================================
@pl.jit()
def struct_conditional_kernel(
    out: pl.Tensor[[2], pl.DT_INT32],
):
    br = pl.struct("Branch", cnt=0, part=0)
    with pl.section_vector():
        for i in pl.range(0, 6):
            br.cnt = br.cnt + 1
            if i < 3:
                br.part = br.part + i
        pl.setval(out, 0, br.cnt)
        pl.setval(out, 1, br.part)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_struct_conditional():
    _check_npu()
    out = torch.zeros(2, device=ST_DEVICE, dtype=torch.int32)
    struct_conditional_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([6, 3], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# Test 4: struct 变量赋值别名 + for 循环 — 通过别名修改，原变量可见
# =============================================================================
@pl.jit()
def struct_alias_for_kernel(
    out: pl.Tensor[[2], pl.DT_INT32],
):
    s = pl.struct("AliasA", val=0, acc=0)
    t = s
    with pl.section_vector():
        for i in pl.range(0, 5):
            t.val = i
            t.acc = t.acc + i
        pl.setval(out, 0, s.val)
        pl.setval(out, 1, s.acc)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_struct_alias_for():
    _check_npu()
    out = torch.zeros(2, device=ST_DEVICE, dtype=torch.int32)
    struct_alias_for_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([4, 10], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# Test 5: 多变量嵌套循环交叉赋值
#   accumulator/source/snapshot/recorder 四个 struct，外层 for (0..3) + 内层 for (0..2)
#   嵌套，内层 if/else 条件分支，accumulator.v←accumulator.v+source.v 自增、
#   snapshot.sum 分支改值、外层每轮用 snapshot.sum 更新 source.v、
#   用 accumulator.acc 记录 recorder.val
#   最终 out: [accumulator.v, source.cnt, snapshot.sum, recorder.val]
# =============================================================================
@pl.jit()
def struct_multi_cross_nested_kernel(
    out: pl.Tensor[[4], pl.DT_INT32],
):
    accumulator = pl.struct("AccumX", v=0, acc=0)
    source = pl.struct("SourceX", v=10, cnt=0)
    snapshot = pl.struct("SnapX", sum=0, flag=0)
    recorder = pl.struct("RecorderX", val=0)
    with pl.section_vector():
        for _i in pl.range(0, 4):
            for j in pl.range(0, 3):
                accumulator.v = accumulator.v + source.v
                accumulator.acc = accumulator.acc + 1
                source.cnt = source.cnt + 1
                if j == 1:
                    snapshot.sum = accumulator.v
                else:
                    snapshot.sum = snapshot.sum + source.cnt
            source.v = snapshot.sum
            recorder.val = accumulator.acc
        pl.setval(out, 0, accumulator.v)
        pl.setval(out, 1, source.cnt)
        pl.setval(out, 2, snapshot.sum)
        pl.setval(out, 3, recorder.val)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_struct_multi_cross_nested():
    _check_npu()
    out = torch.zeros(4, device=ST_DEVICE, dtype=torch.int32)
    struct_multi_cross_nested_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([1161, 12, 901, 12], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# Test 6: 多重别名 + 分支常量赋值 + 条件嵌套循环
#   primary / primary_alias 别名对、secondary / secondary_alias 别名对、
#   constants 用作常量 struct (c1=100, c2=200)、aggregator 累计;
#   外层 for (0..4) + if/elif/else 三分支，分支内各 struct 交叉赋值，
#   else 分支内含内层 for (0..2)，内外层结束后 secondary.a←secondary.a+primary.y、
#   aggregator.v←secondary_alias.b+secondary.a
# =============================================================================
@pl.jit()
def struct_multi_alias_const_kernel(
    out: pl.Tensor[[6], pl.DT_INT32],
):
    primary = pl.struct("PriY", x=0, y=0, z=0)
    primary_alias = primary
    secondary = pl.struct("SecY", a=5, b=10)
    secondary_alias = secondary
    aggregator = pl.struct("AggY", v=0)
    constants = pl.struct("ConstY", c1=100, c2=200)
    with pl.section_vector():
        for i in pl.range(0, 5):
            if i < 2:
                primary.x = primary.x + secondary.a
                primary_alias.y = primary_alias.y + secondary.b + i + constants.c1
            elif i < 4:
                secondary_alias.b = secondary_alias.b + primary_alias.y
                secondary.a = secondary.a + primary.x + constants.c2
                aggregator.v = aggregator.v + secondary.a
            else:
                for j in pl.range(0, 3):
                    primary.z = primary.z + secondary.a * j + secondary_alias.b + constants.c1
                secondary.a = secondary.a + primary_alias.y
                aggregator.v = secondary_alias.b + secondary.a
        pl.setval(out, 0, primary.x)
        pl.setval(out, 1, primary_alias.y)
        pl.setval(out, 2, primary.z)
        pl.setval(out, 3, secondary.a)
        pl.setval(out, 4, secondary_alias.b)
        pl.setval(out, 5, aggregator.v)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_struct_multi_alias_const():
    _check_npu()
    out = torch.zeros(6, device=ST_DEVICE, dtype=torch.int32)
    struct_multi_alias_const_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([10, 221, 2931, 646, 452, 1098], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# Test 7: 引用传递——链式别名 holder→ref1→ref2
#   别名链: ref2→ref1→holder，通过最远端 ref2 修改字段，再通过
#   原始变量 holder 读出，验证引用链路完整传播
#   最终 out: [holder.val, holder.flag] = [3, 70]
# =============================================================================
@pl.jit()
def struct_chain_alias_kernel(
    out: pl.Tensor[[2], pl.DT_INT32],
):
    holder = pl.struct("ChnA", val=0, flag=0)
    ref1 = holder
    ref2 = ref1
    with pl.section_vector():
        for i in pl.range(0, 5):
            if i < 3:
                ref1.val = ref1.val + i
            else:
                ref2.flag = ref2.flag + i * 10
        pl.setval(out, 0, holder.val)
        pl.setval(out, 1, holder.flag)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_struct_chain_alias():
    _check_npu()
    out = torch.zeros(2, device=ST_DEVICE, dtype=torch.int32)
    struct_chain_alias_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([3, 70], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# Test 8: 函数入参传引用
#   真实函数 struct_increase(entry,offset) 和 struct_scale_and_shift(entry,mul,add)
#   接收 struct 参数，在函数体内直接修改字段。循环中按分支调用不同函数，
#   验证 struct 以引用传递——函数内修改立即反映到原始变量。
#   最终 out: [data.val, data.count] = [118, 5]
# =============================================================================


def struct_increase(entry, offset):
    entry.val = entry.val + offset
    entry.count = entry.count + 1


def struct_scale_and_shift(entry, mul, add_val):
    entry.val = entry.val * mul + add_val
    entry.count = entry.count + 1


@pl.jit()
def struct_pass_by_ref_kernel(
    out: pl.Tensor[[2], pl.DT_INT32],
):
    data = pl.struct("PassFn", val=1, count=0)
    with pl.section_vector():
        for i in pl.range(0, 5):
            if i < 2:
                struct_increase(data, i * 10)
            elif i < 4:
                struct_scale_and_shift(data, 2, 5)
            else:
                struct_increase(data, data.val)
        pl.setval(out, 0, data.val)
        pl.setval(out, 1, data.count)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_struct_pass_by_ref():
    _check_npu()
    out = torch.zeros(2, device=ST_DEVICE, dtype=torch.int32)
    struct_pass_by_ref_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([118, 5], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# Standalone runner
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    tests = [
        test_struct_basic,
        test_struct_for_accum,
        test_struct_conditional,
        test_struct_alias_for,
        test_struct_multi_cross_nested,
        test_struct_multi_alias_const,
        test_struct_chain_alias,
        test_struct_pass_by_ref,
    ]
    for t in tests:
        t()
        logging.info("%s passed!", t.__name__)
    logging.info("All struct NPU tests passed!")
