# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""NPU coverage for active ``pl.struct_array`` indexing and control-flow scenarios.

Covers indexed access, loops, conditionals, break handling, cross-array
assignment, aggregation, aliasing, nested ring updates, and reference passing.
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
# Test 1: 常索引读写 — 3 槽独立赋值后读回
# =============================================================================
@pl.jit()
def struct_array_basic_kernel(
    out: pl.Tensor[[1], pl.DT_INT32],
):
    arr = pl.struct_array(3, "Slot1", a=0, b=0)
    with pl.section_vector():
        arr[0].a = 10
        arr[0].b = 20
        arr[1].a = 30
        arr[1].b = 40
        arr[2].a = 50
        arr[2].b = 60
        c0 = arr[0]
        c1 = arr[1]
        c2 = arr[2]
        pl.setval(out, 0, c0.a + c1.b + c2.a)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_struct_array_basic():
    _check_npu()
    out = torch.zeros(1, device=ST_DEVICE, dtype=torch.int32)
    struct_array_basic_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([100], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# Test 2: for 循环遍历 — 两轮循环（先写各槽字段值，再读回写输出）
# =============================================================================
@pl.jit()
def struct_array_for_loop_kernel(
    out: pl.Tensor[[4], pl.DT_INT32],
):
    ctx_arr = pl.struct_array(4, "Ctx2", vid=0, count=0)
    with pl.section_vector():
        for i in pl.range(0, 4):
            ctx = ctx_arr[i]
            ctx.vid = i * 10
            ctx.count = i + 1
        for i in pl.range(0, 4):
            ctx = ctx_arr[i]
            pl.setval(out, i, ctx.vid + ctx.count)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_struct_array_for_loop():
    _check_npu()
    out = torch.zeros(4, device=ST_DEVICE, dtype=torch.int32)
    struct_array_for_loop_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([1, 12, 23, 34], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# Test 3: for + if/else 条件分支 — 按槽位索引分流赋值
# =============================================================================
@pl.jit()
def struct_array_conditional_kernel(
    out: pl.Tensor[[3], pl.DT_INT32],
):
    arr = pl.struct_array(6, "Tag", tag=0, val=0)
    with pl.section_vector():
        for i in pl.range(0, 6):
            ctx = arr[i]
            if i < 3:
                ctx.tag = 1
                ctx.val = i
            else:
                ctx.tag = 2
                ctx.val = i * 10
        v0 = arr[0]
        v2 = arr[2]
        v4 = arr[4]
        pl.setval(out, 0, v0.tag + v0.val)
        pl.setval(out, 1, v2.tag + v2.val)
        pl.setval(out, 2, v4.tag + v4.val)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_struct_array_conditional():
    _check_npu()
    out = torch.zeros(3, device=ST_DEVICE, dtype=torch.int32)
    struct_array_conditional_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([1, 3, 42], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# Test 4: for + break — struct_array 环形索引 + 循环中断
# =============================================================================
@pl.jit()
def struct_array_break_kernel(
    out: pl.Tensor[[1], pl.DT_INT32],
):
    buf = pl.struct_array(4, "Buf", snap=0)
    with pl.section_vector():
        for i in pl.range(0, 100):
            ctx = buf[i % 4]
            ctx.snap = i
            if i >= 6:
                break
        c0 = buf[0]
        c1 = buf[1]
        c2 = buf[2]
        c3 = buf[3]
        pl.setval(out, 0, c0.snap + c1.snap + c2.snap + c3.snap)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_struct_array_break():
    _check_npu()
    out = torch.zeros(1, device=ST_DEVICE, dtype=torch.int32)
    struct_array_break_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([18], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# Test 5: 两个同类型 struct_array 相互赋值 — for 循环内从 arr_a 读字段写入 arr_b
# =============================================================================
@pl.jit()
def struct_array_cross_assign_kernel(
    out: pl.Tensor[[4], pl.DT_INT32],
):
    arr_a = pl.struct_array(4, "SrcA", x=0, y=0)
    arr_b = pl.struct_array(4, "DstB", v=0)
    with pl.section_vector():
        for i in pl.range(0, 4):
            ctx_a = arr_a[i]
            ctx_a.x = i * 5
            ctx_a.y = i + 1
        for i in pl.range(0, 4):
            ctx_a = arr_a[i]
            ctx_b = arr_b[i]
            ctx_b.v = ctx_a.x + ctx_a.y
        for i in pl.range(0, 4):
            b = arr_b[i]
            pl.setval(out, i, b.v)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_struct_array_cross_assign():
    _check_npu()
    out = torch.zeros(4, device=ST_DEVICE, dtype=torch.int32)
    struct_array_cross_assign_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([1, 7, 13, 19], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# Test 6: struct 聚合 + struct_array 分槽 — for + if/else 混合
# =============================================================================
@pl.jit()
def struct_and_array_aggregate_kernel(
    out: pl.Tensor[[2], pl.DT_INT32],
):
    agg = pl.struct("Agg", sm=0, cnt=0)
    arr = pl.struct_array(8, "Slot", val=0)
    with pl.section_vector():
        for i in pl.range(0, 8):
            slot = arr[i]
            slot.val = i * 3
            if i % 2 == 0:
                agg.sm = agg.sm + slot.val
                agg.cnt = agg.cnt + 1
        pl.setval(out, 0, agg.sm)
        pl.setval(out, 1, agg.cnt)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_struct_and_array_aggregate():
    _check_npu()
    out = torch.zeros(2, device=ST_DEVICE, dtype=torch.int32)
    struct_and_array_aggregate_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([36, 4], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# Test 7: struct_array slot 变量赋值别名 + for 循环 — 通过别名修改，原 slot 可见
# =============================================================================
@pl.jit()
def struct_array_alias_for_kernel(
    out: pl.Tensor[[4], pl.DT_INT32],
):
    arr = pl.struct_array(4, "AliasArrA", v=0)
    with pl.section_vector():
        for i in pl.range(0, 4):
            slot = arr[i]
            alias = slot
            slot.v = i * 10
            alias.v = alias.v + i + 1
        for i in pl.range(0, 4):
            s = arr[i]
            pl.setval(out, i, s.v)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_struct_array_alias_for():
    _check_npu()
    out = torch.zeros(4, device=ST_DEVICE, dtype=torch.int32)
    struct_array_alias_for_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([1, 12, 23, 34], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# Test 8: 三数组嵌套循环 + 环形索引交叉赋值 + 常数 struct
#   bank_a / bank_b / bank_c 三个 size=4 的 struct_array，params 常量 (mul=3, offset=2);
#   三轮外层 for (0..2)，每轮先填 bank_a 槽位值，再根据 i<2 条件分流累加到 bank_b，
#   然后 bank_c 快照 bank_b，最后用 params.offset 做环形索引把 bank_c 值回写到 bank_a
# =============================================================================
@pl.jit()
def struct_array_triple_ring_kernel(
    out: pl.Tensor[[4], pl.DT_INT32],
):
    bank_a = pl.struct_array(4, "BankA", val=0)
    bank_b = pl.struct_array(4, "BankB", acc=0)
    bank_c = pl.struct_array(4, "BankC", snap=0)
    params = pl.struct("Params", mul=3, offset=2)
    with pl.section_vector():
        for outer in pl.range(0, 3):
            for i in pl.range(0, 4):
                slot_a = bank_a[i]
                slot_a.val = params.mul * outer + i + 1
            for i in pl.range(0, 4):
                a = bank_a[i]
                b = bank_b[i]
                if i < 2:
                    b.acc = b.acc + a.val
                else:
                    b.acc = b.acc + a.val * 2
            for i in pl.range(0, 4):
                b = bank_b[i]
                c = bank_c[i]
                c.snap = b.acc
        for i in pl.range(0, 4):
            a = bank_a[i]
            c = bank_c[(i + params.offset) % 4]
            a.val = a.val + c.snap
        a2 = bank_a[2]
        b0 = bank_b[0]
        b3 = bank_b[3]
        c1 = bank_c[1]
        pl.setval(out, 0, a2.val)
        pl.setval(out, 1, b0.acc)
        pl.setval(out, 2, b3.acc)
        pl.setval(out, 3, c1.snap)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_struct_array_triple_ring():
    _check_npu()
    out = torch.zeros(4, device=ST_DEVICE, dtype=torch.int32)
    struct_array_triple_ring_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([21, 12, 42, 15], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# Test 9: 双数组 while 循环 + 别名交换 + 条件分支嵌套 + 尾随 for 交叉累加
#   primary_arr / backup_arr 两个 size=5 的 struct_array，params 常量 (base=10, step=2);
#   while idx<5 内取 slot 别名、if/elif/else 三分支分流写入 backup_arr，
#   else 分支内含 for 嵌套累加 + 跨槽引用; while 结束后 for 循环做交叉累加回写
# =============================================================================
@pl.jit()
def struct_array_while_alias_nested_kernel(
    out: pl.Tensor[[6], pl.DT_INT32],
):
    primary_arr = pl.struct_array(5, "Primary", x=0, y=0)
    backup_arr = pl.struct_array(5, "Backup", n=0)
    params = pl.struct("Params2", base=10, step=2)
    tracker = pl.struct("Tracker", v=0, acc=0)
    idx = 0
    with pl.section_vector():
        while idx < 5:
            slot_a = primary_arr[idx]
            slot_b = backup_arr[idx]
            alias_to_a = slot_a
            alias_to_a.x = params.base + idx * params.step
            alias_to_a.y = idx * idx
            if idx < 2:
                slot_b.n = alias_to_a.x + alias_to_a.y + params.base
            elif idx < 4:
                ref = backup_arr[idx - 2]
                slot_b.n = alias_to_a.x * 2 + ref.n
            else:
                tracker.v = 0
                for _j in pl.range(0, 3):
                    tracker.v = tracker.v + alias_to_a.x
                alias_to_a.y = tracker.v
                slot_b.n = alias_to_a.y + params.step
            idx = idx + 1
        for k in pl.range(0, 3):
            pa = primary_arr[k]
            bb = backup_arr[k + 2]
            bb.n = pa.x + bb.n
            tracker.acc = tracker.acc + bb.n
        p0 = primary_arr[0]
        p4 = primary_arr[4]
        b2 = backup_arr[2]
        b4 = backup_arr[4]
        pl.setval(out, 0, p0.x)
        pl.setval(out, 1, p0.y)
        pl.setval(out, 2, b2.n)
        pl.setval(out, 3, b4.n)
        pl.setval(out, 4, p4.y)
        pl.setval(out, 5, tracker.acc)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_struct_array_while_alias_nested():
    _check_npu()
    out = torch.zeros(6, device=ST_DEVICE, dtype=torch.int32)
    struct_array_while_alias_nested_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([10, 0, 58, 70, 54, 195], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# Test 10: 引用传递——链式别名 slot→handle→alias
#   arr 4 个槽，每轮取 slot=arr[i]，handle=slot，alias=handle
#   (链式别名: alias→handle→slot)，通过 alias 写入，再通过
#   arr[i] 读出验证数据已回写原槽
#   最终 out per-slot: [val+tag] = [6, 16, 42, 62]
# =============================================================================
@pl.jit()
def struct_array_chain_alias_kernel(
    out: pl.Tensor[[4], pl.DT_INT32],
):
    arr = pl.struct_array(4, "ArrChn", val=0, tag=0)
    with pl.section_vector():
        for i in pl.range(0, 4):
            slot = arr[i]
            handle = slot
            alias = handle
            if i < 2:
                alias.val = i * 10 + 5
                handle.tag = 1
            else:
                alias.val = i * 20
                handle.tag = 2
        for i in pl.range(0, 4):
            s = arr[i]
            pl.setval(out, i, s.val + s.tag)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_struct_array_chain_alias():
    _check_npu()
    out = torch.zeros(4, device=ST_DEVICE, dtype=torch.int32)
    struct_array_chain_alias_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([6, 16, 42, 62], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# Test 11: 函数入参传引用
#   真实函数 arr_fill(arr,idx,base,step) 和 arr_accum(arr_a,arr_b,idx,factor)
#   接收 struct_array 整体+索引参数，函数体内通过 arr[idx] 访问槽位。
#   循环中依次调用 arr_init_slot 填值、arr_transfer 搬运，验证
#   通过函数参数修改的槽位数据已写回原数组。
#   最终 out per-slot: [dst_arr[i].acc] = [0, 22, 66, 132]
# =============================================================================


def arr_fill(arr, idx, base, step):
    slot = arr[idx]
    slot.val = base + step
    slot.tag = base


def arr_accum(arr_a, arr_b, idx, factor):
    slot_a = arr_a[idx]
    slot_b = arr_b[idx]
    slot_b.acc = slot_a.val * factor + slot_b.acc


@pl.jit()
def struct_array_pass_by_ref_kernel(
    out: pl.Tensor[[4], pl.DT_INT32],
):
    src_arr = pl.struct_array(4, "SrcFn", val=0, tag=0)
    dst_arr = pl.struct_array(4, "DstFn", acc=0)
    with pl.section_vector():
        for i in pl.range(0, 4):
            arr_fill(src_arr, i, i * 10, i)
        for i in pl.range(0, 4):
            arr_accum(src_arr, dst_arr, i, i + 1)
        for i in pl.range(0, 4):
            s = dst_arr[i]
            pl.setval(out, i, s.acc)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_struct_array_pass_by_ref():
    _check_npu()
    out = torch.zeros(4, device=ST_DEVICE, dtype=torch.int32)
    struct_array_pass_by_ref_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([0, 22, 66, 132], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# Standalone runner
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    tests = [
        test_struct_array_basic,
        test_struct_array_for_loop,
        test_struct_array_conditional,
        test_struct_array_break,
        test_struct_array_cross_assign,
        test_struct_and_array_aggregate,
        test_struct_array_alias_for,
        test_struct_array_triple_ring,
        test_struct_array_while_alias_nested,
        test_struct_array_chain_alias,
        test_struct_array_pass_by_ref,
    ]
    for t in tests:
        t()
        logging.info("%s passed!", t.__name__)
    logging.info("All struct_array NPU tests passed!")
