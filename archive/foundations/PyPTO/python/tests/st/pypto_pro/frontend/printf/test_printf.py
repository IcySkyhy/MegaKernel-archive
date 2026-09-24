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

"""pl.printf 调测接口泛化测试 — 覆盖多种格式串、控制流和场景用法。

参考文献:
  docs/zh/pypto_pro/api/Utils-API/debugging/printf.md

控制流覆盖: for / if-else / while / 多格式说明符组合
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
        name = torch.npu.get_device_name()
        if "Ascend950" not in name:
            pytest.skip(f"Device {name} is not A5 (Ascend950). Skip.")
        return True
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
        return False


# =============================================================================
# Test 1: %d/%i 有符号整数 — 多类型标量参数
# =============================================================================
@pl.jit()
def printf_di_kernel(
    out: pl.Tensor[[1], pl.DT_INT32],
    flag: pl.DT_BOOL,
    v_i8: pl.DT_INT8,
    v_i16: pl.DT_INT16,
    v_i32: pl.DT_INT32,
    v_i64: pl.DT_INT64,
):
    with pl.section_vector():
        pl.printf("printf_di: flag=%d i8=%d i16=%d i32=%d i64=%d\n", flag, v_i8, v_i16, v_i32, v_i64)
        pl.printf("printf_i: flag=%i i8=%i i16=%i i32=%i i64=%i\n", flag, v_i8, v_i16, v_i32, v_i64)
        out[0] = 1


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_printf_di():
    """测试 pl.printf 的 %d/%i 格式说明符，验证有符号整数（bool、INT8、INT16、INT32、INT64）标量参数的打印功能。

    输入：out=全零 INT32 张量[1]，flag=True，v_i8=-42，v_i16=-1234，v_i32=-56789，v_i64=-1234567890。
    预期：kernel 内通过 %d 和 %i 各打印一行含上述参数的格式化字符串，
    最后将 out[0] 置为 1；host 侧通过 assert out[0]==1 验证 kernel 正常执行完毕。
    """
    _check_npu()
    logging.info("------------test_printf_di--------------")
    out = torch.zeros(1, device=ST_DEVICE, dtype=torch.int32)
    printf_di_kernel(out, True, -42, -1234, -56789, -1234567890)
    torch.npu.synchronize()
    assert out.tolist()[0] == 1, f"got {out.tolist()}"
    logging.info("printf_di passed!")


# =============================================================================
# Test 2: %u 无符号整数
# =============================================================================
@pl.jit()
def printf_u_kernel(
    out: pl.Tensor[[1], pl.DT_INT32],
    flag: pl.DT_BOOL,
    v_u8: pl.DT_UINT8,
    v_u16: pl.DT_UINT16,
    v_u32: pl.DT_UINT32,
    v_u64: pl.DT_UINT64,
):
    with pl.section_vector():
        pl.printf("printf_u: flag=%u u8=%u u16=%u u32=%u u64=%u\n", flag, v_u8, v_u16, v_u32, v_u64)
        out[0] = 2


# =============================================================================
# Test 3: %f 浮点数 + 精度控制
# =============================================================================
@pl.jit()
def printf_f_kernel(
    out: pl.Tensor[[1], pl.DT_INT32],
    value_f: pl.DT_FP32,
):
    with pl.section_vector():
        pl.printf("printf_f default: val=%f\n", value_f)
        pl.printf("printf_f signed: val=%+08.3f\n", value_f)
        out[0] = 3


# =============================================================================
# Test 4: %x 十六进制
# =============================================================================
@pl.jit()
def printf_x_kernel(
    out: pl.Tensor[[1], pl.DT_INT32],
    v_u8: pl.DT_UINT8,
    v_u16: pl.DT_UINT16,
    v_u32: pl.DT_UINT32,
    v_u64: pl.DT_UINT64,
):
    with pl.section_vector():
        pl.printf("printf_x: u8=%#04x u16=%#06x u32=%#08x u64=%#010x\n", v_u8, v_u16, v_u32, v_u64)
        out[0] = 4


# =============================================================================
# Test 5: %p 指针地址
# =============================================================================
@pl.jit()
def printf_ptr_kernel(
    out: pl.Tensor[[1], pl.DT_INT32],
    data: pl.Ptr[pl.DT_FP16],
):
    with pl.section_vector():
        pl.printf("ptr addr: %p\n", data)
        out[0] = 6


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_printf_ptr():
    """测试 pl.printf 的 %p 格式说明符，验证指针地址的打印功能。

    输入：out=全零 INT32 张量[1]，data=FP16 张量的指针。
    预期：kernel 内通过 %p 打印 data 指针的十六进制地址值，
    最后将 out[0] 置为 6；host 侧通过 assert out[0]==6 验证 kernel 正常执行完毕。
    """
    _check_npu()
    logging.info("------------test_printf_ptr--------------")
    out = torch.zeros(1, device=ST_DEVICE, dtype=torch.int32)
    data = torch.randn(16, device=ST_DEVICE, dtype=torch.float16)
    printf_ptr_kernel(out, data)
    torch.npu.synchronize()
    assert out.tolist()[0] == 6, f"got {out.tolist()}"
    logging.info("printf_ptr passed!")


# =============================================================================
# Test 6: 纯文本 + loc=True
# =============================================================================
@pl.jit()
def printf_text_loc_kernel(
    out: pl.Tensor[[1], pl.DT_INT32],
):
    with pl.section_vector():
        pl.printf("checkpoint: entering kernel\n", loc=True)
        out[0] = 5


# =============================================================================
# Test 7: for 循环内使用 printf
# =============================================================================
@pl.jit()
def printf_for_loop_kernel(
    out: pl.Tensor[[1], pl.DT_INT32],
):
    with pl.section_vector():
        sum_val: pl.DT_INT32 = 0
        for i in pl.range(0, 5):
            pl.printf("for_loop: i=%d sum=%d\n", i, sum_val)
            sum_val = sum_val + i
        out[0] = sum_val


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_printf_for_loop():
    """测试 pl.printf 在 for 循环内的使用，验证 printf 在循环控制流中每轮迭代能正确打印格式字符串。

    输入：out=全零 INT32 张量[1]，无其他参数。kernel 内 for i in range(0,5) 累加 sum_val += i。
    预期：循环 5 次，每次打印 "for_loop: i=%d sum=%d"，最终 sum_val = 0+1+2+3+4 = 10，
    out[0] 置为 sum_val；host 侧通过 torch.equal(out, [10]) 验证结果正确。
    """
    _check_npu()
    logging.info("------------test_printf_for_loop--------------")
    out = torch.zeros(1, device=ST_DEVICE, dtype=torch.int32)
    printf_for_loop_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([10], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"
    logging.info("printf_for_loop passed!")


# =============================================================================
# Test 8: if/else 分支内使用 printf
# =============================================================================
@pl.jit()
def printf_if_else_kernel(
    out: pl.Tensor[[1], pl.DT_INT32],
    flag: pl.DT_BOOL,
):
    with pl.section_vector():
        if flag:
            pl.printf("if_branch: flag is True\n")
            out[0] = 100
        else:
            pl.printf("else_branch: flag is False\n")
            out[0] = -100


# =============================================================================
# Test 9: while 循环内使用 printf
# =============================================================================
@pl.jit()
def printf_while_kernel(
    out: pl.Tensor[[1], pl.DT_INT32],
):
    with pl.section_vector():
        x: pl.DT_INT32 = 0
        while x < 4:
            pl.printf("while_loop: x=%d\n", x)
            x = x + 1
        out[0] = x


# =============================================================================
# Test 10: 带 loc=True + 多格式组合
# =============================================================================
@pl.jit()
def printf_loc_combo_kernel(
    out: pl.Tensor[[1], pl.DT_INT32],
    v_i32: pl.DT_INT32,
    v_fp: pl.DT_FP32,
):
    with pl.section_vector():
        pl.printf("loc_combo: pre v_i32=%d v_fp=%f\n", v_i32, v_fp, loc=True)
        out[0] = v_i32
        pl.printf("loc_combo: post setval out[0]=%d\n", v_i32, loc=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    tests = [
        test_printf_di,
        test_printf_ptr,
        test_printf_for_loop,
    ]
    for t in tests:
        t()
        logging.info(f"{t.__name__} passed!")
    logging.info("\nAll pl.printf NPU tests passed!")
