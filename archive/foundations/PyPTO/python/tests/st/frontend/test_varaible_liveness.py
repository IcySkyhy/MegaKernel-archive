#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""测试 pypto.frontend.jit 代码块内定义变量的生效情况"""

import os

import torch
import torch_npu

import pypto


def gen_data(shape):
    x = torch.empty(shape, dtype=torch.float32).uniform_(-1, 1)
    expected = x + 2
    return x, expected


@pypto.frontend.jit()
def kernel_if(
    a: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    result: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(64, 64)

    if True:
        b = a + 1

    result[:] = b + 1


@pypto.frontend.jit()
def kernel_loop(
    a: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    result: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(64, 64)

    for _ in range(2):
        b = a + 1
        a[:] = a + 1
    result[:] = b * 1.0


def run_kernel_test(kernel_func):
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    torch.manual_seed(42)
    m, n = 16, 64
    shape = (m, n)
    x, expected = gen_data(shape)

    x_npu = x.to(device=f'npu:{device_id}')
    result = torch.zeros(shape, dtype=torch.float32, device=device_id)

    kernel_func(x_npu, result)
    torch_npu.npu.synchronize()

    result_cpu = result.cpu()
    assert torch.allclose(result_cpu, expected, atol=0.0001, rtol=0.0078125)


@pypto.options(pass_options={"enable_slice": True})
def test_if_variable_scope():
    run_kernel_test(kernel_if)


@pypto.options(pass_options={"enable_slice": True})
def test_range_unroll():
    run_kernel_test(kernel_loop)


if __name__ == "__main__":
    test_range_unroll()
    test_if_variable_scope()
