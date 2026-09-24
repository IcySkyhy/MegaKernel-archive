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
"""
Matmul BASIC_TESTS test script.
Supports both pytest and direct execution modes.
"""

import os

import pytest
from testcase.matmul_test_case import BASIC_TESTS, NZ_FORMAT_TESTS, MatmulConfig
import torch
import torch.nn.functional as functional
import torch_npu

import pypto


@pypto.frontend.jit(debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0})
def matmul_pto_kernel(
    a_tensor: pypto.Tensor(),
    b_tensor: pypto.Tensor(),
    out_tensor: pypto.Tensor(),
    config: MatmulConfig,
):
    m, k, n = config.shape
    m_view, n_view = config.view_shape
    pypto.set_cube_tile_shapes(*config.tile_shape)

    m_loop = (m + m_view - 1) // m_view
    n_loop = (n + n_view - 1) // n_view

    for m_idx in pypto.loop(0, m_loop, 1, name="LOOP_L0_mIdx", idx_name="m_idx"):
        for n_idx in pypto.loop(0, n_loop, 1, name="LOOP_L0_nIdx", idx_name="n_idx"):
            if config.a_trans:
                a_view = a_tensor[:, m_idx * m_view:m_idx * m_view + m_view]
            else:
                a_view = a_tensor[m_idx * m_view:m_idx * m_view + m_view, :]

            if config.b_trans:
                b_view = b_tensor[n_idx * n_view:n_idx * n_view + n_view, :]
            else:
                b_view = b_tensor[:, n_idx * n_view:n_idx * n_view + n_view]

            out_view = pypto.matmul(
                a_view,
                b_view,
                out_dtype=config.out_dtype,
                a_trans=config.a_trans,
                b_trans=config.b_trans,
            )

            out_tensor[
                m_idx * m_view:m_idx * m_view + m_view,
                n_idx * n_view:n_idx * n_view + n_view,
            ] = out_view


def run_matmul_test(case: dict):
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)

    config = MatmulConfig.from_test_case(case)

    m, k, n = config.shape
    a_shape = [k, m] if config.a_trans else [m, k]
    b_shape = [n, k] if config.b_trans else [k, n]
    c_shape = config.out_shape

    a_dtype = MatmulConfig.get_torch_dtype(case["a_dtype"])
    b_dtype = MatmulConfig.get_torch_dtype(case["b_dtype"])
    c_dtype = MatmulConfig.get_torch_dtype(case["c_dtype"])

    # Step 1: 按指定 dtype 随机生成输入
    if a_dtype == torch.int8:
        a_tensor_cpu = torch.randint(-5, 6, a_shape, dtype=a_dtype)
        b_tensor_cpu = torch.randint(-5, 6, b_shape, dtype=b_dtype)
    else:
        a_tensor_cpu = torch.rand(a_shape, dtype=a_dtype)
        b_tensor_cpu = torch.rand(b_shape, dtype=b_dtype)

    # Step 2: 计算 golden
    a_cpu = a_tensor_cpu.T if config.a_trans else a_tensor_cpu
    b_cpu = b_tensor_cpu.T if config.b_trans else b_tensor_cpu

    # 统一转换到累积精度再计算
    # 整数用 int32 累积，浮点用 fp32 累积
    accum_dtype = torch.int32 if a_dtype == torch.int8 else torch.float32
    golden = torch.matmul(a_cpu.to(accum_dtype), b_cpu.to(accum_dtype)).to(c_dtype)

    # Step 3: 将 CPU tensor 转为 NPU tensor
    a_tensor = a_tensor_cpu.to(f"npu:{device_id}")
    b_tensor = b_tensor_cpu.to(f"npu:{device_id}")
    c_tensor = torch.zeros(c_shape, dtype=c_dtype, device=f"npu:{device_id}")

    # Step 4: 将 ND format tensor 转为 NZ format tensor
    if config.a_format == "NZ":
        a_tensor = torch_npu.npu_format_cast(a_tensor, 29)
    if config.b_format == "NZ":
        b_tensor = torch_npu.npu_format_cast(b_tensor, 29)

    # Step 5: 调用 PTO kernel
    matmul_pto_kernel(a_tensor, b_tensor, c_tensor, config)

    # Step 6: 比对
    padding_m = abs(config.out_shape[0] - m)
    padding_n = abs(config.out_shape[1] - n)
    golden = functional.pad(golden, ((0, padding_n, 0, padding_m)), "constant")
    atol, rtol = MatmulConfig.get_tolerance(case["c_dtype"])
    assert torch.allclose(c_tensor.cpu(), golden.cpu(), atol=atol, rtol=rtol), (
        f"Test case {case['id']} ({case['name']}) failed"
    )


@pytest.mark.parametrize("case", [pytest.param(case, marks=pytest.mark.soc(*case["products"])) for case in BASIC_TESTS])
def test_matmul_basic(case: dict):
    run_matmul_test(case)


@pytest.mark.parametrize(
    "case", [pytest.param(case, marks=pytest.mark.soc(*case["products"])) for case in NZ_FORMAT_TESTS]
)
@pypto.options(pass_options={"enable_slice": False})
def test_matmul_basic(case: dict):  # noqa: F811
    run_matmul_test(case)


def run_matmul_demo(run_mode):
    m_size, k_size, n_size = 256, 256, 256
    m_view_size, n_view_size = 128, 128

    if run_mode == "npu":
        mode = pypto.RunMode.NPU
    elif run_mode == "sim":
        mode = pypto.RunMode.SIM
    else:
        raise ValueError(f"Invalid run_mode: {run_mode}. Must be 'npu' or 'sim'")

    @pypto.frontend.jit(
        debug_options={"runtime_debug_mode": 1, "compile_debug_mode": 1}, runtime_options={"run_mode": mode}
    )
    def matmul_demo_kernel(
        a: pypto.Tensor([], pypto.DT_FP16),
        b: pypto.Tensor([], pypto.DT_FP16),
        out: pypto.Tensor([], pypto.DT_FP16),
    ):
        pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])

        m_loop = (m_size + m_view_size - 1) // m_view_size
        n_loop = (n_size + n_view_size - 1) // n_view_size

        for m_idx in pypto.loop(0, m_loop, 1, name="LOOP_L0_mIdx", idx_name="m_idx"):
            for n_idx in pypto.loop(0, n_loop, 1, name="LOOP_L0_nIdx", idx_name="n_idx"):
                a_view = a[m_idx * m_view_size:m_idx * m_view_size + m_view_size, :]
                b_view = b[:, n_idx * n_view_size:n_idx * n_view_size + n_view_size]
                out_view = pypto.matmul(a_view, b_view, pypto.DT_FP16)
                out[
                    m_idx * m_view_size:m_idx * m_view_size + m_view_size,
                    n_idx * n_view_size:n_idx * n_view_size + n_view_size,
                ] = out_view

    device = "npu:0" if run_mode == "npu" else "cpu"
    a = torch.randn([m_size, k_size], dtype=torch.float16, device=device)
    b = torch.randn([k_size, n_size], dtype=torch.float16, device=device)
    out = torch.empty(m_size, n_size, dtype=torch.float16, device=device)
    matmul_demo_kernel(a, b, out)


if __name__ == "__main__":
    run_matmul_demo("npu")
