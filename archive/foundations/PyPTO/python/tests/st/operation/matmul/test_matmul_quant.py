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
Matmul QUANT_TESTS test script.
Supports both pytest and direct execution modes.
"""

import os
import struct

import numpy as np
import pytest
from testcase.matmul_quant_test_case import PERCHANNEL_TESTS, PERTENSOR_TESTS, MatmulQuantConfig
import torch
import torch_npu

import pypto


def fixpipe_mask_scale(scale_input):
    mask = 0xFFFFE000

    if isinstance(scale_input, torch.Tensor):
        scale_np = scale_input.cpu().numpy()
        tensor_data = scale_np.view(np.uint32)
        tensor_data = tensor_data & mask
        golden_scale = tensor_data.view(np.float32)
        scale_input_uint64 = tensor_data.astype(np.uint64)

        golden_scale_torch = torch.from_numpy(golden_scale).to(torch.float32)
        scale_input_torch = torch.from_numpy(scale_input_uint64).to(torch.uint64)

        return scale_input_torch, golden_scale_torch
    else:
        packed = struct.pack('f', np.float32(scale_input))
        as_int = struct.unpack('I', packed)[0]
        masked_int = as_int & mask
        golden_scale = struct.unpack('f', struct.pack('I', masked_int))[0]

        scale_input_float = golden_scale

        return scale_input_float, np.float32(golden_scale)


@pypto.frontend.jit(debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0})
def matmul_quant_pertensor_kernel(
    a_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC]),
    b_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC]),
    bias_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC]),
    out_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC]),
    config: MatmulQuantConfig,
):
    scale = float(MatmulQuantConfig.scale_value)
    m, k, n = config.shape
    m_view, n_view = config.view_shape
    pypto.set_cube_tile_shapes(*config.tile_shape)

    m_loop = (m + m_view - 1) // m_view
    n_loop = (n + n_view - 1) // n_view

    relu_mode = pypto.ReLuType.RELU if config.relu_type == 1 else pypto.ReLuType.NO_RELU

    for m_idx in pypto.loop(0, m_loop, 1, name="QUANT_LOOP_L0_mIdx", idx_name="quant_m_idx"):
        for n_idx in pypto.loop(0, n_loop, 1, name="QUANT_LOOP_L0_nIdx", idx_name="quant_n_idx"):
            if config.a_trans:
                a_view = a_tensor[0:k, m_idx * m_view:m_idx * m_view + m_view]
            else:
                a_view = a_tensor[m_idx * m_view:m_idx * m_view + m_view, 0:k]

            if config.b_trans:
                b_view = b_tensor[n_idx * n_view:n_idx * n_view + n_view, 0:k]
            else:
                b_view = b_tensor[0:k, n_idx * n_view:n_idx * n_view + n_view]
            bias_view = bias_tensor[0:1, n_idx * n_view:n_idx * n_view + n_view]
            out_view = pypto.matmul(
                a_view,
                b_view,
                out_dtype=config.out_dtype,
                a_trans=config.a_trans,
                b_trans=config.b_trans,
                extend_params={"bias_tensor": bias_view, "scale": scale, "relu_type": relu_mode},
            )

            out_tensor[
                m_idx * m_view:m_idx * m_view + m_view,
                n_idx * n_view:n_idx * n_view + n_view,
            ] = out_view


@pypto.frontend.jit(debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0})
def matmul_quant_perchannel_kernel(
    a_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC]),
    b_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC]),
    scale_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC]),
    bias_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC]),
    out_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC]),
    config: MatmulQuantConfig,
):
    m, k, n = config.shape
    m_view, n_view = config.view_shape
    pypto.set_cube_tile_shapes(*config.tile_shape)

    m_loop = (m + m_view - 1) // m_view
    n_loop = (n + n_view - 1) // n_view

    relu_mode = pypto.ReLuType.RELU if config.relu_type == 1 else pypto.ReLuType.NO_RELU

    for m_idx in pypto.loop(0, m_loop, 1, name="LOOP_L0_mIdx", idx_name="m_idx"):
        for n_idx in pypto.loop(0, n_loop, 1, name="LOOP_L0_nIdx", idx_name="n_idx"):
            if config.b_trans:
                b_view = b_tensor[n_idx * n_view:n_idx * n_view + n_view, 0:k]
            else:
                b_view = b_tensor[0:k, n_idx * n_view:n_idx * n_view + n_view]
            if config.a_trans:
                a_view = a_tensor[0:k, m_idx * m_view:m_idx * m_view + m_view]
            else:
                a_view = a_tensor[m_idx * m_view:m_idx * m_view + m_view, 0:k]

            scale_view = scale_tensor[0:1, n_idx * n_view:n_idx * n_view + n_view]
            bias_view = bias_tensor[0:1, n_idx * n_view:n_idx * n_view + n_view]
            out_view = pypto.matmul(
                a_view,
                b_view,
                out_dtype=config.out_dtype,
                a_trans=config.a_trans,
                b_trans=config.b_trans,
                extend_params={"bias_tensor": bias_view, "scale_tensor": scale_view, "relu_type": relu_mode},
            )

            out_tensor[
                m_idx * m_view:m_idx * m_view + m_view,
                n_idx * n_view:n_idx * n_view + n_view,
            ] = out_view


def run_matmul_quant_test(case: dict):
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)

    config = MatmulQuantConfig.from_test_case(case)

    m, k, n = config.shape
    a_shape = [k, m] if config.a_trans else [m, k]
    b_shape = [n, k] if config.b_trans else [k, n]
    c_shape = [m, n]
    bias_shape = [1, n]

    a_dtype = MatmulQuantConfig.get_torch_dtype(case["a_dtype"])
    b_dtype = MatmulQuantConfig.get_torch_dtype(case["b_dtype"])
    c_dtype = MatmulQuantConfig.get_torch_dtype(case["c_dtype"])
    bias_dtype = MatmulQuantConfig.get_torch_dtype(case["bias_dtype"])

    if a_dtype == torch.int8:
        a_tensor_cpu = torch.randint(-5, 6, a_shape, dtype=a_dtype)
        b_tensor_cpu = torch.randint(-5, 6, b_shape, dtype=b_dtype)
        bias_tensor_cpu = torch.randint(-5, 6, bias_shape, dtype=bias_dtype)
        accum_dtype = torch.int32
    else:
        a_tensor_cpu = torch.rand(a_shape, dtype=a_dtype)
        b_tensor_cpu = torch.rand(b_shape, dtype=b_dtype)
        bias_tensor_cpu = torch.rand(bias_shape, dtype=bias_dtype)
        accum_dtype = torch.float

    a_cpu = a_tensor_cpu.T if config.a_trans else a_tensor_cpu
    b_cpu = b_tensor_cpu.T if config.b_trans else b_tensor_cpu

    matmul_result = torch.matmul(a_cpu.to(accum_dtype), b_cpu.to(accum_dtype)) + bias_tensor_cpu.to(accum_dtype)
    if config.relu_type == 1:
        matmul_result = torch.relu(matmul_result)

    a_tensor_npu = a_tensor_cpu.to(f"npu:{device_id}")
    b_tensor_npu = b_tensor_cpu.to(f"npu:{device_id}")
    bias_tensor_npu = bias_tensor_cpu.to(f"npu:{device_id}")
    if config.a_format == "NZ":
        a_tensor_npu = torch_npu.npu_format_cast(a_tensor_npu, 29)
    if config.b_format == "NZ":
        b_tensor_npu = torch_npu.npu_format_cast(b_tensor_npu, 29)

    if config.quant_type == 1:
        scale_dequant = np.random.uniform(0.1, 2.0)
        scale_input, golden_scale = fixpipe_mask_scale(scale_dequant)
        golden_scale_torch = torch.tensor(golden_scale, dtype=torch.float32)
        MatmulQuantConfig.scale_value = scale_input

        if c_dtype == torch.int8:
            golden = torch.round((matmul_result * golden_scale_torch).clamp(-128, 127).to(c_dtype))
        else:
            golden = (matmul_result * golden_scale_torch).to(c_dtype)

        c_tensor = torch.zeros(c_shape, dtype=c_dtype, device=f"npu:{device_id}")

        matmul_quant_pertensor_kernel(a_tensor_npu, b_tensor_npu, bias_tensor_npu, c_tensor, config)

    else:
        scale_dequant = torch.from_numpy(np.random.uniform(0.1, 2.0, [1, n]).astype(np.float32))
        scale_input, golden_scale = fixpipe_mask_scale(scale_dequant)

        if c_dtype == torch.int8:
            golden = torch.round((matmul_result * golden_scale).clamp(-128, 127).to(c_dtype))
            scale_tensor = torch_npu.npu_trans_quant_param(golden_scale.to(f"npu:{device_id}"))
        else:
            golden = (matmul_result * golden_scale).to(c_dtype)
            scale_tensor = scale_input.to(f"npu:{device_id}")

        c_tensor = torch.zeros(c_shape, dtype=c_dtype, device=f"npu:{device_id}")

        matmul_quant_perchannel_kernel(a_tensor_npu, b_tensor_npu, scale_tensor, bias_tensor_npu, c_tensor, config)

    atol, rtol = MatmulQuantConfig.get_tolerance(case["c_dtype"])
    assert torch.allclose(c_tensor.cpu(), golden.cpu(), atol=atol, rtol=rtol), (
        f"Test case {case['id']} ({case['name']}) failed"
    )


@pytest.mark.parametrize(
    "case",
    [pytest.param(case, marks=pytest.mark.soc(*case["products"])) for case in PERTENSOR_TESTS + PERCHANNEL_TESTS],
)
def test_matmul_quant(case: dict):
    if case["id"] not in {"PERTENSOR01", "PERTENSOR02", "PERCHANNEL01", "PERCHANNEL03"}:
        run_matmul_quant_test(case)
        return
    with pypto.options(pass_options={"enable_slice": False}):
        run_matmul_quant_test(case)


def run_matmul_quant_pertensor_demo():
    m_size, k_size, n_size = 256, 256, 256
    m_view_size, n_view_size = 128, 128

    scale_dequant = np.random.uniform(0.1, 2.0)
    scale_input, golden_scale = fixpipe_mask_scale(scale_dequant)

    @pypto.frontend.jit(debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0})
    def matmul_quant_pertensor_demo_kernel(
        a: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC]),
        b: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC]),
        out: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC]),
        scale_value,
    ):
        pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])

        m_loop = (m_size + m_view_size - 1) // m_view_size
        n_loop = (n_size + n_view_size - 1) // n_view_size

        scale = float(scale_value)

        for m_idx in pypto.loop(0, m_loop, 1, name="LOOP_L0_mIdx", idx_name="m_idx"):
            for n_idx in pypto.loop(0, n_loop, 1, name="LOOP_L0_nIdx", idx_name="n_idx"):
                a_view = a[m_idx * m_view_size:m_idx * m_view_size + m_view_size, 0:k_size]
                b_view = b[0:k_size, n_idx * n_view_size:n_idx * n_view_size + n_view_size]
                out_view = pypto.matmul(
                    a_view, b_view, pypto.DT_FP16, extend_params={"scale": scale, "relu_type": pypto.ReLuType.NO_RELU}
                )
                out[
                    m_idx * m_view_size:m_idx * m_view_size + m_view_size,
                    n_idx * n_view_size:n_idx * n_view_size + n_view_size,
                ] = out_view

    a = torch.randint(-5, 6, [m_size, k_size], dtype=torch.int8, device="npu:0")
    b = torch.randint(-5, 6, [k_size, n_size], dtype=torch.int8, device="npu:0")
    out = torch.empty(m_size, n_size, dtype=torch.float16, device="npu:0")
    matmul_quant_pertensor_demo_kernel(a, b, out, scale_input)


def run_matmul_quant_perchannel_demo():
    m_size, k_size, n_size = 256, 256, 256
    m_view_size, n_view_size = 128, 128

    scale_dequant = torch.from_numpy(np.random.uniform(0.1, 2.0, [1, n_size]).astype(np.float32))
    scale_input, golden_scale = fixpipe_mask_scale(scale_dequant)

    @pypto.frontend.jit(debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0})
    def matmul_quant_perchannel_demo_kernel(
        a: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC]),
        b: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC]),
        scale: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC]),
        out: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC]),
    ):
        pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])

        m_loop = (m_size + m_view_size - 1) // m_view_size
        n_loop = (n_size + n_view_size - 1) // n_view_size

        for m_idx in pypto.loop(0, m_loop, 1, name="LOOP_L0_mIdx", idx_name="m_idx"):
            for n_idx in pypto.loop(0, n_loop, 1, name="LOOP_L0_nIdx", idx_name="n_idx"):
                b_view = b[0:k_size, n_idx * n_view_size:n_idx * n_view_size + n_view_size]
                a_view = a[m_idx * m_view_size:m_idx * m_view_size + m_view_size, 0:k_size]
                scale_view = scale[0:1, n_idx * n_view_size:n_idx * n_view_size + n_view_size]
                out_view = pypto.matmul(
                    a_view,
                    b_view,
                    pypto.DT_FP16,
                    extend_params={"scale_tensor": scale_view, "relu_type": pypto.ReLuType.NO_RELU},
                )
                out[
                    m_idx * m_view_size:m_idx * m_view_size + m_view_size,
                    n_idx * n_view_size:n_idx * n_view_size + n_view_size,
                ] = out_view

    a = torch.randint(-5, 6, [m_size, k_size], dtype=torch.int8, device="npu:0")
    b = torch.randint(-5, 6, [k_size, n_size], dtype=torch.int8, device="npu:0")
    scale_tensor = scale_input.to("npu:0")
    out = torch.empty(m_size, n_size, dtype=torch.float16, device="npu:0")
    matmul_quant_perchannel_demo_kernel(a, b, scale_tensor, out)


if __name__ == "__main__":
    run_matmul_quant_pertensor_demo()
    run_matmul_quant_perchannel_demo()
