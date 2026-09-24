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
ScaledMM ST test script.
Supports both pytest and direct execution modes.
"""

from dataclasses import dataclass
import os

import pytest
from testcase.scaled_mm_mxfp8_test_case import SCALED_MM_TESTS, ScaledMMConfig
import torch
import torch.nn.functional as functional
import torch_npu

import pypto

K_BLOCK_SIZE_64 = 64
K_BLOCK_SIZE_32 = 32
SHAPE_DIM_2 = 2


@dataclass
class ScaledMMInputs:
    a_npu: torch.Tensor
    b_npu: torch.Tensor
    scale_a_npu: torch.Tensor
    scale_b_npu: torch.Tensor
    bias_npu: torch.Tensor
    golden: torch.Tensor


@pypto.frontend.jit(debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0})
def scaled_mm_kernel_no_bias(
    a_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC]),
    b_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC]),
    out_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC]),
    scale_a_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC], dtype=pypto.DT_FP8E8M0),
    scale_b_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC], dtype=pypto.DT_FP8E8M0),
    config: ScaledMMConfig,
):
    m, n = out_tensor.shape
    k = config.ori_shape[1]
    vm, vn = config.view_shape
    m_loop = (m + vm - 1) // vm
    n_loop = (n + vn - 1) // vn
    scale_k = (k + K_BLOCK_SIZE_64 - 1) // K_BLOCK_SIZE_64

    pypto.set_vec_tile_shapes(config.m_tile_shape[0], config.n_tile_shape[0])
    for m_idx in pypto.loop(0, m_loop, 1, name="LOOP_LO_mIdx", idx_name="m_idx"):
        for n_idx in pypto.loop(0, n_loop, 1, name="LOOP_L1_nIdx", idx_name="n_idx"):
            m_offset = m_idx * vm
            n_offset = n_idx * vn

            if config.a_trans:
                a_view = pypto.view(a_tensor, [k, vm], [0, m_offset], valid_shape=[k, min(vm, m - m_offset)])
            else:
                a_view = pypto.view(a_tensor, [vm, k], [m_offset, 0], valid_shape=[min(vm, m - m_offset), k])

            if config.b_trans:
                b_view = pypto.view(b_tensor, [vn, k], [n_offset, 0], valid_shape=[min(vn, n - n_offset), k])
            else:
                b_view = pypto.view(b_tensor, [k, vn], [0, n_offset], valid_shape=[k, min(vn, n - n_offset)])

            pypto.set_vec_tile_shapes(config.m_tile_shape[0], config.n_tile_shape[0], 32)
            if config.scale_a_trans:
                scale_a_view = pypto.view(
                    scale_a_tensor, [scale_k, vm, 2], [0, m_offset, 0], valid_shape=[scale_k, min(vm, m - m_offset), 2]
                )
            else:
                scale_a_view = pypto.view(
                    scale_a_tensor, [vm, scale_k, 2], [m_offset, 0, 0], valid_shape=[min(vm, m - m_offset), scale_k, 2]
                )

            if config.scale_b_trans:
                scale_b_view = pypto.view(
                    scale_b_tensor, [vn, scale_k, 2], [n_offset, 0, 0], valid_shape=[min(vn, n - n_offset), scale_k, 2]
                )
            else:
                scale_b_view = pypto.view(
                    scale_b_tensor, [scale_k, vn, 2], [0, n_offset, 0], valid_shape=[scale_k, min(vn, n - n_offset), 2]
                )

            tile_shape = (config.m_tile_shape, config.k_tile_shape, config.n_tile_shape)
            pypto.set_cube_tile_shapes(*tile_shape, config.enable_ksplit)

            out_view = pypto.scaled_mm(
                a_view,
                b_view,
                config.out_dtype,
                scale_a_view,
                scale_b_view,
                a_trans=config.a_trans,
                b_trans=config.b_trans,
                scale_a_trans=config.scale_a_trans,
                scale_b_trans=config.scale_b_trans,
                c_matrix_nz=config.c_format == "NZ",
            )
            pypto.assemble(out_view, [m_offset, n_offset], out_tensor)


@pypto.frontend.jit(debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0})
def scaled_mm_kernel_with_bias(
    a_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC]),
    b_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC]),
    out_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC]),
    scale_a_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC], dtype=pypto.DT_FP8E8M0),
    scale_b_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC], dtype=pypto.DT_FP8E8M0),
    bias_tensor: pypto.Tensor([pypto.STATIC, pypto.DYNAMIC]),
    config: ScaledMMConfig,
):
    k = config.ori_shape[1]
    m, n = out_tensor.shape
    vm, vn = config.view_shape
    n_loop = (n + vn - 1) // vn
    m_loop = (m + vm - 1) // vm
    scale_k = (k + K_BLOCK_SIZE_64 - 1) // K_BLOCK_SIZE_64

    pypto.set_vec_tile_shapes(config.m_tile_shape[0], config.n_tile_shape[0])
    for m_idx in pypto.loop(0, m_loop, 1, name="LOOP_LO_mIdx", idx_name="m_idx"):
        for n_idx in pypto.loop(0, n_loop, 1, name="LOOP_L1_nIdx", idx_name="n_idx"):
            m_offset = m_idx * vm
            n_offset = n_idx * vn

            if config.b_trans:
                b_view = pypto.view(b_tensor, [vn, k], [n_offset, 0], valid_shape=[min(vn, n - n_offset), k])
            else:
                b_view = pypto.view(b_tensor, [k, vn], [0, n_offset], valid_shape=[k, min(vn, n - n_offset)])

            if config.a_trans:
                a_view = pypto.view(a_tensor, [k, vm], [0, m_offset], valid_shape=[k, min(vm, m - m_offset)])
            else:
                a_view = pypto.view(a_tensor, [vm, k], [m_offset, 0], valid_shape=[min(vm, m - m_offset), k])

            bias_view = bias_tensor[:, n_offset:n_offset + vn]

            pypto.set_vec_tile_shapes(config.m_tile_shape[0], config.n_tile_shape[0], 32)

            if config.scale_b_trans:
                scale_b_view = pypto.view(
                    scale_b_tensor, [vn, scale_k, 2], [n_offset, 0, 0], valid_shape=[min(vn, n - n_offset), scale_k, 2]
                )
            else:
                scale_b_view = pypto.view(
                    scale_b_tensor, [scale_k, vn, 2], [0, n_offset, 0], valid_shape=[scale_k, min(vn, n - n_offset), 2]
                )
            if config.scale_a_trans:
                scale_a_view = pypto.view(
                    scale_a_tensor, [scale_k, vm, 2], [0, m_offset, 0], valid_shape=[scale_k, min(vm, m - m_offset), 2]
                )
            else:
                scale_a_view = pypto.view(
                    scale_a_tensor, [vm, scale_k, 2], [m_offset, 0, 0], valid_shape=[min(vm, m - m_offset), scale_k, 2]
                )

            extend_params = {'bias_tensor': bias_view}
            tile_shape = (config.m_tile_shape, config.k_tile_shape, config.n_tile_shape)
            pypto.set_cube_tile_shapes(*tile_shape, config.enable_ksplit)

            out_view = pypto.scaled_mm(
                a_view,
                b_view,
                config.out_dtype,
                scale_a_view,
                scale_b_view,
                a_trans=config.a_trans,
                b_trans=config.b_trans,
                scale_a_trans=config.scale_a_trans,
                scale_b_trans=config.scale_b_trans,
                c_matrix_nz=config.c_format == "NZ",
                extend_params=extend_params,
            )
            pypto.assemble(out_view, [m_offset, n_offset], out_tensor)


def _process_scale_tensors(scale_a_cpu, scale_b_cpu, config):
    m, k, n = config.ori_shape
    scale_k_32 = (k + K_BLOCK_SIZE_64 - 1) // K_BLOCK_SIZE_64 * SHAPE_DIM_2

    if config.scale_a_trans:
        scale_a_tmp = torch.transpose(scale_a_cpu, -2, -1).reshape(scale_k_32, m).T
    else:
        scale_a_tmp = scale_a_cpu.view(m, scale_k_32)

    if config.scale_b_trans:
        scale_b_tmp = scale_b_cpu.view(n, scale_k_32).T
    else:
        scale_b_tmp = torch.transpose(scale_b_cpu, -2, -1).reshape(scale_k_32, n)

    scale_a_tmp = scale_a_tmp.to(torch.float32).repeat_interleave(32, dim=1)
    scale_b_tmp = scale_b_tmp.to(torch.float32).repeat_interleave(32, dim=0)

    return scale_a_tmp, scale_b_tmp


def prepare_inputs(config: ScaledMMConfig, device_id: int):
    m, k, n = config.ori_shape
    output_m, output_n = config.output_shape
    a_shape = [k, m] if config.a_trans else [m, k]
    b_shape = [n, k] if config.b_trans else [k, n]

    scale_k = (k + K_BLOCK_SIZE_64 - 1) // K_BLOCK_SIZE_64
    padding_k = scale_k * K_BLOCK_SIZE_64 - k
    scale_a_shape = [scale_k, m, SHAPE_DIM_2] if config.scale_a_trans else [m, scale_k, SHAPE_DIM_2]
    scale_b_shape = [n, scale_k, SHAPE_DIM_2] if config.scale_b_trans else [scale_k, n, SHAPE_DIM_2]

    torch_in_dtype = ScaledMMConfig.pto_to_torch(config.in_dtype)
    mat_a_cpu = torch.rand(a_shape, dtype=torch.float32).uniform_(-3, 3).to(torch_in_dtype)
    mat_b_cpu = torch.rand(b_shape, dtype=torch.float32).uniform_(-3, 3).to(torch_in_dtype)
    scale_a_cpu = torch.rand(scale_a_shape, dtype=torch.float32).uniform_(0, 1).to(torch.float8_e8m0fnu)
    scale_b_cpu = torch.rand(scale_b_shape, dtype=torch.float32).uniform_(0, 1).to(torch.float8_e8m0fnu)

    bias_cpu = torch.rand([1, output_n], dtype=torch.float32).uniform_(-3, 3) if config.has_bias else None
    scale_a_tmp, scale_b_tmp = _process_scale_tensors(scale_a_cpu, scale_b_cpu, config)

    mat_b_tmp = mat_b_cpu.to(torch.float32).T if config.b_trans else mat_b_cpu.to(torch.float32)
    mat_b_tmp = functional.pad(mat_b_tmp, ((0, 0, 0, padding_k)), "constant")
    mat_b_tmp = scale_b_tmp * mat_b_tmp

    mat_a_tmp = mat_a_cpu.to(torch.float32).T if config.a_trans else mat_a_cpu.to(torch.float32)
    mat_a_tmp = functional.pad(mat_a_tmp, ((0, padding_k, 0, 0)), "constant")
    mat_a_tmp = mat_a_tmp * scale_a_tmp

    golden = torch.matmul(mat_a_tmp, mat_b_tmp)
    padding_m = abs(output_m - m)
    padding_n = abs(output_n - n)
    golden = functional.pad(golden, ((0, padding_n, 0, padding_m)), "constant")
    if config.has_bias:
        golden = golden + bias_cpu.to(golden.dtype).repeat_interleave(output_m, dim=0)

    out_torch_dtype = ScaledMMConfig.pto_to_torch(config.out_dtype)
    golden = golden.to(out_torch_dtype)

    device = f"npu:{device_id}"
    a_npu = mat_a_cpu.to(device)
    b_npu = mat_b_cpu.to(device)

    if config.a_format == "NZ":
        a_npu = torch_npu.npu_format_cast(a_npu, 29)
    if config.b_format == "NZ":
        b_npu = torch_npu.npu_format_cast(b_npu, 29)

    scale_a_npu = scale_a_cpu.to(device)
    scale_b_npu = scale_b_cpu.to(device)
    bias_npu = bias_cpu.to(device) if bias_cpu is not None else None

    return ScaledMMInputs(a_npu, b_npu, scale_a_npu, scale_b_npu, bias_npu, golden)


def run_scaled_mm_test(case: dict):
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)

    config = ScaledMMConfig.from_test_case(case)
    inputs = prepare_inputs(config, device_id)

    _m, _n = config.ori_shape[0], config.ori_shape[2]
    output_m, output_n = config.output_shape
    out_torch_dtype = ScaledMMConfig.pto_to_torch(config.out_dtype)
    out_npu = torch.zeros([output_m, output_n], dtype=out_torch_dtype, device=f"npu:{device_id}")

    if config.has_bias:
        scaled_mm_kernel_with_bias(
            inputs.a_npu, inputs.b_npu, out_npu, inputs.scale_a_npu, inputs.scale_b_npu, inputs.bias_npu, config
        )
    else:
        scaled_mm_kernel_no_bias(inputs.a_npu, inputs.b_npu, out_npu, inputs.scale_a_npu, inputs.scale_b_npu, config)

    atol, rtol = ScaledMMConfig.get_tolerance(case["out_dtype"])
    assert torch.allclose(out_npu.cpu(), inputs.golden, atol=atol, rtol=rtol), (
        f"Test case {case['id']} ({case['name']}) failed"
    )


@pytest.mark.parametrize(
    "case", [pytest.param(case, marks=pytest.mark.soc(*case["products"])) for case in SCALED_MM_TESTS]
)
@pypto.options(pass_options={"enable_slice": False})
def test_scaled_mm(case: dict):
    run_scaled_mm_test(case)


def run_scaled_mm_demo(run_mode):
    m_size, k_size, n_size = 256, 128, 64
    vm_view_size, vn_view_size = 128, 32

    if run_mode == "npu":
        mode = pypto.RunMode.NPU
    elif run_mode == "sim":
        mode = pypto.RunMode.SIM
    else:
        raise ValueError(f"Invalid run_mode: {run_mode}. Must be 'npu' or 'sim'")

    @pypto.frontend.jit(
        debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0}, runtime_options={"run_mode": mode}
    )
    def scaled_mm_demo_kernel(
        a_tensor: pypto.Tensor([], pypto.DT_FP8E4M3),
        b_tensor: pypto.Tensor([], pypto.DT_FP8E4M3),
        out_tensor: pypto.Tensor([], pypto.DT_FP16),
        scale_a_tensor: pypto.Tensor([], pypto.DT_FP8E8M0),
        scale_b_tensor: pypto.Tensor([], pypto.DT_FP8E8M0),
    ):
        pypto.set_cube_tile_shapes([64, 64], [64, 64], [64, 64])
        pypto.set_vec_tile_shapes(64, 64)

        m_loop = (m_size + vm_view_size - 1) // vm_view_size
        n_loop = (n_size + vn_view_size - 1) // vn_view_size

        for m_idx in pypto.loop(0, m_loop, 1, name="LOOP_LO_mIdx", idx_name="m_idx"):
            for n_idx in pypto.loop(0, n_loop, 1, name="LOOP_L1_nIdx", idx_name="n_idx"):
                m_offset = m_idx * vm_view_size
                n_offset = n_idx * vn_view_size

                a_view = a_tensor[m_offset:m_offset + vm_view_size, :]
                b_view = b_tensor[n_offset:n_offset + vn_view_size, :]

                scale_a_view = scale_a_tensor[m_offset:m_offset + vm_view_size, :, :]
                scale_b_view = scale_b_tensor[:, n_offset:n_offset + vn_view_size, :]

                out_view = pypto.scaled_mm(
                    a_view,
                    b_view,
                    pypto.DT_FP16,
                    scale_a_view,
                    scale_b_view,
                    a_trans=False,
                    b_trans=True,
                    scale_a_trans=False,
                    scale_b_trans=False,
                    c_matrix_nz=False,
                )
                out_tensor[m_offset:m_offset + vm_view_size, n_offset:n_offset + vn_view_size] = out_view

    scale_k = k_size // 64
    device = "npu:0" if run_mode == "npu" else "cpu"
    a = torch.randn([m_size, k_size], dtype=torch.float32).uniform_(-3, 3).to(torch.float8_e4m3fn).to(device)
    b = torch.randn([n_size, k_size], dtype=torch.float32).uniform_(-3, 3).to(torch.float8_e4m3fn).to(device)
    scale_a = torch.randn([m_size, scale_k, 2], dtype=torch.float32).uniform_(0, 1).to(torch.float8_e8m0fnu).to(device)
    scale_b = torch.randn([scale_k, n_size, 2], dtype=torch.float32).uniform_(0, 1).to(torch.float8_e8m0fnu).to(device)
    out = torch.zeros([m_size, n_size], dtype=torch.float16).to(device)

    scaled_mm_demo_kernel(a, b, out, scale_a, scale_b)


if __name__ == "__main__":
    run_scaled_mm_demo("npu")
