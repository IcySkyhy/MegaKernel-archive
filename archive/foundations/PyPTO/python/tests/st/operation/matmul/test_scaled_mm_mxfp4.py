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
FP4 ScaledMM ST test script.
Supports both pytest and direct execution modes.
"""

from dataclasses import dataclass
import os
from typing import Optional

import numpy as np
import pytest
from testcase.scaled_mm_mxfp4_test_case import MXFP4_TESTS, MXFP4Config
import torch
import torch.nn.functional as functional
import torch_npu

import pypto

K_BLOCK_SIZE_64 = 64
K_BLOCK_SIZE_32 = 32
SHAPE_DIM_2 = 2


@dataclass
class FP4TestData:
    mat_a: torch.Tensor
    scale_a: torch.Tensor
    mat_b: torch.Tensor
    scale_b: torch.Tensor
    bias: Optional[torch.Tensor]
    golden: torch.Tensor


@dataclass
class FP4Shapes:
    a_shape: list
    b_shape: list
    scale_a_shape: list
    scale_b_shape: list


@dataclass
class FP4TensorShapes:
    a_shape: list
    a_shape_ori: list
    b_shape: list
    b_shape_ori: list
    scale_a_shape: list
    scale_b_shape: list


@dataclass
class FP4GoldenComputeParams:
    mat_a: torch.Tensor
    mat_b: torch.Tensor
    scale_a: torch.Tensor
    scale_b: torch.Tensor
    config: MXFP4Config
    bias: Optional[torch.Tensor]
    m: int
    n: int
    k: int
    a_shape_ori: list
    b_shape_ori: list


@dataclass
class ViewInfoParams:
    trans: bool
    dim1: int
    dim2: int
    offset1: int
    offset2: int
    min1: int
    min2: int
    is_scale: bool = False


def convert_pypto_dtype_to_torch(dtype):
    if dtype == pypto.DataType.DT_FP8E4M3:
        return torch.float8_e4m3fn
    elif dtype == pypto.DataType.DT_FP8E5M2:
        return torch.float8_e5m2
    elif dtype == pypto.DataType.DT_FP4_E2M1X2 or dtype == pypto.DataType.DT_FP4_E2M1:
        return torch.uint8
    else:
        raise ValueError(f"Unsupported pypto DataType: {dtype}")


def unpack_fp4_to_float32(packed_uint8_tensor, tensor_shape):
    low_nibble = packed_uint8_tensor & 0x0F
    high_nibble = (packed_uint8_tensor >> 4) & 0x0F
    unpacked_indices = torch.empty(tensor_shape, dtype=torch.uint8)
    unpacked_indices[:, 0::2] = low_nibble
    unpacked_indices[:, 1::2] = high_nibble
    fp4_values = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=torch.float32
    )
    float32_matrix = fp4_values[unpacked_indices.view(-1).to(torch.int)].view(tensor_shape)
    return float32_matrix


def scaledmm_pypto_basic(input_dtype):
    pypto_a_dtype = input_dtype
    pypto_b_dtype = input_dtype

    @pypto.frontend.jit(debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0})
    def scaledmm_basic_kernel(
        input_a_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], dtype=pypto_a_dtype),
        input_b_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], dtype=pypto_b_dtype),
        output_c_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC]),
        scale_a_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC], dtype=pypto.DT_FP8E8M0),
        scale_b_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC], dtype=pypto.DT_FP8E8M0),
        output_dtype,
        input_a_trans,
        input_b_trans,
        scale_a_trans,
        scale_b_trans,
        c_matrix_nz,
        enable_k_split,
        k,
        view_shape,
        tile_shape,
    ):
        m, n = output_c_tensor.shape
        vm, vn = view_shape
        m_loop = (m + vm - 1) // vm
        n_loop = (n + vn - 1) // vn
        scale_k = (k + K_BLOCK_SIZE_64 - 1) // K_BLOCK_SIZE_64
        pypto.set_vec_tile_shapes(tile_shape[0][0], tile_shape[2][0])
        for m_idx in pypto.loop(0, m_loop, 1, name="LOOP_LO_mIdx", idx_name="m_idx"):
            for n_idx in pypto.loop(0, n_loop, 1, name="LOOP_L1_nIdx", idx_name="n_idx"):
                m_offset = m_idx * vm
                n_offset = n_idx * vn
                if input_a_trans:
                    view_shape_m = [k, vm]
                    input_a_view = pypto.view(
                        input_a_tensor, view_shape_m, [0, m_offset], valid_shape=[k, (m - m_offset).min(vm)]
                    )
                else:
                    view_shape_m = [vm, k]
                    input_a_view = pypto.view(
                        input_a_tensor, view_shape_m, [m_offset, 0], valid_shape=[(m - m_offset).min(vm), k]
                    )
                if input_b_trans:
                    view_shape_n = [vn, k]
                    input_b_view = pypto.view(
                        input_b_tensor, view_shape_n, [n_offset, 0], valid_shape=[(n - n_offset).min(vn), k]
                    )
                else:
                    view_shape_n = [k, vn]
                    input_b_view = pypto.view(
                        input_b_tensor, view_shape_n, [0, n_offset], valid_shape=[k, (n - n_offset).min(vn)]
                    )
                pypto.set_vec_tile_shapes(tile_shape[0][0], tile_shape[2][0], 32)
                if scale_a_trans:
                    view_shape_scale_a = [scale_k, vm, 2]
                    scale_a_view = pypto.view(
                        scale_a_tensor,
                        view_shape_scale_a,
                        [0, m_offset, 0],
                        valid_shape=[scale_k, (m - m_offset).min(vm), 2],
                    )
                else:
                    view_shape_scale_a = [vm, scale_k, 2]
                    scale_a_view = pypto.view(
                        scale_a_tensor,
                        view_shape_scale_a,
                        [m_offset, 0, 0],
                        valid_shape=[(m - m_offset).min(vm), scale_k, 2],
                    )
                if scale_b_trans:
                    view_shape_scale_b = [vn, scale_k, 2]
                    scale_b_view = pypto.view(
                        scale_b_tensor,
                        view_shape_scale_b,
                        [n_offset, 0, 0],
                        valid_shape=[(n - n_offset).min(vn), scale_k, 2],
                    )
                else:
                    view_shape_scale_b = [scale_k, vn, 2]
                    scale_b_view = pypto.view(
                        scale_b_tensor,
                        view_shape_scale_b,
                        [0, n_offset, 0],
                        valid_shape=[scale_k, (n - n_offset).min(vn), 2],
                    )
                pypto.set_cube_tile_shapes(*tile_shape, enable_k_split)
                output_view = pypto.scaled_mm(
                    input_a_view,
                    input_b_view,
                    output_dtype,
                    scale_a_view,
                    scale_b_view,
                    a_trans=input_a_trans,
                    b_trans=input_b_trans,
                    scale_a_trans=scale_a_trans,
                    scale_b_trans=scale_b_trans,
                    c_matrix_nz=c_matrix_nz,
                )
                output_offsets = [m_offset, n_offset]
                pypto.assemble(output_view, output_offsets, output_c_tensor)

    return scaledmm_basic_kernel


def get_view_info(params: ViewInfoParams):
    if params.trans:
        view_shape = [params.dim1, params.dim2]
        offset = [params.offset1, params.offset2]
        valid_shape = [params.min1, params.min2]
    else:
        view_shape = [params.dim2, params.dim1]
        offset = [params.offset2, params.offset1]
        valid_shape = [params.min2, params.min1]
    if params.is_scale:
        view_shape.append(2)
        offset.append(0)
        valid_shape.append(2)
    return view_shape, offset, valid_shape


def scaledmm_pypto_bias(input_dtype):
    pypto_a_dtype = input_dtype
    pypto_b_dtype = input_dtype

    @pypto.frontend.jit(debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0})
    def scaledmm_bias_kernel(
        input_a_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], dtype=pypto_a_dtype),
        input_b_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], dtype=pypto_b_dtype),
        output_c_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC]),
        scale_a_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC]),
        scale_b_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC]),
        bias_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC]),
        output_dtype,
        input_b_trans,
        input_a_trans,
        scale_a_trans,
        scale_b_trans,
        c_matrix_nz,
        enable_k_split,
        k,
        view_shape,
        tile_shape,
    ):
        vm, vn = view_shape
        m, n = output_c_tensor.shape
        n_loop = (n + vn - 1) // vn
        m_loop = (m + vm - 1) // vm
        scale_k = (k + K_BLOCK_SIZE_64 - 1) // K_BLOCK_SIZE_64

        pypto.set_vec_tile_shapes(tile_shape[0][0], tile_shape[2][0])
        for m_idx in pypto.loop(0, m_loop, 1, name="LOOP_LO_mIdx", idx_name="m_idx"):
            for n_idx in pypto.loop(0, n_loop, 1, name="LOOP_L1_nIdx", idx_name="n_idx"):
                n_offset = n_idx * vn
                m_offset = m_idx * vm

                min_n_vn = (n - n_offset).min(vn)
                min_m_vm = (m - m_offset).min(vm)

                view_shape_n, off_b, valid_b = get_view_info(
                    ViewInfoParams(input_b_trans, vn, k, n_offset, 0, min_n_vn, k)
                )
                input_b_view = pypto.view(input_b_tensor, view_shape_n, off_b, valid_shape=valid_b)

                view_shape_m, off_a, valid_a = get_view_info(
                    ViewInfoParams(input_a_trans, k, vm, 0, m_offset, k, min_m_vm)
                )
                input_a_view = pypto.view(input_a_tensor, view_shape_m, off_a, valid_shape=valid_a)

                bias_view = bias_tensor[:, n_offset:n_offset + vn]
                extend_params = {'bias_tensor': bias_view}

                pypto.set_vec_tile_shapes(tile_shape[0][0], tile_shape[2][0], 32)

                view_shape_sb, off_sb, valid_sb = get_view_info(
                    ViewInfoParams(scale_b_trans, vn, scale_k, n_offset, 0, min_n_vn, scale_k, is_scale=True)
                )
                scale_b_view = pypto.view(scale_b_tensor, view_shape_sb, off_sb, valid_shape=valid_sb)

                view_shape_sa, off_sa, valid_sa = get_view_info(
                    ViewInfoParams(scale_a_trans, scale_k, vm, 0, m_offset, scale_k, min_m_vm, is_scale=True)
                )
                scale_a_view = pypto.view(scale_a_tensor, view_shape_sa, off_sa, valid_shape=valid_sa)

                pypto.set_cube_tile_shapes(*tile_shape, enable_k_split)
                output_view = pypto.scaled_mm(
                    input_a_view,
                    input_b_view,
                    output_dtype,
                    scale_a_view,
                    scale_b_view,
                    a_trans=input_a_trans,
                    b_trans=input_b_trans,
                    scale_a_trans=scale_a_trans,
                    scale_b_trans=scale_b_trans,
                    c_matrix_nz=c_matrix_nz,
                    extend_params=extend_params,
                )
                output_offsets = [m_offset, n_offset]
                pypto.assemble(output_view, output_offsets, output_c_tensor)

    return scaledmm_bias_kernel


def compute_shapes(config: MXFP4Config) -> FP4TensorShapes:
    m, k = config.a_shape
    n = config.b_shape[1]
    scale_k = (k + K_BLOCK_SIZE_64 - 1) // K_BLOCK_SIZE_64
    a_shape = [k, m // 2] if config.a_trans else [m, k // 2]
    a_shape_ori = [k, m] if config.a_trans else [m, k]
    b_shape = [n, k // 2] if config.b_trans else [k, n // 2]
    b_shape_ori = [n, k] if config.b_trans else [k, n]
    scale_a_shape = [scale_k, m, SHAPE_DIM_2] if config.scale_a_trans else [m, scale_k, SHAPE_DIM_2]
    scale_b_shape = [n, scale_k, SHAPE_DIM_2] if config.scale_b_trans else [scale_k, n, SHAPE_DIM_2]

    return FP4TensorShapes(
        a_shape=a_shape,
        a_shape_ori=a_shape_ori,
        b_shape=b_shape,
        b_shape_ori=b_shape_ori,
        scale_a_shape=scale_a_shape,
        scale_b_shape=scale_b_shape,
    )


def apply_format_cast(tensor, fmt: str):
    if fmt == "NZ":
        return torch_npu.npu_format_cast(tensor, 29)
    return tensor


def compute_golden(params: FP4GoldenComputeParams) -> torch.Tensor:
    mat_a = params.mat_a
    mat_b = params.mat_b
    scale_a = params.scale_a
    scale_b = params.scale_b
    config = params.config
    bias = params.bias
    m = params.m
    n = params.n
    k = params.k
    scale_k = (k + K_BLOCK_SIZE_64 - 1) // K_BLOCK_SIZE_64
    padding_k = scale_k * K_BLOCK_SIZE_64 - k
    a_shape_ori = params.a_shape_ori
    b_shape_ori = params.b_shape_ori

    if not config.scale_a_trans:
        scale_a_tmp = scale_a.view(m, scale_k * SHAPE_DIM_2)
    else:
        scale_a_tmp = torch.transpose(scale_a, -2, -1).reshape(scale_k * SHAPE_DIM_2, m).T

    if not config.scale_b_trans:
        scale_b_tmp = torch.transpose(scale_b, -2, -1).reshape(scale_k * SHAPE_DIM_2, n)
    else:
        scale_b_tmp = scale_b.view(n, scale_k * SHAPE_DIM_2).T

    scale_a_tmp = np.repeat(scale_a_tmp.to(torch.float32), 32, axis=1)  # (m, k)
    scale_b_tmp = np.repeat(scale_b_tmp.to(torch.float32), 32, axis=0)  # (k, n)

    mat_a_fp4 = unpack_fp4_to_float32(mat_a, a_shape_ori)
    mat_a_tmp = mat_a_fp4.to(torch.float32).T if config.a_trans else mat_a_fp4.to(torch.float32)
    mat_a_tmp = functional.pad(mat_a_tmp, ((0, padding_k, 0, 0)), "constant")
    mat_a_tmp = mat_a_tmp * scale_a_tmp.to(torch.float32)

    mat_b_fp4 = unpack_fp4_to_float32(mat_b, b_shape_ori)
    mat_b_tmp = mat_b_fp4.to(torch.float32).T if config.b_trans else mat_b_fp4.to(torch.float32)
    mat_b_tmp = functional.pad(mat_b_tmp, ((0, 0, 0, padding_k)), "constant")
    mat_b_tmp = scale_b_tmp.to(torch.float32) * mat_b_tmp

    golden = torch.matmul(mat_a_tmp.to(torch.float32), mat_b_tmp.to(torch.float32))

    if config.has_bias:
        bias_tmp = np.repeat(bias, m, axis=0)
        golden += bias_tmp

    return golden


def prepare_fp4_inputs(config: MXFP4Config, device_id: int) -> FP4TestData:
    m, k = config.a_shape
    n = config.b_shape[1]

    tensor_shapes = compute_shapes(config)

    torch_in_dtype = convert_pypto_dtype_to_torch(config.in_dtype)
    mat_a = torch.randn(tensor_shapes.a_shape, dtype=torch.float32).uniform_(-3, 3).to(torch_in_dtype)
    scale_a = torch.randn(tensor_shapes.scale_a_shape, dtype=torch.float32).uniform_(0, 1).to(torch.float8_e8m0fnu)
    mat_b = torch.randn(tensor_shapes.b_shape, dtype=torch.float32).uniform_(-3, 3).to(torch_in_dtype)
    scale_b = torch.randn(tensor_shapes.scale_b_shape, dtype=torch.float32).uniform_(0, 1).to(torch.float8_e8m0fnu)

    bias = None
    if config.has_bias:
        bias = torch.randn([1, n], dtype=torch.float16).uniform_(-3, 3)

    golden_params = FP4GoldenComputeParams(
        mat_a=mat_a,
        mat_b=mat_b,
        scale_a=scale_a,
        scale_b=scale_b,
        config=config,
        bias=bias,
        m=m,
        n=n,
        k=k,
        a_shape_ori=tensor_shapes.a_shape_ori,
        b_shape_ori=tensor_shapes.b_shape_ori,
    )
    golden = compute_golden(golden_params)

    mat_a = apply_format_cast(mat_a, config.a_format)
    mat_b = apply_format_cast(mat_b, config.b_format)
    device = f"npu:{device_id}"

    return FP4TestData(
        mat_a=mat_a.to(device),
        scale_a=scale_a.to(device),
        mat_b=mat_b.to(device),
        scale_b=scale_b.to(device),
        bias=bias.to(device) if bias is not None else None,
        golden=golden,
    )


def run_fp4_test(case: dict):
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)

    config = MXFP4Config.from_test_case(case)
    data = prepare_fp4_inputs(config, device_id)

    m, n = config.a_shape[0], config.b_shape[1]
    out_torch_dtype = MXFP4Config.pto_to_torch(config.out_dtype)
    out = torch.zeros([m, n], dtype=out_torch_dtype, device=f"npu:{device_id}")

    if config.has_bias:
        scaledmm_pypto_bias(config.in_dtype)(
            data.mat_a,
            data.mat_b,
            out,
            data.scale_a,
            data.scale_b,
            data.bias,
            config.out_dtype,
            config.b_trans,
            config.a_trans,
            config.scale_a_trans,
            config.scale_b_trans,
            False,
            False,
            config.a_shape[1],
            config.view_shape,
            [config.m_tile_shape, config.k_tile_shape, config.n_tile_shape],
        )
    else:
        scaledmm_pypto_basic(config.in_dtype)(
            data.mat_a,
            data.mat_b,
            out,
            data.scale_a,
            data.scale_b,
            config.out_dtype,
            config.a_trans,
            config.b_trans,
            config.scale_a_trans,
            config.scale_b_trans,
            False,
            False,
            config.a_shape[1],
            config.view_shape,
            [config.m_tile_shape, config.k_tile_shape, config.n_tile_shape],
        )

    atol, rtol = MXFP4Config.get_tolerance(case["out_dtype"])
    assert torch.allclose(out.cpu().to(torch.float32), data.golden, atol=atol, rtol=rtol), (
        f"Test case {case['id']} ({case['name']}) failed"
    )


@pytest.mark.parametrize("case", [pytest.param(case, marks=pytest.mark.soc(*case["products"])) for case in MXFP4_TESTS])
@pypto.options(pass_options={"enable_slice": False})
def test_fp4(case: dict):
    run_fp4_test(case)


def run_fp4_demo(run_mode):
    m_size, k_size, n_size = 128, 256, 256
    vm_view_size, vn_view_size = 64, 128

    if run_mode == "npu":
        mode = pypto.RunMode.NPU
    elif run_mode == "sim":
        mode = pypto.RunMode.SIM
    else:
        raise ValueError(f"Invalid run_mode: {run_mode}. Must be 'npu' or 'sim'")

    @pypto.frontend.jit(
        debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0}, runtime_options={"run_mode": mode}
    )
    def fp4_demo_kernel(
        a_tensor: pypto.Tensor([], pypto.DT_FP4_E2M1),
        b_tensor: pypto.Tensor([], pypto.DT_FP4_E2M1),
        out_tensor: pypto.Tensor([], pypto.DT_FP16),
        scale_a_tensor: pypto.Tensor([], pypto.DT_FP8E8M0),
        scale_b_tensor: pypto.Tensor([], pypto.DT_FP8E8M0),
    ):
        pypto.set_cube_tile_shapes([64, 64], [64, 256], [64, 128])

        m_loop = (m_size + vm_view_size - 1) // vm_view_size
        n_loop = (n_size + vn_view_size - 1) // vn_view_size

        for m_idx in pypto.loop(0, m_loop, 1, name="LOOP_L0_mIdx", idx_name="m_idx"):
            for n_idx in pypto.loop(0, n_loop, 1, name="LOOP_L0_nIdx", idx_name="n_idx"):
                m_offset = m_idx * vm_view_size
                n_offset = n_idx * vn_view_size

                a_view = a_tensor[m_offset:m_offset + vm_view_size, :]
                b_view = b_tensor[n_offset:n_offset + vn_view_size, :]

                scale_a_view = scale_a_tensor[:, m_offset:m_offset + vm_view_size, :]
                scale_b_view = scale_b_tensor[:, n_offset:n_offset + vn_view_size, :]

                out_view = pypto.scaled_mm(
                    a_view,
                    b_view,
                    pypto.DT_FP16,
                    scale_a_view,
                    scale_b_view,
                    a_trans=False,
                    b_trans=True,
                    scale_a_trans=True,
                    scale_b_trans=False,
                )
                out_tensor[m_offset:m_offset + vm_view_size, n_offset:n_offset + vn_view_size] = out_view

    scale_k = k_size // 64
    device = "npu:0" if run_mode == "npu" else "cpu"
    a = torch.randn([m_size, k_size // 2], dtype=torch.float32).uniform_(-3, 3).to(torch.uint8).to(device)
    b = torch.randn([n_size, k_size // 2], dtype=torch.float32).uniform_(-3, 3).to(torch.uint8).to(device)
    scale_a = torch.randn([scale_k, m_size, 2], dtype=torch.float32).uniform_(0, 1).to(torch.float8_e8m0fnu).to(device)
    scale_b = torch.randn([scale_k, n_size, 2], dtype=torch.float32).uniform_(0, 1).to(torch.float8_e8m0fnu).to(device)
    out = torch.zeros([m_size, n_size], dtype=torch.float16).to(device)

    fp4_demo_kernel(a, b, out, scale_a, scale_b)


if __name__ == "__main__":
    run_fp4_demo("npu")
