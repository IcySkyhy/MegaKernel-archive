#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
K-axis split tests for GM accumulation validshape fix.

Tests verify correct behavior when k-axis is manually split into multiple blocks with tail blocks
where viewshape > validshape, using atomic_add to accumulate partial results.

优化说明:
- 包含 2D 和 3D BMM 场景
- 转置通过config参数控制
- 所有可配置参数从k_split_test_case.py读取
- 包含 matmul 和 scaled_mm 两种测试场景
"""

import os

import pytest
from testcase.k_view_split_test_case import (
    K_BLOCK_SIZE_64,
    K_SPLIT_2D_TESTS,
    K_SPLIT_3D_TESTS,
    SCALED_MM_K_SPLIT_TESTS,
    SHAPE_DIM_2,
    KSplitConfig,
    _process_scale_tensors,
)
import torch
import torch.nn.functional as functional
import torch_npu

import pypto


# ============================================================================
# 2D Matmul K-Split Kernel (handles transpose via config)
# ============================================================================
@pypto.frontend.jit(debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0})
def matmul_2d_k_split_kernel(
    a_tensor: pypto.Tensor([], pypto.DT_FP16),
    b_tensor: pypto.Tensor([], pypto.DT_FP16),
    out_tensor: pypto.Tensor([], pypto.DT_FP32),
    config: KSplitConfig,
):
    """
    2D Matmul kernel with manual k-axis split using view and atomic_add.
    Supports transpose via config.a_trans and config.b_trans.
    """
    m_size = config.m
    n_size = config.n
    k_size = config.k
    k_view = config.k_view
    a_trans = config.a_trans
    b_trans = config.b_trans

    pypto.set_cube_tile_shapes(*config.cube_tile_shape, config.is_acc)
    pypto.set_vec_tile_shapes(*config.vec_tile_shape)
    pypto.set_matrix_size([m_size, k_size, n_size])

    if a_trans:
        # A transposed: view as [k_view, m] with offset [0, 0]
        a_k_view = pypto.view(a_tensor, [k_view, m_size], [0, 0])
    else:
        # A normal: view as [m, k_view] with offset [0, 0]
        a_k_view = pypto.view(a_tensor, [m_size, k_view], [0, 0])

    if b_trans:
        # B transposed: view as [n, k_view] with offset [0, 0]
        b_k_view = pypto.view(b_tensor, [n_size, k_view], [0, 0])
    else:
        # B normal: view as [k_view, n] with offset [0, 0]
        b_k_view = pypto.view(b_tensor, [k_view, n_size], [0, 0])

    partial_out = pypto.matmul(
        a_k_view,
        b_k_view,
        out_dtype=pypto.DT_FP32,
        a_trans=a_trans,
        b_trans=b_trans,
    )
    out_tensor[:, :] = partial_out


# ============================================================================
# 3D BMM K-Split Kernel (handles transpose via config)
# ============================================================================
@pypto.frontend.jit(debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0})
def matmul_3d_bmm_k_split_kernel(
    a_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP16),
    b_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP16),
    out_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    config: KSplitConfig,
):
    """
    3D BMM kernel with manual k-axis split using view and atomic_add.
    Supports transpose via config.a_trans and config.b_trans.
    Batch dimension is handled via vec_tile_shape.
    """
    batch_size = config.batch_size
    m_size = config.m
    k_size = config.k
    n_size = config.n
    k_view = config.k_view
    a_trans = config.a_trans
    b_trans = config.b_trans

    pypto.set_cube_tile_shapes(*config.cube_tile_shape, config.is_acc)
    pypto.set_vec_tile_shapes(*config.vec_tile_shape)
    pypto.set_matrix_size([m_size, k_size, n_size])

    if a_trans:
        # A transposed: view as [batch, k_view, m] with offset [0, 0, 0]
        a_k_view = pypto.view(a_tensor, [batch_size, k_view, m_size], [0, 0, 0])
    else:
        # A normal: view as [batch, m, k_view] with offset [0, 0, 0]
        a_k_view = pypto.view(a_tensor, [batch_size, m_size, k_view], [0, 0, 0])

    if b_trans:
        # B transposed: view as [batch, n, k_view] with offset [0, 0, 0]
        b_k_view = pypto.view(b_tensor, [batch_size, n_size, k_view], [0, 0, 0])
    else:
        # B normal: view as [batch, k_view, n] with offset [0, 0, 0]
        b_k_view = pypto.view(b_tensor, [batch_size, k_view, n_size], [0, 0, 0])

    partial_out = pypto.matmul(
        a_k_view,
        b_k_view,
        out_dtype=pypto.DT_FP32,
        a_trans=a_trans,
        b_trans=b_trans,
    )
    out_tensor[:, :, :] = partial_out


# ============================================================================
# 2D ScaledMM K-Split Kernel
# ============================================================================
@pypto.frontend.jit(debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0})
def scaled_mm_2d_k_split_kernel(
    a_tensor: pypto.Tensor([], pypto.DT_FP8E4M3),
    b_tensor: pypto.Tensor([], pypto.DT_FP8E4M3),
    out_tensor: pypto.Tensor([], pypto.DT_FP32),
    scale_a_tensor: pypto.Tensor([], pypto.DT_FP8E8M0),
    scale_b_tensor: pypto.Tensor([], pypto.DT_FP8E8M0),
    config: KSplitConfig,
):
    """
    2D ScaledMM kernel with manual k-axis split using view and atomic_add.
    Supports transpose via config parameters.
    """
    m_size = config.m
    _k_size = config.k
    n_size = config.n
    k_view = config.k_view
    a_trans = config.a_trans
    b_trans = config.b_trans
    scale_a_trans = config.scale_a_trans
    scale_b_trans = config.scale_b_trans

    scale_k = (k_view + K_BLOCK_SIZE_64 - 1) // K_BLOCK_SIZE_64

    # Setup A view based on transpose - use fixed shape with valid_shape for tail handling
    if a_trans:
        a_view = pypto.view(a_tensor, [k_view, m_size], [0, 0])
    else:
        a_view = pypto.view(a_tensor, [m_size, k_view], [0, 0])

    # Setup B view based on transpose
    if b_trans:
        b_view = pypto.view(b_tensor, [n_size, k_view], [0, 0])
    else:
        b_view = pypto.view(b_tensor, [k_view, n_size], [0, 0])

    # Setup scale_a view based on transpose
    if scale_a_trans:
        scale_a_view = pypto.view(scale_a_tensor, [scale_k, m_size, 2], [0, 0, 0])
    else:
        scale_a_view = pypto.view(scale_a_tensor, [m_size, scale_k, 2], [0, 0, 0])

    # Setup scale_b view based on transpose
    if scale_b_trans:
        scale_b_view = pypto.view(scale_b_tensor, [n_size, scale_k, 2], [0, 0, 0])
    else:
        scale_b_view = pypto.view(scale_b_tensor, [scale_k, n_size, 2], [0, 0, 0])

    # Set cube tile shapes
    tile_shape = (config.m_tile_shape, config.k_tile_shape, config.n_tile_shape)
    pypto.set_cube_tile_shapes(*tile_shape, config.is_acc)

    # Perform scaled_mm
    partial_out = pypto.scaled_mm(
        a_view,
        b_view,
        pypto.DT_FP32,
        scale_a_view,
        scale_b_view,
        a_trans=a_trans,
        b_trans=b_trans,
        scale_a_trans=scale_a_trans,
        scale_b_trans=scale_b_trans,
        c_matrix_nz=config.c_format == "NZ",
    )

    out_tensor[:, :] = partial_out


# ============================================================================
# Test Runner Functions
# ============================================================================
def run_2d_k_split_test(case: dict):
    """Run 2D matmul k-split test."""
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)

    config = KSplitConfig.from_test_case(case)

    m, k, n = config.m, config.k, config.n
    a_shape = [k, m] if config.a_trans else [m, k]
    b_shape = [n, k] if config.b_trans else [k, n]
    c_shape = [m, n]

    a_torch_dtype = KSplitConfig.get_torch_dtype(config.a_dtype)
    b_torch_dtype = KSplitConfig.get_torch_dtype(config.b_dtype)
    c_torch_dtype = KSplitConfig.get_torch_dtype(config.c_dtype)

    torch.manual_seed(42)
    a_cpu = torch.randn(a_shape, dtype=a_torch_dtype)
    b_cpu = torch.randn(b_shape, dtype=b_torch_dtype)

    a_for_compute = a_cpu.T if config.a_trans else a_cpu
    b_for_compute = b_cpu.T if config.b_trans else b_cpu
    golden = torch.matmul(a_for_compute.to(torch.float32), b_for_compute.to(torch.float32))

    a_npu = a_cpu.to(f"npu:{device_id}")
    b_npu = b_cpu.to(f"npu:{device_id}")
    c_npu = torch.zeros(c_shape, dtype=c_torch_dtype, device=f"npu:{device_id}")

    matmul_2d_k_split_kernel(a_npu, b_npu, c_npu, config)

    atol, rtol = KSplitConfig.get_tolerance(config.c_dtype)
    match = torch.allclose(c_npu.cpu(), golden, atol=atol, rtol=rtol)
    assert match, f"2D k-split test failed: {case['name']}"


def run_3d_bmm_k_split_test(case: dict):
    """Run 3D BMM k-split test."""
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)

    config = KSplitConfig.from_test_case(case)

    batch_size = config.batch_size
    m, k, n = config.m, config.k, config.n
    a_shape = [batch_size, k, m] if config.a_trans else [batch_size, m, k]
    b_shape = [batch_size, n, k] if config.b_trans else [batch_size, k, n]
    c_shape = [batch_size, m, n]

    a_torch_dtype = KSplitConfig.get_torch_dtype(config.a_dtype)
    b_torch_dtype = KSplitConfig.get_torch_dtype(config.b_dtype)
    c_torch_dtype = KSplitConfig.get_torch_dtype(config.c_dtype)

    torch.manual_seed(42)
    a_cpu = torch.randn(a_shape, dtype=a_torch_dtype)
    b_cpu = torch.randn(b_shape, dtype=b_torch_dtype)

    a_for_compute = a_cpu.transpose(1, 2) if config.a_trans else a_cpu
    b_for_compute = b_cpu.transpose(1, 2) if config.b_trans else b_cpu
    golden = torch.matmul(a_for_compute.to(torch.float32), b_for_compute.to(torch.float32))

    a_npu = a_cpu.to(f"npu:{device_id}")
    b_npu = b_cpu.to(f"npu:{device_id}")
    c_npu = torch.zeros(c_shape, dtype=c_torch_dtype, device=f"npu:{device_id}")

    matmul_3d_bmm_k_split_kernel(a_npu, b_npu, c_npu, config)

    atol, rtol = KSplitConfig.get_tolerance(config.c_dtype)
    match = torch.allclose(c_npu.cpu(), golden, atol=atol, rtol=rtol)
    assert match, f"3D BMM k-split test failed: {case['name']}"


def run_scaled_mm_k_split_test(case: dict):
    """Run 2D scaled_mm k-split test."""
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)

    config = KSplitConfig.from_test_case(case)

    m, k, n = config.m, config.k, config.n
    a_shape = [k, m] if config.a_trans else [m, k]
    b_shape = [n, k] if config.b_trans else [k, n]

    # Calculate scale tensor shapes
    scale_k = (k + K_BLOCK_SIZE_64 - 1) // K_BLOCK_SIZE_64
    scale_a_shape = [scale_k, m, SHAPE_DIM_2] if config.scale_a_trans else [m, scale_k, SHAPE_DIM_2]
    scale_b_shape = [n, scale_k, SHAPE_DIM_2] if config.scale_b_trans else [scale_k, n, SHAPE_DIM_2]

    # Get torch dtypes
    a_torch_dtype = KSplitConfig.get_torch_dtype(config.a_dtype)
    b_torch_dtype = KSplitConfig.get_torch_dtype(config.b_dtype)
    c_torch_dtype = KSplitConfig.get_torch_dtype(config.c_dtype)

    # Generate input data
    torch.manual_seed(42)
    a_cpu = torch.rand(a_shape, dtype=torch.float32).uniform_(-3, 3).to(a_torch_dtype)
    b_cpu = torch.rand(b_shape, dtype=torch.float32).uniform_(-3, 3).to(b_torch_dtype)
    scale_a_cpu = torch.rand(scale_a_shape, dtype=torch.float32).uniform_(0, 1).to(torch.float8_e8m0fnu)
    scale_b_cpu = torch.rand(scale_b_shape, dtype=torch.float32).uniform_(0, 1).to(torch.float8_e8m0fnu)

    # Compute golden
    padding_k = scale_k * K_BLOCK_SIZE_64 - k
    scale_a_tmp, scale_b_tmp = _process_scale_tensors(scale_a_cpu, scale_b_cpu, config)

    mat_b_tmp = b_cpu.to(torch.float32).T if config.b_trans else b_cpu.to(torch.float32)
    mat_b_tmp = functional.pad(mat_b_tmp, ((0, 0, 0, padding_k)), "constant")
    mat_b_tmp = scale_b_tmp * mat_b_tmp

    mat_a_tmp = a_cpu.to(torch.float32).T if config.a_trans else a_cpu.to(torch.float32)
    mat_a_tmp = functional.pad(mat_a_tmp, ((0, padding_k, 0, 0)), "constant")
    mat_a_tmp = scale_a_tmp * mat_a_tmp

    golden = torch.matmul(mat_a_tmp, mat_b_tmp)
    golden = golden.to(c_torch_dtype)

    # Transfer to NPU
    a_npu = a_cpu.to(f"npu:{device_id}")
    b_npu = b_cpu.to(f"npu:{device_id}")
    scale_a_npu = scale_a_cpu.to(f"npu:{device_id}")
    scale_b_npu = scale_b_cpu.to(f"npu:{device_id}")
    c_npu = torch.zeros([m, n], dtype=c_torch_dtype, device=f"npu:{device_id}")

    # Apply format if needed
    if config.a_format == "NZ":
        a_npu = torch_npu.npu_format_cast(a_npu, 29)
    if config.b_format == "NZ":
        b_npu = torch_npu.npu_format_cast(b_npu, 29)

    # Run kernel
    scaled_mm_2d_k_split_kernel(a_npu, b_npu, c_npu, scale_a_npu, scale_b_npu, config)

    # Check accuracy
    atol, rtol = KSplitConfig.get_tolerance(config.c_dtype)
    match = torch.allclose(c_npu.cpu(), golden, atol=atol, rtol=rtol)
    assert match, f"ScaledMM k-split test failed: {case['name']}"


# ============================================================================
# Pytest Test Functions
# ============================================================================
@pytest.mark.parametrize(
    "case", [pytest.param(case, marks=pytest.mark.soc(*case["products"])) for case in K_SPLIT_2D_TESTS]
)
@pypto.options(pass_options={"enable_slice": True})
def test_2d_k_split(case: dict):
    """Test 2D matmul with k-axis split."""
    run_2d_k_split_test(case)


@pytest.mark.parametrize(
    "case", [pytest.param(case, marks=pytest.mark.soc(*case["products"])) for case in K_SPLIT_3D_TESTS]
)
@pypto.options(pass_options={"enable_slice": False})
def test_3d_bmm_k_split(case: dict):
    """Test 3D BMM with k-axis split."""
    run_3d_bmm_k_split_test(case)


@pytest.mark.parametrize(
    "case", [pytest.param(case, marks=pytest.mark.soc(*case["products"])) for case in SCALED_MM_K_SPLIT_TESTS]
)
@pypto.options(pass_options={"enable_slice": False})
def test_scaled_mm_k_split(case: dict):
    """Test 2D scaled_mm with k-axis split."""
    run_scaled_mm_k_split_test(case)


# ============================================================================
# Main Entry Point
# ============================================================================
if __name__ == "__main__":
    for case in K_SPLIT_2D_TESTS:
        run_2d_k_split_test(case)
