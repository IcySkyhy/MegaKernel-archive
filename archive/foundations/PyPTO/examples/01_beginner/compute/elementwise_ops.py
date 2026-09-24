#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Element-wise Operation Examples for PyPTO

This file contains all element-wise operation examples merged into a single file.
You can run all examples or select specific ones using command-line arguments.

Usage:
    python elementwise.py              # Run all examples
    python elementwise.py --list       # List all available examples
    python elementwise.py abs::test_abs_basic    # Run a specific case
"""

import argparse
import os
import sys

from numpy.testing import assert_allclose
import torch

import pypto


def _peek_run_mode_from_argv(default: str = "npu") -> str:
    """Read run_mode early so module-level decorators can use it."""
    for idx, arg in enumerate(sys.argv):
        if arg == "--run_mode" and idx + 1 < len(sys.argv):
            value = sys.argv[idx + 1]
            if value in ("npu", "sim"):
                return value
        if arg.startswith("--run_mode="):
            value = arg.split("=", 1)[1]
            if value in ("npu", "sim"):
                return value
    return default


global_run_mode = pypto.RunMode.NPU
if _peek_run_mode_from_argv("npu") == "sim":
    global_run_mode = pypto.RunMode.SIM


def get_device_id():
    """
    Get and validate TILE_FWK_DEVICE_ID from environment variable.

    Returns:
        int: The device ID if valid, None otherwise.
    """
    if 'TILE_FWK_DEVICE_ID' not in os.environ:
        print("Please set the environment variable TILE_FWK_DEVICE_ID before running:")
        print("  export TILE_FWK_DEVICE_ID=0")
        return None

    try:
        device_id = int(os.environ['TILE_FWK_DEVICE_ID'])
        return device_id
    except ValueError:
        print(f"ERROR: TILE_FWK_DEVICE_ID must be an integer, got: {os.environ['TILE_FWK_DEVICE_ID']}")
        return None


# ============================================================================
# ABS Examples
# ============================================================================


@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode})
def abs_kernel(x: pypto.Tensor([], pypto.DT_FP32), out: pypto.Tensor([], pypto.DT_FP32)):
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.abs(x)


@pypto.options(pass_options={"enable_slice": True})
def test_abs_basic(device_id: int = None):
    """Test basic usage of abs function"""
    print("=" * 60)
    print("Test: Basic Usage of abs Function")
    print("=" * 60)

    device = f'npu:{device_id}' if global_run_mode == pypto.RunMode.NPU and device_id is not None else 'cpu'

    dtype = torch.float32
    x = torch.tensor([-1, -8, 2], dtype=dtype, device=device)
    expected = torch.tensor([1, 8, 2], dtype=dtype, device=device)

    out = torch.empty(x.shape, dtype=dtype, device=device)
    abs_kernel(x, out)
    if global_run_mode == pypto.RunMode.NPU:
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print(f"Output: {out}")
    print(f"Expected: {expected}")
    print("✓ Basic usage of abs function completed successfully")


# ============================================================================
# ADD Examples
# ============================================================================


@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode})
def add_kernel(
    a: pypto.Tensor([], pypto.DT_FP32), b: pypto.Tensor([], pypto.DT_FP32), out: pypto.Tensor([], pypto.DT_FP32)
):
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.add(a, b)


@pypto.options(pass_options={"enable_slice": True})
def test_add_basic(device_id: int = None):
    """Test basic usage of add function"""
    print("=" * 60)
    print("Test: Basic Usage of add Function")
    print("=" * 60)

    device = f'npu:{device_id}' if global_run_mode == pypto.RunMode.NPU and device_id is not None else 'cpu'

    dtype = torch.float32
    a = torch.tensor([1, 2, 3], dtype=dtype, device=device)
    b = torch.tensor([4, 5, 6], dtype=dtype, device=device)
    expected = torch.tensor([5, 7, 9], dtype=dtype, device=device)

    out = torch.empty(a.shape, dtype=dtype, device=device)
    add_kernel(a, b, out)
    if global_run_mode == pypto.RunMode.NPU:
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print(f"Output: {out}")
    print(f"Expected: {expected}")
    print("✓ Basic usage of add function completed successfully")


@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode})
def add_broadcast_kernel(
    a: pypto.Tensor([], pypto.DT_FP32), b: pypto.Tensor([], pypto.DT_FP32), out: pypto.Tensor([], pypto.DT_FP32)
):
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.add(a, b)


@pypto.options(pass_options={"enable_slice": True})
def test_add_broadcast(device_id: int = None):
    """Test broadcasting between tensors of different shapes"""
    print("=" * 60)
    print("Test: Broadcasting Between Tensors")
    print("=" * 60)

    device = f'npu:{device_id}' if global_run_mode == pypto.RunMode.NPU and device_id is not None else 'cpu'

    dtype = torch.float32
    a = torch.tensor([[1, 2], [3, 4]], dtype=dtype, device=device)
    b = torch.tensor([1, 2], dtype=dtype, device=device)
    expected = torch.tensor([[2, 4], [4, 6]], dtype=dtype, device=device)

    out = torch.empty(a.shape, dtype=dtype, device=device)
    add_broadcast_kernel(a, b, out)
    if global_run_mode == pypto.RunMode.NPU:
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print(f"Output: {out}")
    print(f"Expected: {expected}")
    print("✓ Test Broadcasting Between Tensors completed successfully")


@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode})
def add_scalar_kernel(x: pypto.Tensor([], pypto.DT_FP32), out: pypto.Tensor([], pypto.DT_FP32), scalar: float):
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.add(x, scalar)


@pypto.options(pass_options={"enable_slice": True})
def test_add_scalar(device_id: int = None):
    """Test adding a scalar to a tensor"""
    print("=" * 60)
    print("Test: Adding a scalar to a tensor")
    print("=" * 60)

    device = f'npu:{device_id}' if global_run_mode == pypto.RunMode.NPU and device_id is not None else 'cpu'

    dtype = torch.float32
    a = torch.tensor([1, 2, 3], dtype=dtype, device=device)
    scalar = 2.0
    expected = torch.tensor([3, 4, 5], dtype=dtype, device=device)

    out = torch.empty(a.shape, dtype=dtype, device=device)
    add_scalar_kernel(a, out, scalar)
    if global_run_mode == pypto.RunMode.NPU:
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print(f"Output: {out}")
    print(f"Expected: {expected}")
    print("✓ Test Adding a scalar to a tensor completed successfully")


# ============================================================================
# CLIP Examples
# ============================================================================


@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode})
def clip_kernel(
    a: pypto.Tensor([], pypto.DT_FP32),
    min_: pypto.Tensor([], pypto.DT_FP32),
    max_: pypto.Tensor([], pypto.DT_FP32),
    out: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.clip(a, min_, max_)


@pypto.options(pass_options={"enable_slice": True})
def test_clip_basic(device_id: int = None):
    """Test basic usage of clip function"""
    print("=" * 60)
    print("Test: Basic Usage of clip Function")
    print("=" * 60)

    device = f'npu:{device_id}' if global_run_mode == pypto.RunMode.NPU and device_id is not None else 'cpu'

    dtype = torch.float32
    a = torch.tensor([[0, 2, 4], [3, 4, 6]], dtype=dtype, device=device)
    min_ = torch.tensor([[1, 1, 1], [1, 1, 1]], dtype=dtype, device=device)
    max_ = torch.tensor([[3, 3, 3], [3, 3, 3]], dtype=dtype, device=device)
    expected = torch.tensor([[1, 2, 3], [3, 3, 3]], dtype=dtype, device=device)

    out = torch.empty(a.shape, dtype=dtype, device=device)
    clip_kernel(a, min_, max_, out)
    if global_run_mode == pypto.RunMode.NPU:
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print(f"Output: {out}")
    print(f"Expected: {expected}")
    print("✓ Basic usage of clip function completed successfully")


@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode})
def clip_broadcast_kernel(
    a: pypto.Tensor([], pypto.DT_FP32),
    min_: pypto.Tensor([], pypto.DT_FP32),
    max_: pypto.Tensor([], pypto.DT_FP32),
    out: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.clip(a, min_, max_)


@pypto.options(pass_options={"enable_slice": True})
def test_clip_broadcast(device_id: int = None):
    """Test broadcasting between tensors of different shapes"""
    print("=" * 60)
    print("Test: Broadcasting Between Tensors")
    print("=" * 60)

    device = f'npu:{device_id}' if global_run_mode == pypto.RunMode.NPU and device_id is not None else 'cpu'

    dtype = torch.float32
    a = torch.tensor([[0, 2, 4], [3, 4, 6]], dtype=dtype, device=device)
    min_ = torch.tensor([1, 1, 1], dtype=dtype, device=device)
    max_ = torch.tensor([3, 3, 3], dtype=dtype, device=device)
    expected = torch.tensor([[1, 2, 3], [3, 3, 3]], dtype=dtype, device=device)

    out = torch.empty(a.shape, dtype=dtype, device=device)
    clip_broadcast_kernel(a, min_, max_, out)
    if global_run_mode == pypto.RunMode.NPU:
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print(f"Output: {out}")
    print(f"Expected: {expected}")
    print("✓ Test Broadcasting Between Tensors completed successfully")


# ============================================================================
# DIV Examples
# ============================================================================


@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode})
def div_kernel(
    a: pypto.Tensor([], pypto.DT_FP32), b: pypto.Tensor([], pypto.DT_FP32), out: pypto.Tensor([], pypto.DT_FP32)
):
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.div(a, b)


@pypto.options(pass_options={"enable_slice": True})
def test_div_basic(device_id: int = None):
    """Test basic usage of div function"""
    print("=" * 60)
    print("Test: Basic Usage of div Function")
    print("=" * 60)

    device = f'npu:{device_id}' if global_run_mode == pypto.RunMode.NPU and device_id is not None else 'cpu'

    dtype = torch.float32
    a = torch.tensor([6, 10, 15], dtype=dtype, device=device)
    b = torch.tensor([2, 5, 3], dtype=dtype, device=device)
    expected = torch.tensor([3, 2, 5], dtype=dtype, device=device)

    out = torch.empty(a.shape, dtype=dtype, device=device)
    div_kernel(a, b, out)
    if global_run_mode == pypto.RunMode.NPU:
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print(f"Output: {out}")
    print(f"Expected: {expected}")
    print("✓ Basic usage of div function completed successfully")


@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode})
def div_broadcast_kernel(
    a: pypto.Tensor([], pypto.DT_FP32), b: pypto.Tensor([], pypto.DT_FP32), out: pypto.Tensor([], pypto.DT_FP32)
):
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.div(a, b)


@pypto.options(pass_options={"enable_slice": True})
def test_div_broadcast(device_id: int = None):
    """Test broadcasting between tensors of different shapes"""
    print("=" * 60)
    print("Test: Broadcasting Between Tensors")
    print("=" * 60)

    device = f'npu:{device_id}' if global_run_mode == pypto.RunMode.NPU and device_id is not None else 'cpu'

    dtype = torch.float32
    a = torch.tensor([[1, 2], [3, 4]], dtype=dtype, device=device)
    b = torch.tensor([1, 2], dtype=dtype, device=device)
    expected = torch.tensor([[1, 1], [3, 2]], dtype=dtype, device=device)

    out = torch.empty(a.shape, dtype=dtype, device=device)
    div_broadcast_kernel(a, b, out)
    if global_run_mode == pypto.RunMode.NPU:
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print(f"Output: {out}")
    print(f"Expected: {expected}")
    print("✓ Test Broadcasting Between Tensors completed successfully")


@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode})
def div_scalar_kernel(a: pypto.Tensor([], pypto.DT_FP32), out: pypto.Tensor([], pypto.DT_FP32), scalar: float):
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.div(a, scalar)


@pypto.options(pass_options={"enable_slice": True})
def test_div_scalar(device_id: int = None):
    """Test diving a scalar to a tensor"""
    print("=" * 60)
    print("Test: Diving a scalar to a tensor")
    print("=" * 60)

    device = f'npu:{device_id}' if global_run_mode == pypto.RunMode.NPU and device_id is not None else 'cpu'

    dtype = torch.float32
    a = torch.tensor([1, 2, 3], dtype=dtype, device=device)
    scalar = 2.0
    expected = torch.tensor([0.5, 1, 1.5], dtype=dtype, device=device)

    out = torch.empty(a.shape, dtype=dtype, device=device)
    div_scalar_kernel(a, out, scalar)
    if global_run_mode == pypto.RunMode.NPU:
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print(f"Output: {out}")
    print(f"Expected: {expected}")
    print("✓ Test Diving a scalar to a tensor completed successfully")


# ============================================================================
# EXP Examples
# ============================================================================


@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode})
def exp_kernel(x: pypto.Tensor([], pypto.DT_FP32), out: pypto.Tensor([], pypto.DT_FP32)):
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.exp(x)


@pypto.options(pass_options={"enable_slice": True})
def test_exp_basic(device_id: int = None):
    """Test basic usage of exp function"""
    print("=" * 60)
    print("Test: Basic Usage of exp Function")
    print("=" * 60)

    device = f'npu:{device_id}' if global_run_mode == pypto.RunMode.NPU and device_id is not None else 'cpu'

    dtype = torch.float32
    x = torch.tensor([0, 1, 2], dtype=dtype, device=device)
    expected = torch.tensor([1.0000, 2.7183, 7.3891], dtype=dtype, device=device)

    out = torch.empty(x.shape, dtype=dtype, device=device)
    exp_kernel(x, out)
    if global_run_mode == pypto.RunMode.NPU:
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print(f"Output: {out}")
    print(f"Expected: {expected}")
    print("✓ Basic usage of exp function completed successfully")


# ============================================================================
# EXP2 Examples
# ============================================================================


@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode})
def exp2_kernel(x: pypto.Tensor([], pypto.DT_FP32), out: pypto.Tensor([], pypto.DT_FP32)):
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.exp2(x)


@pypto.options(pass_options={"enable_slice": True})
def test_exp2_basic(device_id: int = None):
    """Test basic usage of exp2 function"""
    print("=" * 60)
    print("Test: Basic Usage of exp2 Function")
    print("=" * 60)

    device = f'npu:{device_id}' if global_run_mode == pypto.RunMode.NPU and device_id is not None else 'cpu'

    dtype = torch.float32
    x = torch.tensor([0, 1, 2], dtype=dtype, device=device)
    expected = torch.tensor([1.0000, 2.0000, 4.0000], dtype=dtype, device=device)

    out = torch.empty(x.shape, dtype=dtype, device=device)
    exp2_kernel(x, out)
    if global_run_mode == pypto.RunMode.NPU:
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print(f"Output: {out}")
    print(f"Expected: {expected}")
    print("✓ Basic usage of exp2 function completed successfully")


# ============================================================================
# EXPM1 Examples
# ============================================================================


@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode})
def expm1_kernel(x: pypto.Tensor([], pypto.DT_FP32), out: pypto.Tensor([], pypto.DT_FP32)):
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.expm1(x)


@pypto.options(pass_options={"enable_slice": True})
def test_expm1_basic(device_id: int = None):
    """Test basic usage of expm1 function"""
    print("=" * 60)
    print("Test: Basic Usage of expm1 Function")
    print("=" * 60)

    device = f'npu:{device_id}' if global_run_mode == pypto.RunMode.NPU and device_id is not None else 'cpu'

    dtype = torch.float32
    x = torch.tensor([0, 1, 2], dtype=dtype, device=device)
    expected = torch.tensor([0.0000, 1.7183, 6.3891], dtype=dtype, device=device)

    out = torch.empty(x.shape, dtype=dtype, device=device)
    expm1_kernel(x, out)
    if global_run_mode == pypto.RunMode.NPU:
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print(f"Output: {out}")
    print(f"Expected: {expected}")
    print("✓ Basic usage of expm1 function completed successfully")


# ============================================================================
# LOG Examples
# ============================================================================


@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode})
def log_kernel(a: pypto.Tensor([], pypto.DT_FP32), out: pypto.Tensor([], pypto.DT_FP32)):
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.log(a)


@pypto.options(pass_options={"enable_slice": True})
def test_log_basic(device_id: int = None):
    """Test basic usage of log function"""
    print("=" * 60)
    print("Test: Basic Usage of log Function")
    print("=" * 60)

    device = f'npu:{device_id}' if global_run_mode == pypto.RunMode.NPU and device_id is not None else 'cpu'

    dtype = torch.float32
    a = torch.tensor([1, 2, 3], dtype=dtype, device=device)
    expected = torch.tensor([0, 0.6931, 1.0986], dtype=dtype, device=device)

    out = torch.empty(a.shape, dtype=dtype, device=device)
    log_kernel(a, out)
    if global_run_mode == pypto.RunMode.NPU:
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print(f"Output: {out}")
    print(f"Expected: {expected}")
    print("✓ Basic usage of log function completed successfully")


# ============================================================================
# MUL Examples
# ============================================================================


@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode})
def mul_kernel(
    a: pypto.Tensor([], pypto.DT_FP32), b: pypto.Tensor([], pypto.DT_FP32), out: pypto.Tensor([], pypto.DT_FP32)
):
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.mul(a, b)


@pypto.options(pass_options={"enable_slice": True})
def test_mul_basic(device_id: int = None):
    """Test basic usage of mul function"""
    print("=" * 60)
    print("Test: Basic Usage of mul Function")
    print("=" * 60)

    device = f'npu:{device_id}' if global_run_mode == pypto.RunMode.NPU and device_id is not None else 'cpu'

    dtype = torch.float32
    a = torch.tensor([1, 2, 3], dtype=dtype, device=device)
    b = torch.tensor([4, 5, 6], dtype=dtype, device=device)
    expected = torch.tensor([4, 10, 18], dtype=dtype, device=device)

    out = torch.empty(a.shape, dtype=dtype, device=device)
    mul_kernel(a, b, out)
    if global_run_mode == pypto.RunMode.NPU:
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print(f"Output: {out}")
    print(f"Expected: {expected}")
    print("✓ Basic usage of mul function completed successfully")


@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode})
def mul_broadcast_kernel(
    a: pypto.Tensor([], pypto.DT_FP32), b: pypto.Tensor([], pypto.DT_FP32), out: pypto.Tensor([], pypto.DT_FP32)
):
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.mul(a, b)


@pypto.options(pass_options={"enable_slice": True})
def test_mul_broadcast(device_id: int = None):
    """Test broadcasting between tensors of different shapes"""
    print("=" * 60)
    print("Test: Broadcasting Between Tensors")
    print("=" * 60)

    device = f'npu:{device_id}' if global_run_mode == pypto.RunMode.NPU and device_id is not None else 'cpu'

    dtype = torch.float32
    a = torch.tensor([[1, 2], [3, 4]], dtype=dtype, device=device)
    b = torch.tensor([1, 2], dtype=dtype, device=device)
    expected = torch.tensor([[1, 4], [3, 8]], dtype=dtype, device=device)

    out = torch.empty(a.shape, dtype=dtype, device=device)
    mul_broadcast_kernel(a, b, out)
    if global_run_mode == pypto.RunMode.NPU:
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print(f"Output: {out}")
    print(f"Expected: {expected}")
    print("✓ Test Broadcasting Between Tensors completed successfully")


@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode})
def mul_scalar_kernel(a: pypto.Tensor([], pypto.DT_FP32), out: pypto.Tensor([], pypto.DT_FP32), scalar: float):
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.mul(a, scalar)


@pypto.options(pass_options={"enable_slice": True})
def test_mul_scalar(device_id: int = None):
    """Test muling a scalar to a tensor"""
    print("=" * 60)
    print("Test: Muling a scalar to a tensor")
    print("=" * 60)

    device = f'npu:{device_id}' if global_run_mode == pypto.RunMode.NPU and device_id is not None else 'cpu'

    dtype = torch.float32
    a = torch.tensor([1, 2, 3], dtype=dtype, device=device)
    scalar = 2.0
    expected = torch.tensor([2, 4, 6], dtype=dtype, device=device)

    out = torch.empty(a.shape, dtype=dtype, device=device)
    mul_scalar_kernel(a, out, scalar)
    if global_run_mode == pypto.RunMode.NPU:
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print(f"Output: {out}")
    print(f"Expected: {expected}")
    print("✓ Test Muling a scalar to a tensor completed successfully")


# ============================================================================
# NEG Examples
# ============================================================================


@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode})
def neg_kernel(a: pypto.Tensor([], pypto.DT_FP32), out: pypto.Tensor([], pypto.DT_FP32)):
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.neg(a)


@pypto.options(pass_options={"enable_slice": True})
def test_neg_basic(device_id: int = None):
    """Test basic usage of neg function"""
    print("=" * 60)
    print("Test: Basic Usage of neg Function")
    print("=" * 60)

    device = f'npu:{device_id}' if global_run_mode == pypto.RunMode.NPU and device_id is not None else 'cpu'

    dtype = torch.float32
    a = torch.tensor([[1, 4], [16, 9]], dtype=dtype, device=device)
    expected = torch.tensor([[-1, -4], [-16, -9]], dtype=dtype, device=device)

    out = torch.empty(a.shape, dtype=dtype, device=device)
    neg_kernel(a, out)
    if global_run_mode == pypto.RunMode.NPU:
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print(f"Output: {out}")
    print(f"Expected: {expected}")
    print("✓ Basic usage of neg function completed successfully")


# ============================================================================
# POW Examples
# ============================================================================


@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode})
def pow_kernel(a: pypto.Tensor([], pypto.DT_FP32), out: pypto.Tensor([], pypto.DT_FP32), b: float):
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.pow(a, b)


@pypto.options(pass_options={"enable_slice": True})
def test_pow_basic(device_id: int = None):
    """Test basic usage of pow function"""
    print("=" * 60)
    print("Test: Basic Usage of pow Function")
    print("=" * 60)

    device = f'npu:{device_id}' if global_run_mode == pypto.RunMode.NPU and device_id is not None else 'cpu'

    dtype = torch.float32
    a = torch.tensor([3, 3], dtype=dtype, device=device)
    b = 2.0
    expected = torch.tensor([9, 9], dtype=dtype, device=device)

    out = torch.empty(a.shape, dtype=dtype, device=device)
    pow_kernel(a, out, b)
    if global_run_mode == pypto.RunMode.NPU:
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print(f"Output: {out}")
    print(f"Expected: {expected}")
    print("✓ Basic usage of pow function completed successfully")


# ============================================================================
# ROUND Examples
# ============================================================================


@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode})
def round_kernel(a: pypto.Tensor([], pypto.DT_FP32), out: pypto.Tensor([], pypto.DT_FP32), decimals: int):
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.round(a, decimals=decimals)


@pypto.options(pass_options={"enable_slice": True})
def test_round_basic(device_id: int = None):
    """Test basic usage of round function"""
    print("=" * 60)
    print("Test: Basic Usage of round Function")
    print("=" * 60)

    device = f'npu:{device_id}' if global_run_mode == pypto.RunMode.NPU and device_id is not None else 'cpu'

    dtype = torch.float32
    a = torch.tensor([[1.21, 2.35], [3.65, 4.76]], dtype=dtype, device=device)
    decimals = 1
    expected = torch.tensor([[1.2, 2.4], [3.6, 4.8]], dtype=dtype, device=device)

    out = torch.empty(a.shape, dtype=dtype, device=device)
    round_kernel(a, out, decimals)
    if global_run_mode == pypto.RunMode.NPU:
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print(f"Output: {out}")
    print(f"Expected: {expected}")
    print("✓ Basic usage of round function completed successfully")


# ============================================================================
# RSQRT Examples
# ============================================================================


@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode})
def rsqrt_kernel(a: pypto.Tensor([], pypto.DT_FP32), out: pypto.Tensor([], pypto.DT_FP32)):
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.rsqrt(a)


@pypto.options(pass_options={"enable_slice": True})
def test_rsqrt_basic(device_id: int = None):
    """Test basic usage of rsqrt function"""
    print("=" * 60)
    print("Test: Basic Usage of rsqrt Function")
    print("=" * 60)

    device = f'npu:{device_id}' if global_run_mode == pypto.RunMode.NPU and device_id is not None else 'cpu'

    dtype = torch.float32
    a = torch.tensor([[1, 4], [16, 9]], dtype=dtype, device=device)
    expected = torch.tensor([[1, 0.5], [0.25, 0.333333]], dtype=dtype, device=device)

    out = torch.empty(a.shape, dtype=dtype, device=device)
    rsqrt_kernel(a, out)
    if global_run_mode == pypto.RunMode.NPU:
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print(f"Output: {out}")
    print(f"Expected: {expected}")
    print("✓ Basic usage of rsqrt function completed successfully")


# ============================================================================
# CEIL Examples
# ============================================================================


@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode})
def ceil_kernel(a: pypto.Tensor([], pypto.DT_FP32), out: pypto.Tensor([], pypto.DT_FP32)):
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.ceil(a)


@pypto.options(pass_options={"enable_slice": True})
def test_ceil_basic(device_id: int = None):
    """Test basic usage of ceil function"""
    print("=" * 60)
    print("Test: Basic Usage of ceil Function")
    print("=" * 60)

    device = f'npu:{device_id}' if global_run_mode == pypto.RunMode.NPU and device_id is not None else 'cpu'

    dtype = torch.float32
    a = torch.tensor([[1.2, 4.7], [-1.1, 9.0]], dtype=dtype, device=device)
    expected = torch.tensor([[2.0, 5.0], [-1.0, 9.0]], dtype=dtype, device=device)

    out = torch.empty(a.shape, dtype=dtype, device=device)
    ceil_kernel(a, out)
    if global_run_mode == pypto.RunMode.NPU:
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print(f"Output: {out}")
    print(f"Expected: {expected}")
    print("✓ Basic usage of ceil function completed successfully")


# ============================================================================
# FLOOR Examples
# ============================================================================


@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode})
def floor_kernel(a: pypto.Tensor([], pypto.DT_FP32), out: pypto.Tensor([], pypto.DT_FP32)):
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.floor(a)


@pypto.options(pass_options={"enable_slice": True})
def test_floor_basic(device_id: int = None):
    """Test basic usage of floor function"""
    print("=" * 60)
    print("Test: Basic Usage of floor Function")
    print("=" * 60)

    device = f'npu:{device_id}' if global_run_mode == pypto.RunMode.NPU and device_id is not None else 'cpu'

    dtype = torch.float32
    a = torch.tensor([[1.2, 4.7], [-1.1, 9.0]], dtype=dtype, device=device)
    expected = torch.tensor([[1.0, 4.0], [-2.0, 9.0]], dtype=dtype, device=device)

    out = torch.empty(a.shape, dtype=dtype, device=device)
    floor_kernel(a, out)
    if global_run_mode == pypto.RunMode.NPU:
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print(f"Output: {out}")
    print(f"Expected: {expected}")
    print("✓ Basic usage of floor function completed successfully")


# ============================================================================
# TRUNC Examples
# ============================================================================


@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode})
def trunc_kernel(a: pypto.Tensor([], pypto.DT_FP32), out: pypto.Tensor([], pypto.DT_FP32)):
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.trunc(a)


@pypto.options(pass_options={"enable_slice": True})
def test_trunc_basic(device_id: int = None):
    """Test basic usage of trunc function"""
    print("=" * 60)
    print("Test: Basic Usage of trunc Function")
    print("=" * 60)

    device = f'npu:{device_id}' if global_run_mode == pypto.RunMode.NPU and device_id is not None else 'cpu'

    dtype = torch.float32
    a = torch.tensor([[1.2, 4.1], [16.8, 9.3]], dtype=dtype, device=device)
    expected = torch.tensor([[1.0, 4.0], [16.0, 9.0]], dtype=dtype, device=device)

    out = torch.empty(a.shape, dtype=dtype, device=device)
    trunc_kernel(a, out)
    if global_run_mode == pypto.RunMode.NPU:
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print(f"Output: {out}")
    print(f"Expected: {expected}")
    print("✓ Basic usage of trunc function completed successfully")


# ============================================================================
# SQRT Examples
# ============================================================================


@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode})
def sqrt_kernel(a: pypto.Tensor([], pypto.DT_FP32), out: pypto.Tensor([], pypto.DT_FP32)):
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.sqrt(a)


@pypto.options(pass_options={"enable_slice": True})
def test_sqrt_basic(device_id: int = None):
    """Test basic usage of sqrt function"""
    print("=" * 60)
    print("Test: Basic Usage of sqrt Function")
    print("=" * 60)

    device = f'npu:{device_id}' if global_run_mode == pypto.RunMode.NPU and device_id is not None else 'cpu'

    dtype = torch.float32
    a = torch.tensor([[1, 4], [16, 9]], dtype=dtype, device=device)
    expected = torch.tensor([[1, 2], [4, 3]], dtype=dtype, device=device)

    out = torch.empty(a.shape, dtype=dtype, device=device)
    sqrt_kernel(a, out)
    if global_run_mode == pypto.RunMode.NPU:
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print(f"Output: {out}")
    print(f"Expected: {expected}")
    print("✓ Basic usage of sqrt function completed successfully")


# ============================================================================
# SUB Examples
# ============================================================================


@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode})
def sub_kernel(
    a: pypto.Tensor([], pypto.DT_FP32), b: pypto.Tensor([], pypto.DT_FP32), out: pypto.Tensor([], pypto.DT_FP32)
):
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.sub(a, b)


@pypto.options(pass_options={"enable_slice": True})
def test_sub_basic(device_id: int = None):
    """Test basic usage of sub function"""
    print("=" * 60)
    print("Test: Basic Usage of sub Function")
    print("=" * 60)

    device = f'npu:{device_id}' if global_run_mode == pypto.RunMode.NPU and device_id is not None else 'cpu'

    dtype = torch.float32
    a = torch.tensor([4, 5, 6], dtype=dtype, device=device)
    b = torch.tensor([1, 2, 3], dtype=dtype, device=device)
    expected = torch.tensor([3, 3, 3], dtype=dtype, device=device)

    out = torch.empty(a.shape, dtype=dtype, device=device)
    sub_kernel(a, b, out)
    if global_run_mode == pypto.RunMode.NPU:
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print(f"Output: {out}")
    print(f"Expected: {expected}")
    print("✓ Basic usage of sub function completed successfully")


@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode})
def sub_broadcast_kernel(
    a: pypto.Tensor([], pypto.DT_FP32), b: pypto.Tensor([], pypto.DT_FP32), out: pypto.Tensor([], pypto.DT_FP32)
):
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.sub(a, b)


@pypto.options(pass_options={"enable_slice": True})
def test_sub_broadcast(device_id: int = None):
    """Test broadcasting between tensors of different shapes"""
    print("=" * 60)
    print("Test: Broadcasting Between Tensors")
    print("=" * 60)

    device = f'npu:{device_id}' if global_run_mode == pypto.RunMode.NPU and device_id is not None else 'cpu'

    dtype = torch.float32
    a = torch.tensor([[1, 2], [3, 4]], dtype=dtype, device=device)
    b = torch.tensor([1, 2], dtype=dtype, device=device)
    expected = torch.tensor([[0, 0], [2, 2]], dtype=dtype, device=device)

    out = torch.empty(a.shape, dtype=dtype, device=device)
    sub_broadcast_kernel(a, b, out)
    if global_run_mode == pypto.RunMode.NPU:
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print(f"Output: {out}")
    print(f"Expected: {expected}")
    print("✓ Test Broadcasting Between Tensors completed successfully")


@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode})
def sub_scalar_kernel(a: pypto.Tensor([], pypto.DT_FP32), out: pypto.Tensor([], pypto.DT_FP32), scalar: float):
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.sub(a, scalar)


@pypto.options(pass_options={"enable_slice": True})
def test_sub_scalar(device_id: int = None):
    """Test subing a scalar to a tensor"""
    print("=" * 60)
    print("Test: Subing a scalar to a tensor")
    print("=" * 60)

    device = f'npu:{device_id}' if global_run_mode == pypto.RunMode.NPU and device_id is not None else 'cpu'

    dtype = torch.float32
    a = torch.tensor([1, 2, 3], dtype=dtype, device=device)
    scalar = 2.0
    expected = torch.tensor([-1, 0, 1], dtype=dtype, device=device)

    out = torch.empty(a.shape, dtype=dtype, device=device)
    sub_scalar_kernel(a, out, scalar)
    if global_run_mode == pypto.RunMode.NPU:
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print(f"Output: {out}")
    print(f"Expected: {expected}")
    print("✓ Test Subing a scalar to a tensor completed successfully")


# ============================================================================
# Main Function
# ============================================================================


def main():
    """Run element-wise examples.

    Usage:
        python elementwise.py              # Run all examples
        python elementwise.py --list       # List all available examples
        python elementwise.py abs::test_abs_basic    # Run a specific case
    """
    parser = argparse.ArgumentParser(
        description="PyPTO Element-wise Operation Examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                      Run all examples
  %(prog)s --list               List all available examples
  %(prog)s abs::test_abs_basic    Run a specific case
        """,
    )
    parser.add_argument(
        'example_id',
        type=str,
        nargs="?",
        help='Run a specific case (e.g., abs::test_abs_basic). If omitted, all cases run.',
    )
    parser.add_argument('--list', action='store_true', help='List all available examples and exit')
    parser.add_argument(
        "--run_mode",
        "--run-mode",
        nargs="?",
        type=str,
        default="npu",
        choices=["npu", "sim"],
        help='Run mode, supports npu and sim.',
    )

    args = parser.parse_args()

    # Define available examples
    examples = {
        'abs::test_abs_basic': {
            'name': 'Test basic usage of abs function',
            'description': 'Basic usage of abs function example',
            'function': test_abs_basic,
        },
        'add::test_add_basic': {
            'name': 'Test basic usage of add function',
            'description': 'Basic usage of add function example',
            'function': test_add_basic,
        },
        'add::test_add_broadcast': {
            'name': 'Test broadcasting between tensors of different shapes',
            'description': 'Broadcasting between tensors example',
            'function': test_add_broadcast,
        },
        'add::test_add_scalar': {
            'name': 'Test adding a scalar to a tensor',
            'description': 'Adding a scalar to a tensor example',
            'function': test_add_scalar,
        },
        'clip::test_clip_basic': {
            'name': 'Test basic usage of clip function',
            'description': 'Basic usage of clip function example',
            'function': test_clip_basic,
        },
        'clip::test_clip_broadcast': {
            'name': 'Test broadcasting between tensors of different shapes',
            'description': 'Broadcasting between tensors example',
            'function': test_clip_broadcast,
        },
        'div::test_div_basic': {
            'name': 'Test basic usage of div function',
            'description': 'Basic usage of div function example',
            'function': test_div_basic,
        },
        'div::test_div_broadcast': {
            'name': 'Test broadcasting between tensors of different shapes',
            'description': 'Broadcasting between tensors example',
            'function': test_div_broadcast,
        },
        'div::test_div_scalar': {
            'name': 'Test diving a scalar to a tensor',
            'description': 'Diving a scalar to a tensor example',
            'function': test_div_scalar,
        },
        'exp::test_exp_basic': {
            'name': 'Test basic usage of exp function',
            'description': 'Basic usage of exp function example',
            'function': test_exp_basic,
        },
        'log::test_log_basic': {
            'name': 'Test basic usage of log function',
            'description': 'Basic usage of log function example',
            'function': test_log_basic,
        },
        'mul::test_mul_basic': {
            'name': 'Test basic usage of mul function',
            'description': 'Basic usage of mul function example',
            'function': test_mul_basic,
        },
        'mul::test_mul_broadcast': {
            'name': 'Test broadcasting between tensors of different shapes',
            'description': 'Broadcasting between tensors example',
            'function': test_mul_broadcast,
        },
        'mul::test_mul_scalar': {
            'name': 'Test muling a scalar to a tensor',
            'description': 'Muling a scalar to a tensor example',
            'function': test_mul_scalar,
        },
        'neg::test_neg_basic': {
            'name': 'Test basic usage of neg function',
            'description': 'Basic usage of neg function example',
            'function': test_neg_basic,
        },
        'pow::test_pow_basic': {
            'name': 'Test basic usage of pow function',
            'description': 'Basic usage of pow function example',
            'function': test_pow_basic,
        },
        'round::test_round_basic': {
            'name': 'Test basic usage of round function',
            'description': 'Basic usage of round function example',
            'function': test_round_basic,
        },
        'rsqrt::test_rsqrt_basic': {
            'name': 'Test basic usage of rsqrt function',
            'description': 'Basic usage of rsqrt function example',
            'function': test_rsqrt_basic,
        },
        'ceil::test_ceil_basic': {
            'name': 'Test basic usage of ceil function',
            'description': 'Basic usage of ceil function example',
            'function': test_ceil_basic,
        },
        'floor::test_floor_basic': {
            'name': 'Test basic usage of floor function',
            'description': 'Basic usage of floor function example',
            'function': test_floor_basic,
        },
        'trunc::test_trunc_basic': {
            'name': 'Test basic usage of trunc function',
            'description': 'Basic usage of trunc function example',
            'function': test_trunc_basic,
        },
        'sqrt::test_sqrt_basic': {
            'name': 'Test basic usage of sqrt function',
            'description': 'Basic usage of sqrt function example',
            'function': test_sqrt_basic,
        },
        'sub::test_sub_basic': {
            'name': 'Test basic usage of sub function',
            'description': 'Basic usage of sub function example',
            'function': test_sub_basic,
        },
        'sub::test_sub_broadcast': {
            'name': 'Test broadcasting between tensors of different shapes',
            'description': 'Broadcasting between tensors example',
            'function': test_sub_broadcast,
        },
        'sub::test_sub_scalar': {
            'name': 'Test subing a scalar to a tensor',
            'description': 'Subing a scalar to a tensor example',
            'function': test_sub_scalar,
        },
    }

    # List examples if requested
    if args.list:
        print("\n" + "=" * 60)
        print("Available Examples")
        print("=" * 60 + "\n")
        for case_key, ex_info in sorted(examples.items()):
            print(f"  {case_key}")
            print(f"     {ex_info['name']}")
            print(f"     {ex_info['description']}\n")
        return

    # Validate case if provided
    examples_to_run = []
    if args.example_id:
        if args.example_id not in examples:
            print(f"ERROR: Invalid case: {args.example_id}")
            print(f"Valid cases are: {', '.join(sorted(examples.keys()))}")
            print("\nUse --list to see all available examples.")
            sys.exit(1)
        examples_to_run = [(args.example_id, examples[args.example_id])]
    else:
        examples_to_run = [(key, info) for key, info in sorted(examples.items())]

    print("\n" + "=" * 60)
    print("PyPTO Element-wise Operation Examples")
    print("=" * 60 + "\n")

    device_id = None
    if args.run_mode == "npu":
        device_id = get_device_id()
        if device_id is None:
            return
        torch.npu.set_device(device_id)

    try:
        for case_key, ex_info in examples_to_run:
            if args.run_mode == "npu" and device_id is None:
                print(f"Skipping {case_key} ({ex_info['name']}): NPU device not configured")
                continue

            ex_info['function'](device_id)

        if len(examples_to_run) > 1:
            print("=" * 60)
            print("All element-wise tests passed!")
            print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
