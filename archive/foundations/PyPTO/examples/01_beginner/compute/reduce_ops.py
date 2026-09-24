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
Reduce Operation Examples for PyPTO

This file contains all reduce operation examples merged into a single file.
You can run all examples or select specific ones using command-line arguments.

Usage:
    python reduce_ops.py              # Run all examples
    python reduce_ops.py --list       # List all available examples
    python reduce_ops.py sum::test_sum_basic    # Run a specific case
"""

import argparse
import os
import sys

import numpy as np
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
# SUM Examples
# ============================================================================


def sum_op(a: torch.Tensor, dim: int, keepdim: bool = False) -> torch.Tensor:
    dtype = pypto.DT_FP32
    _shape = a.shape

    if keepdim:
        out_shape = list(a.shape)
        out_shape[dim] = 1
        out_shape = tuple(out_shape)
    else:
        out_shape = list(a.shape)
        out_shape.pop(dim)
        out_shape = tuple(out_shape)

    @pypto.frontend.jit(runtime_options={"run_mode": global_run_mode})
    def sum_kernel(a: pypto.Tensor([], dtype), out: pypto.Tensor([], dtype)):
        tile_shapes = [8] * len(a.shape)
        pypto.set_vec_tile_shapes(*tile_shapes)
        out[:] = pypto.sum(a, dim=dim, keepdim=keepdim)

    out = torch.empty(out_shape, dtype=torch.float32, device=a.device)
    sum_kernel(a, out)
    return out


@pypto.options(pass_options={"enable_slice": True})
def test_sum_basic(device_id: int = None):
    """Test basic usage of sum function"""
    print("=" * 60)
    print("Test: Basic Usage of sum Function")
    print("=" * 60)

    device = f'npu:{device_id}' if global_run_mode == pypto.RunMode.NPU and device_id is not None else 'cpu'

    # Test 1: Basic reduction along the last dimension(keepdim=False)
    dtype = torch.float32
    a = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=dtype, device=device)
    expected = torch.tensor([6, 15], dtype=dtype, device=device)
    dim = -1
    out = sum_op(a, dim, keepdim=False)
    print(f"Output (keepdim=False): {out}")
    print(f"Expected (keepdim=False): {expected}")
    if global_run_mode == pypto.RunMode.NPU:
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)

    # Test 2: Basic reduction along the last dimension(keepdim=True)
    dtype = torch.float32
    a = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=dtype, device=device)
    expected = torch.tensor([[6], [15]], dtype=dtype, device=device)

    out = sum_op(a, dim, keepdim=True)
    print(f"Output (keepdim=True): {out}")
    print(f"Expected (keepdim=True): {expected}")
    if global_run_mode == pypto.RunMode.NPU:
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)

    print("✓ Basic usage of sum function completed successfully")


@pypto.options(pass_options={"enable_slice": True})
def test_sum_different_dimensions(device_id: int = None):
    """Test reducing along different dimensions"""
    print("=" * 60)
    print("Test: Reducing Along Different Dimensions")
    print("=" * 60)

    device = f'npu:{device_id}' if global_run_mode == pypto.RunMode.NPU and device_id is not None else 'cpu'

    # Test 1: Reduction along the dim=0
    dtype = torch.float32
    a = torch.tensor(
        [[[10, 20, 30, 40], [15, 25, 35, 45], [12, 22, 32, 42]], [[5, 28, 33, 41], [18, 21, 36, 44], [11, 29, 31, 43]]],
        dtype=dtype,
        device=device,
    )
    expected = torch.tensor([[15, 48, 63, 81], [33, 46, 71, 89], [23, 51, 63, 85]], dtype=dtype, device=device)
    dim = 0
    out = sum_op(a, dim, keepdim=False)
    max_diff = np.abs(out.cpu().numpy() - expected.cpu().numpy()).max()
    print(f"Output (dim=0): {out}")
    print(f"Expected (dim=0): {expected}")
    print(f"Max difference: {max_diff:.6f}")
    if global_run_mode == pypto.RunMode.NPU:
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)

    # Test 2: Reduction along the dim=1
    expected = torch.tensor([[37, 67, 97, 127], [34, 78, 100, 128]], dtype=dtype, device=device)
    dim = 1
    out = sum_op(a, dim, keepdim=False)
    print(f"Output (dim=1): {out}")
    print(f"Expected (dim=1): {expected}")
    if global_run_mode == pypto.RunMode.NPU:
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)

    # Test 3: Reduction along the dim=2
    expected = torch.tensor([[100, 120, 108], [107, 119, 114]], dtype=dtype, device=device)
    dim = -1
    out = sum_op(a, dim, keepdim=False)
    print(f"Output (dim=-1): {out}")
    print(f"Expected (dim=-1): {expected}")
    if global_run_mode == pypto.RunMode.NPU:
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)

    print("✓ Reducing along different dimensions completed successfully")


# ============================================================================
# AMAX Examples
# ============================================================================
def amax_op(a: torch.Tensor, dim: int, keepdim: bool = False) -> torch.Tensor:
    dtype = pypto.DT_FP32
    _shape = a.shape
    if keepdim:
        out_shape = list(a.shape)
        out_shape[dim] = 1
        out_shape = tuple(out_shape)
    else:
        out_shape = list(a.shape)
        out_shape.pop(dim)
        out_shape = tuple(out_shape)

    @pypto.frontend.jit(runtime_options={"run_mode": global_run_mode})
    def amax_kernel(a: pypto.Tensor([], dtype), out: pypto.Tensor([], dtype)):
        tile_shapes = [8] * len(a.shape)
        pypto.set_vec_tile_shapes(*tile_shapes)
        out[:] = pypto.amax(a, dim=dim, keepdim=keepdim)

    out = torch.empty(out_shape, dtype=torch.float32, device=a.device)
    amax_kernel(a, out)
    return out


@pypto.options(pass_options={"enable_slice": True})
def test_amax_basic(device_id: int = None):
    """Test basic usage of amax function"""
    print("=" * 60)
    print("Test: Basic Usage of amax Function")
    print("=" * 60)

    device = f'npu:{device_id}' if global_run_mode == pypto.RunMode.NPU and device_id is not None else 'cpu'

    # Test 1: Basic reduction along the last dimension(keepdim=False)
    dtype = torch.float32
    a = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=dtype, device=device)
    expected = torch.tensor([3, 6], dtype=dtype, device=device)
    dim = -1
    out = amax_op(a, dim, keepdim=False)
    max_diff = np.abs(out.cpu().numpy() - expected.cpu().numpy()).max()
    print(f"Output (keepdim=False): {out}")
    print(f"Expected (keepdim=False): {expected}")
    print(f"Max difference: {max_diff:.6f}")
    if global_run_mode == pypto.RunMode.NPU:
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)

    # Test 2: Basic reduction along the last dimension(keepdim=True)
    dtype = torch.float32
    a = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=dtype, device=device)
    expected = torch.tensor([[3], [6]], dtype=dtype, device=device)

    out = amax_op(a, dim, keepdim=True)
    max_diff = np.abs(out.cpu().numpy() - expected.cpu().numpy()).max()
    print(f"Output (keepdim=True): {out}")
    print(f"Expected (keepdim=True): {expected}")
    print(f"Max difference: {max_diff:.6f}")
    if global_run_mode == pypto.RunMode.NPU:
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)

    print("✓ Basic usage of amax function completed successfully")


@pypto.options(pass_options={"enable_slice": True})
def test_amax_different_dimensions(device_id: int = None):
    """Test reducing along different dimensions"""
    print("=" * 60)
    print("Test: Reducing Along Different Dimensions")
    print("=" * 60)

    device = f'npu:{device_id}' if global_run_mode == pypto.RunMode.NPU and device_id is not None else 'cpu'

    # Test 1: Reduction along the dim=0
    dtype = torch.float32
    a = torch.tensor(
        [[[10, 20, 30, 40], [15, 25, 35, 45], [12, 22, 32, 42]], [[5, 28, 33, 41], [18, 21, 36, 44], [11, 29, 31, 43]]],
        dtype=dtype,
        device=device,
    )
    expected = torch.tensor([[10, 28, 33, 41], [18, 25, 36, 45], [12, 29, 32, 43]], dtype=dtype, device=device)
    dim = 0
    out = amax_op(a, dim, keepdim=False)
    max_diff = np.abs(out.cpu().numpy() - expected.cpu().numpy()).max()
    print(f"Output (dim=0): {out}")
    print(f"Expected (dim=0): {expected}")
    print(f"Max difference: {max_diff:.6f}")
    if global_run_mode == pypto.RunMode.NPU:
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)

    # Test 2: Reduction along the dim=1
    expected = torch.tensor([[15, 25, 35, 45], [18, 29, 36, 44]], dtype=dtype, device=device)
    dim = 1
    out = amax_op(a, dim, keepdim=False)
    print(f"Output (dim=1): {out}")
    print(f"Expected (dim=1): {expected}")
    if global_run_mode == pypto.RunMode.NPU:
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)

    # Test 3: Reduction along the dim=2
    expected = torch.tensor([[40, 45, 42], [41, 44, 43]], dtype=dtype, device=device)
    dim = -1
    out = amax_op(a, dim, keepdim=False)
    print(f"Output (dim=-1): {out}")
    print(f"Expected (dim=-1): {expected}")
    if global_run_mode == pypto.RunMode.NPU:
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)

    print("✓ Reducing along different dimensions completed successfully")


# ============================================================================
# AMIN Examples
# ============================================================================
def amin_op(a: torch.Tensor, dim: int, keepdim: bool = False) -> torch.Tensor:
    dtype = pypto.DT_FP32
    _shape = a.shape
    if keepdim:
        out_shape = list(a.shape)
        out_shape[dim] = 1
        out_shape = tuple(out_shape)
    else:
        out_shape = list(a.shape)
        out_shape.pop(dim)
        out_shape = tuple(out_shape)

    @pypto.frontend.jit(runtime_options={"run_mode": global_run_mode})
    def amin_kernel(a: pypto.Tensor([], dtype), out: pypto.Tensor([], dtype)):
        tile_shapes = [8] * len(a.shape)
        pypto.set_vec_tile_shapes(*tile_shapes)
        out[:] = pypto.amin(a, dim=dim, keepdim=keepdim)

    out = torch.empty(out_shape, dtype=torch.float32, device=a.device)
    amin_kernel(a, out)
    return out


@pypto.options(pass_options={"enable_slice": True})
def test_amin_basic(device_id: int = None):
    """Test basic usage of amin function"""
    print("=" * 60)
    print("Test: Basic Usage of amin Function")
    print("=" * 60)

    device = f'npu:{device_id}' if global_run_mode == pypto.RunMode.NPU and device_id is not None else 'cpu'

    # Test 1: Basic reduction along the last dimension(keepdim=False)
    dtype = torch.float32
    a = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=dtype, device=device)
    expected = torch.tensor([1, 4], dtype=dtype, device=device)
    dim = -1
    out = amin_op(a, dim, keepdim=False)
    max_diff = np.abs(out.cpu().numpy() - expected.cpu().numpy()).max()
    print(f"Output (keepdim=False): {out}")
    print(f"Expected (keepdim=False): {expected}")
    print(f"Max difference: {max_diff:.6f}")
    if global_run_mode == pypto.RunMode.NPU:
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)

    # Test 2: Basic reduction along the last dimension(keepdim=True)
    dtype = torch.float32
    a = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=dtype, device=device)
    expected = torch.tensor([[1], [4]], dtype=dtype, device=device)
    dim = -1
    out = amin_op(a, dim, keepdim=True)
    max_diff = np.abs(out.cpu().numpy() - expected.cpu().numpy()).max()
    print(f"Output (keepdim=True): {out}")
    print(f"Expected (keepdim=True): {expected}")
    print(f"Max difference: {max_diff:.6f}")
    if global_run_mode == pypto.RunMode.NPU:
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)

    print("✓ Basic usage of amin function completed successfully")


@pypto.options(pass_options={"enable_slice": True})
def test_amin_different_dimensions(device_id: int = None):
    """Test reducing along different dimensions"""
    print("=" * 60)
    print("Test: Reducing Along Different Dimensions")
    print("=" * 60)

    device = f'npu:{device_id}' if global_run_mode == pypto.RunMode.NPU and device_id is not None else 'cpu'

    # Test 1: Reduction along the dim=0
    dtype = torch.float32
    a = torch.tensor(
        [[[10, 20, 30, 40], [15, 25, 35, 45], [12, 22, 32, 42]], [[5, 28, 33, 41], [18, 21, 36, 44], [11, 29, 31, 43]]],
        dtype=dtype,
        device=device,
    )
    expected = torch.tensor([[5, 20, 30, 40], [15, 21, 35, 44], [11, 22, 31, 42]], dtype=dtype, device=device)
    dim = 0
    out = amin_op(a, dim, keepdim=False)
    max_diff = np.abs(out.cpu().numpy() - expected.cpu().numpy()).max()
    print(f"Output (dim=0): {out}")
    print(f"Expected (dim=0): {expected}")
    print(f"Max difference: {max_diff:.6f}")
    if global_run_mode == pypto.RunMode.NPU:
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)

    # Test 2: Reduction along the dim=1
    expected = torch.tensor([[10, 20, 30, 40], [5, 21, 31, 41]], dtype=dtype, device=device)
    dim = 1
    out = amin_op(a, dim, keepdim=False)
    max_diff = np.abs(out.cpu().numpy() - expected.cpu().numpy()).max()
    print(f"Output (dim=1): {out}")
    print(f"Expected (dim=1): {expected}")
    print(f"Max difference: {max_diff:.6f}")
    if global_run_mode == pypto.RunMode.NPU:
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)

    # Test 3: Reduction along the dim=2
    expected = torch.tensor([[10, 15, 12], [5, 18, 11]], dtype=dtype, device=device)
    dim = -1
    out = amin_op(a, dim, keepdim=False)
    max_diff = np.abs(out.cpu().numpy() - expected.cpu().numpy()).max()
    print(f"Output (dim=-1): {out}")
    print(f"Expected (dim=-1): {expected}")
    print(f"Max difference: {max_diff:.6f}")
    if global_run_mode == pypto.RunMode.NPU:
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)

    print("✓ Reducing along different dimensions completed successfully")


# ============================================================================
# MAXIMUM Examples
# ============================================================================


def maximum_op(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    dtype = pypto.DT_FP32
    shape1 = a.shape
    _shape2 = b.shape

    @pypto.frontend.jit(runtime_options={"run_mode": global_run_mode})
    def maximum_kernel(a: pypto.Tensor([], dtype), b: pypto.Tensor([], dtype), out: pypto.Tensor([], dtype)):
        tile_shapes = [8] * len(a.shape)
        pypto.set_vec_tile_shapes(*tile_shapes)
        out[:] = pypto.maximum(a, b)

    out = torch.empty(shape1, dtype=torch.float32, device=a.device)
    maximum_kernel(a, b, out)
    return out


@pypto.options(pass_options={"enable_slice": True})
def test_maximum_basic(device_id: int = None):
    """Test basic usage of maximum function"""
    print("=" * 60)
    print("Test: Basic Usage of maximum Function")
    print("=" * 60)

    device = f'npu:{device_id}' if global_run_mode == pypto.RunMode.NPU and device_id is not None else 'cpu'

    # Test 1: Basic Usage of maximum Function
    dtype = torch.float32
    a = torch.tensor([0, 2, 4], dtype=dtype, device=device)
    b = torch.tensor([3, 1, 3], dtype=dtype, device=device)
    expected = torch.tensor([3, 2, 4], dtype=dtype, device=device)

    out = maximum_op(a, b)
    max_diff = np.abs(out.cpu().numpy() - expected.cpu().numpy()).max()
    print(f"Output: {out}")
    print(f"Expected: {expected}")
    if global_run_mode == pypto.RunMode.NPU:
        print(f"Max difference: {max_diff:.6f}")
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)

    # Test 2: Basic Usage of maximum Function with different shapes
    dtype = torch.float32
    a = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=dtype, device=device)
    b = torch.tensor([[0, 9, 2], [1, 3, 10]], dtype=dtype, device=device)
    expected = torch.tensor([[1, 9, 3], [4, 5, 10]], dtype=dtype, device=device)

    out = maximum_op(a, b)
    max_diff = np.abs(out.cpu().numpy() - expected.cpu().numpy()).max()
    print(f"Output: {out}")
    print(f"Expected: {expected}")
    if global_run_mode == pypto.RunMode.NPU:
        print(f"Max difference: {max_diff:.6f}")
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)

    print("✓ Basic usage of maximum function completed successfully")


# ============================================================================
# MINIMUM Examples
# ============================================================================


def minimum_op(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    shape = a.shape
    dtype = pypto.DT_FP32

    @pypto.frontend.jit(runtime_options={"run_mode": global_run_mode})
    def minimum_kernel(a: pypto.Tensor([], dtype), b: pypto.Tensor([], dtype), out: pypto.Tensor([], dtype)):
        tile_shapes = [8] * len(a.shape)
        pypto.set_vec_tile_shapes(*tile_shapes)
        out[:] = pypto.minimum(a, b)

    out = torch.empty(shape, dtype=torch.float32, device=a.device)
    minimum_kernel(a, b, out)
    return out


@pypto.options(pass_options={"enable_slice": True})
def test_minimum_basic(device_id: int = None):
    """Test basic usage of minimum function"""
    print("=" * 60)
    print("Test: Basic Usage of minimum Function")
    print("=" * 60)

    device = f'npu:{device_id}' if global_run_mode == pypto.RunMode.NPU and device_id is not None else 'cpu'

    # Test 1: Basic Usage of minimum Function
    dtype = torch.float32
    a = torch.tensor([0, 2, 4], dtype=dtype, device=device)
    b = torch.tensor([3, 1, 3], dtype=dtype, device=device)
    expected = torch.tensor([0, 1, 3], dtype=dtype, device=device)

    out = minimum_op(a, b)
    max_diff = np.abs(out.cpu().numpy() - expected.cpu().numpy()).max()
    print(f"Output: {out}")
    print(f"Expected: {expected}")
    if global_run_mode == pypto.RunMode.NPU:
        print(f"Max difference: {max_diff:.6f}")
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)

    # Test 2: Basic Usage of minimum Function with different shapes
    dtype = torch.float32
    a = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=dtype, device=device)
    b = torch.tensor([[0, 9, 2], [1, 3, 10]], dtype=dtype, device=device)
    expected = torch.tensor([[0, 2, 2], [1, 3, 6]], dtype=dtype, device=device)

    out = minimum_op(a, b)
    max_diff = np.abs(out.cpu().numpy() - expected.cpu().numpy()).max()
    print(f"Output: {out}")
    print(f"Expected: {expected}")
    if global_run_mode == pypto.RunMode.NPU:
        print(f"Max difference: {max_diff:.6f}")
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)

    print("✓ Basic usage of minimum function completed successfully")


# ============================================================================
# Main Function
# ============================================================================


def main():
    """Run reduce operation examples.

    Usage:
        python reduce_ops.py              # Run all examples
        python reduce_ops.py --list       # List all available examples
        python reduce_ops.py --case sum::test_sum_basic    # Run a specific case
    """
    parser = argparse.ArgumentParser(
        description="PyPTO Reduce Operation Examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                      Run all examples
  %(prog)s --list               List all available examples
  %(prog)s sum::test_sum_basic    Run a specific case
        """,
    )
    parser.add_argument(
        'example_id',
        type=str,
        nargs="?",
        help='Run a specific case (e.g., sum::test_sum_basic). If omitted, all cases run.',
    )
    parser.add_argument('--list', action='store_true', help='List all available examples and exit')
    parser.add_argument(
        '--run_mode', type=str, nargs='?', default='npu', choices=["npu", "sim"], help='Run mode, supports npu and sim.'
    )

    args = parser.parse_args()

    # Define available examples
    examples = {
        'sum::test_sum_basic': {
            'name': 'Test basic usage of sum function',
            'description': 'Basic usage of sum function example',
            'function': test_sum_basic,
        },
        'sum::test_sum_different_dimensions': {
            'name': 'Test reducing along different dimensions',
            'description': 'Reducing along different dimensions example',
            'function': test_sum_different_dimensions,
        },
        'amax::test_amax_basic': {
            'name': 'Test basic usage of amax function',
            'description': 'Basic usage of amax function example',
            'function': test_amax_basic,
        },
        'amax::test_amax_different_dimensions': {
            'name': 'Test reducing along different dimensions',
            'description': 'Reducing along different dimensions example',
            'function': test_amax_different_dimensions,
        },
        'amin::test_amin_basic': {
            'name': 'Test basic usage of amin function',
            'description': 'Basic usage of amin function example',
            'function': test_amin_basic,
        },
        'amin::test_amin_different_dimensions': {
            'name': 'Test reducing along different dimensions',
            'description': 'Reducing along different dimensions example',
            'function': test_amin_different_dimensions,
        },
        'maximum::test_maximum_basic': {
            'name': 'Test basic usage of maximum function',
            'description': 'Basic usage of maximum function example',
            'function': test_maximum_basic,
        },
        'minimum::test_minimum_basic': {
            'name': 'Test basic usage of minimum function',
            'description': 'Basic usage of minimum function example',
            'function': test_minimum_basic,
        },
    }

    # List examples if requested
    if args.list:
        print("\n" + "=" * 60)
        print("Available Examples")
        print("=" * 60 + "\n")
        for ex_id, ex_info in sorted(examples.items()):
            print(f"  ID: {ex_id}")
            print(f"     name: {ex_info['name']}")
            print(f"     description: {ex_info['description']}\n")
        return

    # Validate case if provided
    examples_to_run = []
    device_id = None
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
    print("PyPTO Reduce Operation Examples")
    print("=" * 60 + "\n")

    if args.run_mode == "npu":
        device_id = get_device_id()
        if device_id is None:
            return
        torch.npu.set_device(device_id)
        print("Running examples that require NPU hardware...")
        print("(Make sure CANN environment is configured and NPU is available)\n")

    try:
        for ex_id, ex_info in examples_to_run:
            print(f"Running Example {ex_id}: {ex_info['name']}")
            ex_info['function'](device_id)

        if len(examples_to_run) > 1:
            print("=" * 60)
            print("All reduce tests passed!")
            print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
