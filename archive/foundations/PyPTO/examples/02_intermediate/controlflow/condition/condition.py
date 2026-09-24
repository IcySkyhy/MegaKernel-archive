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
Condition Function Examples for PyPTO

This example demonstrates:
- Nested loops with conditional statements
- Dynamic axis with static condition (compile-time bool flag)
- Dynamic axis with dynamic condition (runtime index comparison)
- Dynamic axis with loop boundary conditions (is_loop_begin / is_loop_end)
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


def _get_mode(run_mode: str):
    if global_run_mode == pypto.RunMode.NPU:
        return pypto.RunMode.NPU
    elif run_mode == "sim":
        return pypto.RunMode.SIM
    raise ValueError(f"Invalid run_mode: {run_mode}. Must be 'npu' or 'sim'")


# ============================================================================
# 1. Nested Loops with Conditions
# ============================================================================
@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode})
def nested_loops_with_conditions_kernel(
    a: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),
    b: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),
    y: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(2, 8)
    for i in pypto.loop(2):
        for j in pypto.loop(2):
            a_view = a[i:i + 1, j:j + 1]
            b_view = b[i:i + 1, j:j + 1]
            if i == 0:
                y[i:i + 1, j:j + 1] = a_view + b_view
            else:
                y[i:i + 1, j:j + 1] = a_view - b_view


@pypto.options(pass_options={"enable_slice": True})
def test_nested_loops_with_conditions(device_id=None, dynamic: bool = True) -> None:
    """Test nested loops with conditional statements."""
    print("=" * 60)
    print("Test: Nested Loops with Conditional Statements")
    print("=" * 60)

    device = f'npu:{device_id}' if global_run_mode == pypto.RunMode.NPU and device_id is not None else 'cpu'

    shape = (2, 2)
    dtype = torch.float
    a = torch.rand(shape, dtype=dtype, device=device)
    b = torch.rand(shape, dtype=dtype, device=device)
    y = torch.zeros(shape, dtype=dtype, device=device)
    nested_loops_with_conditions_kernel(a, b, y)
    golden = torch.zeros(shape, dtype=dtype, device=device)
    golden[0] = a[0] + b[0]
    golden[1] = a[1] - b[1]
    golden = golden.cpu()

    if global_run_mode == pypto.RunMode.NPU:
        assert_allclose(np.array(y.cpu()), np.array(golden.cpu()), rtol=1e-3, atol=1e-3)
        print(f"Output: {y.cpu()}")
        print(f"Expected: {golden.cpu()}")
    print("✓ Nested loops with conditional statements completed successfully")
    print()


# ============================================================================
# 2. Dynamic Axis with Static Condition
# ============================================================================


def add_core(input0: pypto.Tensor, input1: pypto.Tensor, output: pypto.Tensor, val: int, add1_flag: bool = False):
    tensor_shape = input0.shape
    pypto.set_vec_tile_shapes(1, 4, 1, 64)

    b = tensor_shape[0]
    tile_b = 1
    b_loop = b // tile_b

    for idx in pypto.loop(b_loop):
        b_offset = idx * tile_b
        b_offset_end = (idx + 1) * tile_b
        t0_sub = input0[b_offset:b_offset_end, ...]
        t1_sub = input1[b_offset:b_offset_end, ...]
        t3_sub = t0_sub + t1_sub
        if add1_flag:
            output[b_offset:b_offset_end, ...] = t3_sub + val
        else:
            output[b_offset:b_offset_end, ...] = t3_sub


@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode})
def add_scalar_loop_dyn_axis_static_cond_kernel_static(
    input0: pypto.Tensor(), input1: pypto.Tensor(), output: pypto.Tensor(), val: int, flag: bool
):
    add_core(input0, input1, output, val, flag)


@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode})
def add_scalar_loop_dyn_axis_static_cond_kernel_dynamic(
    input0: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
    input1: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
    output: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
    val: int,
    flag: bool,
):
    add_core(input0, input1, output, val, flag)


@pypto.options(pass_options={"enable_slice": True})
def test_add_scalar_loop_dyn_axis_static_cond(device_id=None) -> None:
    """Test dynamic axis with static (compile-time) condition."""
    print("=" * 60)
    print("Test: Dynamic Axis with Static Condition")
    print("=" * 60)

    device = f'npu:{device_id}' if global_run_mode == pypto.RunMode.NPU and device_id is not None else 'cpu'

    shape = (32, 32, 1, 256)
    val = 1
    input_data0 = torch.rand(shape, dtype=torch.float, device=device)
    input_data1 = torch.rand(shape, dtype=torch.float, device=device)
    print(f"Input0 shape: {input_data0.shape}")
    print(f"Input1 shape: {input_data1.shape}")

    output_data = torch.empty(shape, dtype=torch.float, device=device)
    add_scalar_loop_dyn_axis_static_cond_kernel_static(input_data0, input_data1, output_data, val, False)
    golden = torch.add(input_data0, input_data1)
    max_diff = np.abs(output_data.cpu().numpy() - golden.cpu().numpy()).max()
    print(f"Output shape (flag=False): {output_data.shape}")
    print(f"Max difference (flag=False): {max_diff:.6f}")
    if global_run_mode == pypto.RunMode.NPU:
        assert_allclose(np.array(output_data.cpu()), np.array(golden.cpu()), rtol=3e-3, atol=3e-3)

    output_data2 = torch.empty(shape, dtype=torch.float, device=device)
    add_scalar_loop_dyn_axis_static_cond_kernel_dynamic(input_data0, input_data1, output_data2, val, True)
    golden2 = torch.add(input_data0, input_data1) + val
    max_diff = np.abs(output_data2.cpu().numpy() - golden2.cpu().numpy()).max()
    print(f"Output shape (flag=True): {output_data2.shape}")
    print(f"Max difference (flag=True): {max_diff:.6f}")
    if global_run_mode == pypto.RunMode.NPU:
        assert_allclose(np.array(output_data2.cpu()), np.array(golden2.cpu()), rtol=3e-3, atol=3e-3)
    print("✓ add_scalar_loop_dyn_axis_static_cond test passed")
    print()


# ============================================================================
# 3. Dynamic Axis with Dynamic Condition
# ============================================================================
@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode})
def add_scalar_loop_dyn_axis_dyn_cond_kernel(
    input0: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
    input1: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
    output: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
    val: int,
):
    pypto.set_vec_tile_shapes(1, 4, 1, 64)
    b = input0.shape[0]
    tile_b = 1
    b_loop = b // tile_b

    for idx in pypto.loop(b_loop):
        b_offset = idx * tile_b
        b_offset_end = (idx + 1) * tile_b
        t0_sub = input0[b_offset:b_offset_end, ...]
        t1_sub = input1[b_offset:b_offset_end, ...]
        t3_sub = t0_sub + t1_sub
        if idx < 2:
            output[b_offset:b_offset_end, ...] = t3_sub + val
        else:
            output[b_offset:b_offset_end, ...] = t3_sub


@pypto.options(pass_options={"enable_slice": True})
def test_add_scalar_loop_dynamic_axis_dynamic_cond(device_id=None) -> None:
    """Test dynamic axis with dynamic (runtime) condition."""
    print("=" * 60)
    print("Test: Dynamic Axis with Dynamic Condition")
    print("=" * 60)

    device = f'npu:{device_id}' if global_run_mode == pypto.RunMode.NPU and device_id is not None else 'cpu'

    shape = (32, 32, 1, 256)
    val = 1
    input_data0 = torch.rand(shape, dtype=torch.float, device=device)
    input_data1 = torch.rand(shape, dtype=torch.float, device=device)
    output_data = torch.empty(shape, dtype=torch.float, device=device)
    add_scalar_loop_dyn_axis_dyn_cond_kernel(input_data0, input_data1, output_data, val)

    golden = torch.add(input_data0, input_data1)
    golden[0:2, ...] = golden[0:2, ...] + val

    max_diff = np.abs(output_data.cpu().numpy() - golden.cpu().numpy()).max()
    print(f"Input0 shape: {input_data0.shape}")
    print(f"Input1 shape: {input_data1.shape}")
    print(f"Output shape: {output_data.shape}")
    print(f"Max difference: {max_diff:.6f}")

    if global_run_mode == pypto.RunMode.NPU:
        assert_allclose(np.array(output_data.cpu()), np.array(golden.cpu()), rtol=3e-3, atol=3e-3)
    print("✓ add_scalar_loop_dyn_axis_dyn_cond test passed")
    print()


# ============================================================================
# 4. Dynamic Axis with Loop Boundary Conditions
# ============================================================================
@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode})
def add_scalar_loop_dyn_axis_dyn_loop_cond_kernel(
    input0: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
    input1: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
    output: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
    val: int,
):
    pypto.set_vec_tile_shapes(1, 4, 1, 64)
    b = input0.shape[0]
    tile_b = 1
    b_loop = b // tile_b

    for idx in pypto.loop(b_loop):
        b_offset = idx * tile_b
        b_offset_end = (idx + 1) * tile_b
        t0_sub = input0[b_offset:b_offset_end, ...]
        t1_sub = input1[b_offset:b_offset_end, ...]
        t3_sub = t0_sub + t1_sub
        if pypto.is_loop_begin(idx):
            output[b_offset:b_offset_end, ...] = t3_sub + val
        elif pypto.is_loop_end(idx):
            output[b_offset:b_offset_end, ...] = t3_sub + val + 1
        else:
            output[b_offset:b_offset_end, ...] = t3_sub


@pypto.options(pass_options={"enable_slice": True})
def test_add_scalar_loop_dynamic_axis_dynamic_loop_cond(device_id=None) -> None:
    """Test dynamic axis with loop boundary conditions (is_loop_begin / is_loop_end)."""
    print("=" * 60)
    print("Test: Dynamic Axis with Loop Boundary Conditions")
    print("=" * 60)

    device = f'npu:{device_id}' if global_run_mode == pypto.RunMode.NPU and device_id is not None else 'cpu'

    shape = (32, 32, 1, 256)
    val = 1
    input_data0 = torch.rand(shape, dtype=torch.float, device=device)
    input_data1 = torch.rand(shape, dtype=torch.float, device=device)
    output_data = torch.empty(shape, dtype=torch.float, device=device)
    add_scalar_loop_dyn_axis_dyn_loop_cond_kernel(input_data0, input_data1, output_data, val)

    golden = torch.add(input_data0, input_data1)
    golden[0:1, ...] = golden[0:1, ...] + val
    golden[31:32, ...] = golden[31:32, ...] + val + 1

    max_diff = np.abs(output_data.cpu().numpy() - golden.cpu().numpy()).max()
    print(f"Input0 shape: {input_data0.shape}")
    print(f"Input1 shape: {input_data1.shape}")
    print(f"Output shape: {output_data.shape}")
    print(f"Max difference: {max_diff:.6f}")

    if global_run_mode == pypto.RunMode.NPU:
        assert_allclose(np.array(output_data.cpu()), np.array(golden.cpu()), rtol=3e-3, atol=3e-3)
    print("✓ add_scalar_loop_dyn_axis_dyn_loop_cond test passed")
    print()


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="PyPTO Condition Function Examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                                                          Run all examples
  %(prog)s nested_loops_with_conditions::test_nested_loops_with_conditions           Run nested loops example
  %(prog)s dyn_axis_static_cond::test_add_scalar_loop_dyn_axis_static_cond          Run static cond example
  %(prog)s --list                                                                   List all examples
        """,
    )
    parser.add_argument('example_id', type=str, nargs='?', help='Run a specific case. If omitted, all cases run.')
    parser.add_argument('--list', action='store_true', help='List available examples')
    parser.add_argument(
        '--run_mode', type=str, nargs='?', default="npu", choices=["npu", "sim"], help='Run mode, supports npu and sim.'
    )

    args = parser.parse_args()

    examples = {
        'nested_loops_with_conditions::test_nested_loops_with_conditions': {
            'name': 'Nested loops with conditional statements',
            'description': 'Basic if/else inside nested loops',
            'function': test_nested_loops_with_conditions,
        },
        'dyn_axis_static_cond::test_add_scalar_loop_dyn_axis_static_cond': {
            'name': 'Dynamic axis with static condition',
            'description': 'Compile-time bool flag controls loop body behavior',
            'function': test_add_scalar_loop_dyn_axis_static_cond,
        },
        'dyn_axis_dyn_cond::test_add_scalar_loop_dynamic_axis_dynamic_cond': {
            'name': 'Dynamic axis with dynamic condition',
            'description': 'Runtime index comparison (if idx < 2) in loop',
            'function': test_add_scalar_loop_dynamic_axis_dynamic_cond,
        },
        'dyn_axis_dyn_loop_cond::test_add_scalar_loop_dynamic_axis_dynamic_loop_cond': {
            'name': 'Dynamic axis with loop boundary conditions',
            'description': 'is_loop_begin / is_loop_end for boundary handling',
            'function': test_add_scalar_loop_dynamic_axis_dynamic_loop_cond,
        },
    }

    if args.list:
        print("\n" + "=" * 60)
        print("Available Examples")
        print("=" * 60 + "\n")
        for ex_id, ex_info in sorted(examples.items()):
            print(f"  ID: {ex_id}")
            print(f"     name: {ex_info['name']}")
            print(f"     description: {ex_info['description']}\n")
        return

    if args.example_id is not None:
        if args.example_id not in examples:
            print(f"ERROR: Invalid example ID: {args.example_id}")
            print(f"Valid example IDs are: {', '.join(sorted(examples.keys()))}")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("PyPTO Condition Function Examples")
    print("=" * 60 + "\n")

    device_id = None
    if args.example_id is not None:
        examples_to_run = [(args.example_id, examples[args.example_id])]
    else:
        examples_to_run = list(examples.items())

    if args.run_mode == "npu":
        device_id = get_device_id()
        if device_id is None:
            return
        torch.npu.set_device(device_id)

    try:
        for ex_id, ex_info in examples_to_run:
            print(f"Running Example {ex_id}: {ex_info['name']}")
            ex_info['function'](device_id)

        if len(examples_to_run) > 1:
            print("=" * 60)
            print("All condition tests passed!")
            print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
