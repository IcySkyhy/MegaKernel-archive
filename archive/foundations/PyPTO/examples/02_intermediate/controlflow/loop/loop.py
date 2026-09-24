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
Loop Feature Examples for PyPTO

This example demonstrates:
- Basic loop usage with start/stop/step
- Loop compile phase print feature
- Loop with scalar addition (add_scalar_loop)
- Loop with dynamic axis (add_scalar_loop_dyn_axis)
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
# 1. Basic Loop Usage
# ============================================================================
@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode})
def loop_basic_kernel(
    t0: pypto.Tensor(), t1: pypto.Tensor(), out0: pypto.Tensor(), out1: pypto.Tensor(), s: int, n: int
):
    pypto.set_vec_tile_shapes(64, 64)
    for bs_idx in pypto.loop(0, n, 1):  # start, stop, step
        t0s = t0[bs_idx * s:(bs_idx + 1) * s, :]
        t1s = t1[bs_idx * s:(bs_idx + 1) * s, :]
        out0[bs_idx * s:(bs_idx + 1) * s, :] = pypto.add(t0s, t1s)
    new_step = 2
    for bs_idx in pypto.loop(0, n, new_step):  # start, stop, step
        t0s = t0[bs_idx * s:(bs_idx + new_step) * s, :]
        t1s = t1[bs_idx * s:(bs_idx + new_step) * s, :]
        out1[bs_idx * s:(bs_idx + new_step) * s, :] = pypto.add(t0s, t1s)


@pypto.options(pass_options={"enable_slice": True})
def test_loop_basic(device_id: int = None, dynamic: bool = False) -> None:
    """Test basic loop usage."""
    print("=" * 60)
    print("Test: Basic Loop Usage")
    print("=" * 60)

    device = f'npu:{device_id}' if global_run_mode == pypto.RunMode.NPU and device_id is not None else 'cpu'

    s, n = 64, 8
    shape = (n * s, s)
    input_t1 = torch.randn(shape, dtype=torch.float16, device=device)
    input_t2 = torch.randn(shape, dtype=torch.float16, device=device)
    output1 = torch.empty(shape, dtype=torch.float16, device=device)
    output2 = torch.empty(shape, dtype=torch.float16, device=device)
    loop_basic_kernel(input_t1, input_t2, output1, output2, s, n)

    expected = input_t1 + input_t2
    if global_run_mode == pypto.RunMode.NPU:
        max_diff1 = (output1 - expected).abs().max().item()
        max_diff2 = (output2 - expected).abs().max().item()
        equal_output_1_2 = (output1 - output2).abs().max().item() < 1e-6
        print(f"Whether output1 equals output2: {equal_output_1_2}")
        assert max_diff1 < 1e-2, "Result mismatch!"
        assert max_diff2 < 1e-2, "Result mismatch!"
    print("✓ Basic loop usage completed successfully")
    print()


# ============================================================================
# 2. Loop Compile Phase Print
# ============================================================================


@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode})
def loop_compile_phase_print_kernel(
    in_t0: pypto.Tensor(), in_t1: pypto.Tensor(), out_t0: pypto.Tensor(), out_t1: pypto.Tensor()
):
    pypto.set_vec_tile_shapes(64, 64)
    note = '''
    Below are demonstrations of print usage within loops.
    It executes only during compilation, cannot truly print variable values,
    and the number of prints is related to the number of subgraphs generated.
    '''
    separator = "*" * 60
    print(note)
    print(separator)
    cnt_inside_cond = 0
    cnt_outside_cond = 0
    for outside_idx in pypto.loop(5):
        print(f"outside_idx: {outside_idx}")
        for inside_idx in pypto.loop(3):
            print(f"inside_idx: {inside_idx}")
            res = pypto.add(in_t0, in_t0)
            print(f"res: {res}")
            if outside_idx < 3:
                print(f"(outside_idx < 3)_count: {cnt_outside_cond}")
                cnt_outside_cond = cnt_outside_cond + 1
                res = pypto.add(in_t0, in_t0)
            else:
                res = pypto.sub(in_t0, in_t0)
            if inside_idx < 2:
                print(f"(inside_idx < 2)_count: {cnt_inside_cond}")
                cnt_inside_cond = cnt_inside_cond + 1
                res = pypto.div(in_t0, in_t0)
            else:
                res = pypto.add(in_t1, in_t1)
            out_t0.move(pypto.add(in_t0, in_t0))
            out_t1.move(pypto.add(in_t1, in_t1))
    print(separator)


@pypto.options(pass_options={"enable_slice": True})
def test_loop_compile_phase_print(device_id: int = None, dynamic: bool = False) -> None:
    """Test loop compile phase print"""
    print("=" * 60)
    print("Test: Loop Compile Phase Print Feature")
    print("=" * 60)

    device = f'npu:{device_id}' if global_run_mode == pypto.RunMode.NPU and device_id is not None else 'cpu'

    m, n = 6, 8
    shape = (m, n)
    input_t1 = torch.randn(shape, dtype=torch.float16, device=device)
    input_t2 = torch.randn(shape, dtype=torch.float16, device=device)
    output_t1 = torch.empty(shape, dtype=torch.float16, device=device)
    output_t2 = torch.empty(shape, dtype=torch.float16, device=device)
    loop_compile_phase_print_kernel(input_t1, input_t2, output_t1, output_t2)
    expected_t1 = input_t1 + input_t1
    expected_t2 = input_t2 + input_t2
    if global_run_mode == pypto.RunMode.NPU:
        max_diff_t1 = (output_t1 - expected_t1).abs().max().item()
        max_diff_t2 = (output_t2 - expected_t2).abs().max().item()
        print(f"Max difference from PyTorch: {max_diff_t1:.6f}")
        print(f"Max difference from PyTorch: {max_diff_t2:.6f}")
        assert max_diff_t1 < 1e-2, "Result mismatch!"
        assert max_diff_t2 < 1e-2, "Result mismatch!"
    print("✓ Test loop compile phase print completed successfully")
    print()


# ============================================================================
# 3. Loop with Scalar Addition (add_scalar_loop)
# ============================================================================
@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode})
def add_kernel(
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
        t3_sub = t3_sub + val
        pypto.assemble(t3_sub, [b_offset, 0, 0, 0], output)


@pypto.options(pass_options={"enable_slice": True})
def test_add_scalar_loop(device_id=None, dynamic: bool = True) -> None:
    """Test loop-based scalar addition."""
    print("=" * 60)
    print("Test: Loop with Scalar Addition (add_scalar_loop)")
    print("=" * 60)

    device = f'npu:{device_id}' if global_run_mode == pypto.RunMode.NPU and device_id is not None else 'cpu'

    shape = (32, 32, 1, 256)
    val = 1
    x = torch.rand(shape, dtype=torch.float32, device=device)
    y = torch.rand(shape, dtype=torch.float32, device=device)
    z = torch.empty(shape, dtype=torch.float32, device=device)
    add_kernel(x, y, z, val)
    golden = torch.add(x, y) + val

    max_diff = np.abs(z.cpu().numpy() - golden.cpu().numpy()).max()
    print(f"Input0 shape : {x.shape}")
    print(f"Input1 shape : {y.shape}")
    print(f"Output shape: {z.shape}")
    print(f"Max difference: {max_diff:.6f}")

    if global_run_mode == pypto.RunMode.NPU:
        assert_allclose(np.array(z.cpu()), np.array(golden.cpu()), rtol=3e-3, atol=3e-3)
    print("✓ add_scalar_loop test passed")
    print()


# ============================================================================
# 4. Loop with Dynamic Axis (add_scalar_loop_dyn_axis)
# ============================================================================
@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode})
def add_scalar_loop_dynamic_axis_kernel(
    input0: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
    input1: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
    output: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
    val: int,
):
    pypto.set_vec_tile_shapes(1, 4, 1, 64)

    b, w, n, c = input0.shape
    tile_b = 1
    b_loop = b // tile_b

    for idx in pypto.loop(b_loop):
        b_offset = idx * tile_b
        b_offset_end = min((idx + 1) * tile_b, b)

        valid_shape = [b_offset_end - b_offset, w, n, c]

        t0_sub = pypto.view(input0, [tile_b, w, n, c], [b_offset, 0, 0, 0], valid_shape=valid_shape)
        t1_sub = pypto.view(input1, [tile_b, w, n, c], [b_offset, 0, 0, 0], valid_shape=valid_shape)
        t3_sub = t0_sub + t1_sub
        t3_sub = t3_sub + val
        pypto.assemble(t3_sub, [b_offset, 0, 0, 0], output)


@pypto.options(pass_options={"enable_slice": True})
def test_add_scalar_loop_dyn_axis(device_id: int = None) -> None:
    """Test loop with dynamic axis."""
    print("=" * 60)
    print("Test: Loop with Dynamic Axis (add_scalar_loop_dyn_axis)")
    print("=" * 60)

    device = f'npu:{device_id}' if global_run_mode == pypto.RunMode.NPU and device_id is not None else 'cpu'

    shape = (32, 32, 1, 256)
    val = 1
    input_data0 = torch.rand(shape, dtype=torch.float, device=device)
    input_data1 = torch.rand(shape, dtype=torch.float, device=device)
    output_data = torch.empty(shape, dtype=torch.float, device=device)
    add_scalar_loop_dynamic_axis_kernel(input_data0, input_data1, output_data, val)
    golden = torch.add(input_data0, input_data1) + val

    max_diff = np.abs(output_data.cpu().numpy() - golden.cpu().numpy()).max()
    print(f"Input0 shape: {input_data0.shape}")
    print(f"Input1 shape: {input_data1.shape}")
    print(f"Output shape: {output_data.shape}")
    print(f"Max difference: {max_diff:.6f}")

    if global_run_mode == pypto.RunMode.NPU:
        assert_allclose(np.array(output_data.cpu()), np.array(golden.cpu()), rtol=3e-3, atol=3e-3)
    print("✓ add_scalar_loop_dyn_axis test passed")
    print()


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="PyPTO Loop Feature Examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                              Run all examples
  %(prog)s loop_basic::test_loop_basic                  Run basic loop example
  %(prog)s add_scalar_loop::test_add_scalar_loop        Run scalar loop example
  %(prog)s --list                                       List all available examples
        """,
    )
    parser.add_argument('example_id', type=str, nargs='?', help='Run a specific case. If omitted, all cases run.')
    parser.add_argument('--list', action='store_true', help='List available examples')
    parser.add_argument(
        '--run_mode', type=str, nargs='?', default="npu", choices=["npu", "sim"], help='Run mode, supports npu and sim.'
    )

    args = parser.parse_args()

    examples = {
        'loop_basic::test_loop_basic': {
            'name': 'Test basic loop usage',
            'description': 'Basic loop with start/stop/step',
            'function': test_loop_basic,
        },
        'loop_compile_phase_print::test_loop_compile_phase_print': {
            'name': 'Test loop compile phase print',
            'description': 'Loop compile phase print feature',
            'function': test_loop_compile_phase_print,
        },
        'add_scalar_loop::test_add_scalar_loop': {
            'name': 'Test add_scalar_loop',
            'description': 'Loop-based scalar addition with dynamic batch',
            'function': test_add_scalar_loop,
        },
        'add_scalar_loop_dyn_axis::test_add_scalar_loop_dyn_axis': {
            'name': 'Test add_scalar_loop with dynamic axis',
            'description': 'Loop with dynamic axis using view/assemble and valid_shape',
            'function': test_add_scalar_loop_dyn_axis,
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
    print("PyPTO Loop Examples")
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
            print("All loop tests passed!")
            print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
