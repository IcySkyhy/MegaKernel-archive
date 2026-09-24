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
add_scalar_loop_view_assemble Example for PyPTO

This example demonstrates how to implement a add_scalar_loop_view_assemble operation using PyPTO, including:
- Manual add_scalar_loop_view_assemble computation from basic operations
- Dynamic axis marking for variable batch sizes
- Tiling configuration for efficient execution
- Loop-based processing for large tensors

add_scalar_loop_view_assemble is a fundamental operation in neural networks, especially for attention mechanisms.
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


SHAPE = (32, 32, 1, 256)


@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode})
def add_scalar_loop_view_assemble_kernel(
    input0: pypto.Tensor(SHAPE, pypto.DT_FP32),
    input1: pypto.Tensor(SHAPE, pypto.DT_FP32),
    output: pypto.Tensor(SHAPE, pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(1, 4, 1, 64)

    # Calculate the loop parameters
    b, n, s, d = SHAPE
    tile_b = 1
    b_loop = b // tile_b

    for idx in pypto.loop(b_loop):
        b_offset = idx * tile_b
        t0_sub = pypto.view(input0, [tile_b, n, s, d], [b_offset, 0, 0, 0])
        t1_sub = pypto.view(input1, [tile_b, n, s, d], [b_offset, 0, 0, 0])
        t3_sub = t0_sub + t1_sub
        pypto.assemble(t3_sub, [b_offset, 0, 0, 0], output)


@pypto.options(pass_options={"enable_slice": True})
def test_add_scalar_loop_view_assemble(device_id=None, dynamic: bool = True) -> None:
    device = f'npu:{device_id}' if global_run_mode == pypto.RunMode.NPU and device_id is not None else 'cpu'

    shape = (32, 32, 1, 256)
    # prepare data
    x = torch.rand(shape, dtype=torch.float, device=device)
    y = torch.rand(shape, dtype=torch.float, device=device)
    out = torch.empty(shape, dtype=torch.float, device=device)
    add_scalar_loop_view_assemble_kernel(x, y, out)
    golden = torch.add(x, y)

    max_diff = np.abs(out.cpu().numpy() - golden.cpu().numpy()).max()
    print(f"Input0 shape: {x.shape}")
    print(f"Input1 shape: {y.shape}")

    if global_run_mode == pypto.RunMode.NPU:
        print(f"Max difference: {max_diff:.6f}")
        assert_allclose(np.array(out.cpu()), np.array(golden.cpu()), rtol=3e-3, atol=3e-3)
    print("✓ add_scalar_loop_view_assemble test passed")
    print()


def main():
    """Run add_scalar_loop_view_assemble example.

    Usage:
        python add_scalar_loop_view_assemble.py          # Run example
        python add_scalar_loop_view_assemble.py --list   # List available examples
    """
    parser = argparse.ArgumentParser(
        description="PyPTO Softmax Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s add_scalar_loop_view_assemble::test_add_scalar_loop_view_assemble
            Run the add_scalar_loop_view_assemble::test_add_scalar_loop_view_assemble example
  %(prog)s --list       List all available examples
        """,
    )
    parser.add_argument(
        'example_id', type=str, nargs='?', help='Example ID to run (1). If not specified, the example will run.'
    )
    parser.add_argument('--list', action='store_true', help='List all available examples and exit')
    parser.add_argument(
        '--run_mode', type=str, nargs='?', default="npu", choices=["npu", "sim"], help='Run mode, supports npu and sim.'
    )

    args = parser.parse_args()

    # Define available examples
    examples = {
        "add_scalar_loop_view_assemble::test_add_scalar_loop_view_assemble": {
            'name': 'add_scalar_loop_view_assemble',
            'description': 'add_scalar_loop_view_assemble implementation with dynamic batch size',
            'function': test_add_scalar_loop_view_assemble,
        }
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

    # Validate example ID if provided
    if args.example_id is not None:
        if args.example_id not in examples:
            print(f"ERROR: Invalid example ID: {args.example_id}")
            print(f"Valid example IDs are: {', '.join(map(str, sorted(examples.keys())))}")
            print("\nUse --list to see all available examples.")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("PyPTO add_scalar_loop_view_assemble Example")
    print("=" * 60 + "\n")

    # Get and validate device ID (needed for NPU examples)
    device_id = None
    examples_to_run = []

    if args.example_id is not None:
        # Run single example
        example = examples.get(args.example_id)
        if example is None:
            raise ValueError(f"Invalid example ID: {args.example_id}")
        examples_to_run = [(args.example_id, example)]
    else:
        # Run all examples
        examples_to_run = list(examples.items())

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
            print("All add_scalar_loop_view_assemble tests passed!")
            print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
