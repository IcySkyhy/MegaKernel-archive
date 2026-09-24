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
This test case verifies that the swimlane diagram generation for the costmodel works correctly
regardless of whether the CANN is installed in the environment or not.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

import pypto

"""
PyPTO Cost Model Simulation Example

This example demonstrates how to enable swimlane diagram generation in PyPTO's cost model mode.
The test validates that the cost analysis and swimlane visualization work correctly in
simulation environment, independent of actual NPU hardware availability.
"""


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


def safe_json_load(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data, None
    except FileNotFoundError:
        return None, "File not found"
    except json.JSONDecodeError as e:
        return None, f"Invalid json format: {e}"
    except PermissionError:
        return None, "Permission Erro"
    except Exception as e:
        return None, f"Load json fail, unknow error: {e}"


def get_out_put_path():
    out_path = "./output"
    if os.path.exists(out_path):
        # Filter by current process PID to avoid selecting other parallel examples' output dirs.
        current_pid = str(os.getpid())
        subdirs = [
            os.path.join(out_path, d)
            for d in os.listdir(out_path)
            if os.path.isdir(os.path.join(out_path, d)) and current_pid in d
        ]
        if subdirs:
            latest_dir = max(subdirs, key=os.path.getctime)
            return latest_dir
    return None


def softmax_core(input_tensor: pypto.Tensor) -> pypto.Tensor:
    row_max = pypto.amax(input_tensor, dim=-1, keepdim=True)
    sub = pypto.sub(input_tensor, row_max)
    exp = pypto.exp(sub)
    esum = pypto.sum(exp, dim=-1, keepdim=True)
    return pypto.div(exp, esum)


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
def softmax(input_tensor: pypto.Tensor(), output_tensor: pypto.Tensor()):
    tensor_shape = input_tensor.shape
    b = tensor_shape[0]  # Dynamic batch size
    n1, n2, dim = tensor_shape[1:]  # Static dimensions
    tile_b = 1  # Process one batch at a time
    b_loop = b // tile_b

    # Tiling shape setting for efficient execution
    pypto.set_vec_tile_shapes(1, 4, 1, 64)

    for idx in pypto.loop(b_loop):
        b_offset = idx * tile_b
        b_offset_end = (idx + 1) * tile_b

        # Extract batch slice
        input_view = input_tensor[b_offset:b_offset_end, :n1, :n2, :dim]

        # Apply softmax to batch slice
        softmax_out = softmax_core(input_view)

        # Assemble result back to output tensor
        pypto.assemble(softmax_out, [b_offset, 0, 0, 0], output_tensor)


@pypto.options(pass_options={"enable_slice": True})
def test_softmax(cost_model_enable=True):
    """
    Run softmax with optional cost model.

    When cost_model_enable=True, PyPTO generates simulated execution
    outputs (e.g. swimlane JSON) for analysis.
    """
    _cann_is_configed: bool = bool(os.environ.get("ASCEND_HOME_PATH"))

    # Shape for verification: NCHW format, N can be any integer number as it is defined as dynamic axis
    shape = (32, 32, 1, 256)

    # Prepare data
    input_data = torch.rand(shape, dtype=torch.float32)

    # Launch the kernel
    output_data = torch.empty(shape, dtype=torch.float32)
    softmax(input_data, output_data)

    # Verify against PyTorch reference
    torch_softmax = torch.softmax(input_data, dim=3)
    npu_data = output_data.cpu()
    torch_data = torch_softmax.cpu()

    _max_diff = np.abs(npu_data.numpy() - torch_data.numpy()).max()

    output_path = get_out_put_path()
    assert output_path

    if cost_model_enable:
        merged_swimlane, error = safe_json_load(
            os.path.join(output_path, 'CostModelSimulationOutput/merged_swimlane.json')
        )
        assert not error

    print(f"Input shape: {input_data.shape}")
    print(f"Output shape: {output_data.shape}")
    print("✓ Cost model test passed\n")
    print()


def main():
    """Run cost_model example.

    Usage:
        python cost_model.py          # Run example
        python cost_model.py --list   # List available examples
    """
    parser = argparse.ArgumentParser(
        description="PyPTO cost_model Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s cost_model::test_add_direct
            Run the cost_model::test_add_direct example
  %(prog)s --list       List all available examples
        """,
    )
    parser.add_argument(
        'example_id', type=str, nargs='?', help='Example ID to run (1). If not specified, the example will run.'
    )
    parser.add_argument('--list', action='store_true', help='List all available examples and exit')

    args = parser.parse_args()

    # Define available examples
    examples = {
        "cost_model::test_softmax": {
            'name': 'cost_model',
            'description': 'cost_model example',
            'function': test_softmax,
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
    print("PyPTO cost_model Example")
    print("=" * 60 + "\n")

    # Get and validate device ID (needed for NPU examples)
    _device_id = None
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

    try:
        for ex_id, ex_info in examples_to_run:
            print(f"Running Example {ex_id}: {ex_info['name']}")
            ex_info['function']()

        print("=" * 60)
        print("All cost_model tests passed!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
