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
Softmax Example using aclgraph for PyPTO

This example demonstrates how to implement a softmax operation using aclgraph in PyPTO, including:
- Manual softmax computation from basic operations
- Dynamic axis marking for variable batch sizes
- Tiling configuration for efficient execution
- Loop-based processing for large tensors
- Graph capture and execution for performance optimization

Softmax is a fundamental operation in neural networks, especially for attention mechanisms.
"""

import argparse
import os
import sys

import numpy as np
from numpy.testing import assert_allclose
import torch
from torch._dynamo import allow_in_graph
from torch._subclasses.fake_tensor import FakeTensor

import pypto


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


def softmax_core(x: pypto.Tensor) -> pypto.Tensor:
    """
    Core softmax computation: exp(x - max(x)) / sum(exp(x - max(x))).

    Parameters
    ----------
    input_tensor : pypto.Tensor
        Input tensor to apply softmax to

    Returns
    -------
    pypto.tensor
        Softmax normalized tensor
    """
    row_max = pypto.amax(x, dim=-1, keepdim=True)
    sub = x - row_max
    exp = pypto.exp(sub)
    esum = pypto.sum(exp, dim=-1, keepdim=True)
    return exp / esum


B = pypto.DYNAMIC
N1, N2, DIM = 32, 1, 256


@pypto.frontend.jit()
def softmax_kernel(
    input_tensor: pypto.Tensor([B, N1, N2, DIM], pypto.DT_FP32),
    output_tensor: pypto.Tensor([B, N1, N2, DIM], pypto.DT_FP32),
):
    """
    Softmax kernel with return value. Use return pattern for aclgraph compatibility.
    See docs/zh/tutorials/network_integration/pytorch_integration.md
    """

    bs = input_tensor.shape[0]
    tile_b = 1  # Process one batch at a time
    b_loop = bs // tile_b

    # Tiling shape setting for efficient execution
    pypto.set_vec_tile_shapes(1, 4, 1, 64)

    for idx in pypto.loop(0, b_loop, 1, name="LOOP_L0_bIdx", idx_name="idx"):
        b_offset = idx * tile_b
        b_offset_end = min((idx + 1) * tile_b, bs)
        input_view = pypto.view(
            input_tensor, [tile_b, N1, N2, DIM], [b_offset, 0, 0, 0], valid_shape=[b_offset_end - b_offset, N1, N2, DIM]
        )
        softmax_out = softmax_core(input_view)
        output_tensor[b_offset:, ...] = softmax_out


@allow_in_graph
def softmax(x: torch.Tensor, dynamic: bool = True) -> torch.Tensor:
    if isinstance(x, FakeTensor):
        return torch.zeros(x.shape, dtype=x.dtype, device=f'{x.device}')

    # launch the kernel - use return pattern for aclgraph compatibility
    output_tensor = torch.empty(x.shape, dtype=x.dtype, device=f'{x.device}')
    softmax_kernel(x, output_tensor)
    return output_tensor


class MM(torch.nn.Module):
    def forward(self, x, dynamic):
        out = softmax(x, dynamic)
        return out


@pypto.options(pass_options={"enable_slice": True})
def test_softmax_capture(device_id=None, dynamic: bool = True) -> None:
    if not device_id:
        device_id = torch.npu.current_device()
    else:
        torch.npu.set_device(device_id)

    shape = (32, N1, N2, DIM)
    x = torch.rand(shape, dtype=torch.float, device=f'npu:{device_id}')

    model = torch.compile(MM(), backend="eager", dynamic=True)

    # graph capture
    g = torch.npu.NPUGraph()
    with torch.npu.graph(g):
        y = model(x, dynamic)

    # execute graph
    g.replay()
    torch.npu.synchronize()
    golden = torch.softmax(x, dim=-1).cpu()

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    assert_allclose(np.array(y.cpu()), np.array(golden), rtol=3e-3, atol=3e-3)
    print("✓ Softmax test passed")
    print()


def main():
    """Run softmax example.

    Usage:
        python softmax.py          # Run example
        python softmax.py --list   # List available examples
    """
    parser = argparse.ArgumentParser(
        description="PyPTO Softmax Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s              Run the example
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
        "aclgraph::test_aclgraph": {
            'name': 'Softmax',
            'description': 'Softmax implementation using aclgraph with dynamic batch size',
            'function': test_softmax_capture,
            'requires_npu': True,
        }
    }

    # List examples if requested
    if args.list:
        print("\n" + "=" * 60)
        print("Available Examples")
        print("=" * 60 + "\n")
        for ex_id, ex_info in sorted(examples.items()):
            npu_req = " (Requires NPU)" if ex_info['requires_npu'] else " (No NPU required)"
            print(f"  {ex_id}. {ex_info['name']}{npu_req}")
            print(f"     {ex_info['description']}\n")
        return

    # Validate example ID if provided
    if args.example_id is not None:
        if args.example_id not in examples:
            print(f"ERROR: Invalid example ID: {args.example_id}")
            print(f"Valid example IDs are: {', '.join(map(str, sorted(examples.keys())))}")
            print("\nUse --list to see all available examples.")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("PyPTO Softmax Example")
    print("=" * 60 + "\n")

    # Get and validate device ID (needed for NPU examples)
    device_id = None
    examples_to_run = []

    if args.example_id is not None:
        # Run single example
        examples_to_run = [(args.example_id, examples[args.example_id])]
    else:
        # Run all examples
        examples_to_run = list(examples.items())

    # Check if any example requires NPU
    requires_npu = any(ex_info['requires_npu'] for _, ex_info in examples_to_run)

    if requires_npu:
        device_id = get_device_id()
        if device_id is None:
            return
        # Set the device once for all examples
        torch.npu.set_device(device_id)

    try:
        for ex_id, ex_info in examples_to_run:
            if ex_info['requires_npu'] and device_id is None:
                print(f"Skipping example {ex_id} ({ex_info['name']}): NPU device not configured")
                continue

            print(f"Running Example {ex_id}: {ex_info['name']}")
            ex_info['function']()

        if len(examples_to_run) > 1:
            print("=" * 60)
            print("All softmax tests passed!")
            print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
