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
kenel_unordered_input Axis Example for PyPTO

This example demonstrates:
- Run attention module with kenel_unordered_input.
"""

import argparse
from dataclasses import dataclass
import os
import sys
from typing import Optional

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


@dataclass
class AttentionConfig:
    """Configuration for attention operations."""

    num_heads: int = 8
    head_dim: int = 64
    scale: Optional[float] = None  # If None, uses 1/sqrt(head_dim)
    dtype: pypto.DataType = pypto.DT_FP32
    use_dynamic_shape: bool = False


def scaled_dot_product_attention_golden(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float, attn_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """PyTorch reference implementation of scaled dot-product attention."""
    # Compute attention scores: Q @ K^T
    scores = torch.matmul(q, k.transpose(-2, -1))  # [batch, num_heads, seq_len_q, seq_len_kv]
    scores = scores * scale
    # Apply attention mask if provided
    if attn_mask is not None:
        scores = scores + attn_mask
    attn_weights = torch.softmax(scores, dim=-1)  # [batch, num_heads, seq_len_q, seq_len_kv]
    output = torch.matmul(attn_weights, v)  # [batch, num_heads, seq_len_q, head_dim]
    return output


def scaled_dot_product_attention_core(
    q: pypto.Tensor, k: pypto.Tensor, v: pypto.Tensor, scale: float, dtype: pypto.DataType
) -> pypto.Tensor:
    k_t = pypto.transpose(k, 2, 3)
    scores = pypto.matmul(q, k_t, out_dtype=dtype)
    scores_scaled = scores * scale
    attn_weights = pypto.softmax(scores_scaled, dim=-1)
    res = pypto.matmul(attn_weights, v, out_dtype=dtype)
    return res


@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode})
def scaled_dot_product_attention_kernel(
    q: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
    k: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
    v: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
    output_tensor: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
    config: AttentionConfig,
    tile: int,
):
    """Scaled dot-product attention with dynamic batch size."""
    cube_tiling = 64
    pypto.set_cube_tile_shapes([cube_tiling, cube_tiling], [cube_tiling, cube_tiling], [cube_tiling, cube_tiling])
    bs = q.shape[0]
    head = 8
    dim = 64
    q_len = q.shape[2]
    kv_len = k.shape[2]
    scale = config.scale if config.scale is not None else (1.0 / (dim**0.5))

    b_loop = (bs + tile - 1) // tile

    for bs_idx in pypto.loop(b_loop):
        b_offset = bs_idx * tile
        b_offset_end = min(b_offset + tile, bs)
        q_view = pypto.view(
            q, [tile, head, q_len, dim], [b_offset, 0, 0, 0], valid_shape=[b_offset_end - b_offset, head, q_len, dim]
        )
        k_view = pypto.view(
            k, [tile, head, kv_len, dim], [b_offset, 0, 0, 0], valid_shape=[b_offset_end - b_offset, head, kv_len, dim]
        )
        v_view = pypto.view(
            v, [tile, head, kv_len, dim], [b_offset, 0, 0, 0], valid_shape=[b_offset_end - b_offset, head, kv_len, dim]
        )
        pypto.set_vec_tile_shapes(1, 8, 16, 64)
        res = scaled_dot_product_attention_core(q_view, k_view, v_view, scale, config.dtype)
        pypto.assemble(res, [b_offset, 0, 0, 0], output_tensor)


@pypto.options(pass_options={"enable_slice": True})
def test_unordered_input_attention(device_id: int = None, dynamic: bool = True) -> None:
    """Test attention with kenel_unordered_input."""
    print("=" * 60)
    print("Test: kenel_unordered_input Scaled Dot-Product Attention")
    print("=" * 60)

    device = f'npu:{device_id}' if global_run_mode == pypto.RunMode.NPU and device_id is not None else 'cpu'

    num_heads, head_dim = 8, 64

    batch_size, seq_len_q, seq_len_kv = 8, 64, 64
    dtype = torch.float32
    q_torch = torch.randn(batch_size, num_heads, seq_len_q, head_dim, dtype=dtype, device=device)
    k_torch = torch.randn(batch_size, num_heads, seq_len_kv, head_dim, dtype=dtype, device=device)
    v_torch = torch.randn(batch_size, num_heads, seq_len_kv, head_dim, dtype=dtype, device=device)
    config = AttentionConfig(num_heads=num_heads, head_dim=head_dim, dtype=pypto.DT_FP32, use_dynamic_shape=True)

    _q_shape = q_torch.shape
    _k_shape = k_torch.shape
    # Execute
    out_torch = torch.empty(batch_size, num_heads, seq_len_q, head_dim, dtype=dtype, device=device)
    scaled_dot_product_attention_kernel(q_torch, k_torch, v_torch, out_torch, config, batch_size)
    # Verify
    scale = 1.0 / (head_dim**0.5)
    golden = scaled_dot_product_attention_golden(q_torch, k_torch, v_torch, scale)

    print(f"Batch={batch_size}, SeqQ={seq_len_q}, SeqKV={seq_len_kv}")
    print(f"Input shape: {q_torch.shape}")
    print(f"Output shape: {out_torch.shape}")
    if global_run_mode == pypto.RunMode.NPU:
        assert_allclose(np.array(out_torch.cpu()), np.array(golden.cpu()), rtol=3e-3, atol=3e-3)

    print("✓ Attention (kenel_unordered_input) passed for the test case")
    print()


@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode})
def op_unordered_input_kernel(
    a: pypto.Tensor([], pypto.DT_FP32),
    b: pypto.Tensor([], pypto.DT_FP32),
    out1: pypto.Tensor([], pypto.DT_FP32),
    out2: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(16, 16)
    out1.move(a + b)
    out2.move(a * b)


@pypto.options(pass_options={"enable_slice": True})
def test_unordered_input_op(device_id: int = None, dynamic: bool = False) -> None:
    """Test op with kenel_unordered_input"""
    print("=" * 60)
    print("Test: OP with kenel_unordered_input")
    print("=" * 60)

    device = f'npu:{device_id}' if global_run_mode == pypto.RunMode.NPU and device_id is not None else 'cpu'

    shape = (3, 2)
    dtype = torch.float32
    a = torch.rand(shape, dtype=dtype, device=device)
    b = torch.rand(shape, dtype=dtype, device=device)
    # Execute
    y1 = torch.empty(shape, dtype=dtype, device=device)
    y2 = torch.empty(shape, dtype=dtype, device=device)
    op_unordered_input_kernel(a, b, y1, y2)
    y1, y2 = y1.cpu(), y2.cpu()
    # Verify
    golden1 = torch.add(a, b).cpu()
    golden2 = torch.mul(a, b).cpu()

    if global_run_mode == pypto.RunMode.NPU:
        assert_allclose(np.array(y1), np.array(golden1), rtol=1e-3, atol=1e-3)
        assert_allclose(np.array(y2), np.array(golden2), rtol=1e-3, atol=1e-3)
        print(f"Output1: {y1}")
        print(f"Expected1: {golden1}")
        print(f"Output2: {y2}")
        print(f"Expected2: {golden2}")

    print("✓ OP with kenel_unordered_input passed for the test case")
    print()


def main():
    """Run dynamic examples.

    Usage:
        python dynamic.py          # Run all examples
        python dynamic.py 1         # Run example 1 only
        python dynamic.py --list   # List all available examples
    """
    parser = argparse.ArgumentParser(
        description="PyPTO Full Function Examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s              Run all examples
  %(prog)s unordered_input_op::test_unordered_input_op
            Run example unordered_input_op::test_unordered_input_op
  %(prog)s --list       List all available examples
        """,
    )
    parser.add_argument(
        'example_id', type=str, nargs='?', help='Example ID to run (1-2). If not specified, all examples will run.'
    )
    parser.add_argument('--list', action='store_true', help='List all available examples and exit')
    parser.add_argument(
        '--run_mode', type=str, nargs='?', default='npu', choices=["npu", "sim"], help='Run mode, supports npu and sim.'
    )

    args = parser.parse_args()

    # Define available examples
    examples = {
        'unordered_input_attention::test_unordered_input_attention': {
            'name': 'Test attention with kenel_unordered_input',
            'description': 'Attention with kenel_unordered_input example',
            'function': test_unordered_input_attention,
            'requires_npu': True,
        },
        'unordered_input_op::test_unordered_input_op': {
            'name': 'Test op with kenel_unordered_input',
            'description': 'OP with kenel_unordered_input example',
            'function': test_unordered_input_op,
            'requires_npu': True,
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

    # Validate example ID if provided
    if args.example_id is not None:
        if args.example_id not in examples:
            print(f"ERROR: Invalid example ID: {args.example_id}")
            print(f"Valid example IDs are: {', '.join(map(str, sorted(examples.keys())))}")
            print("\nUse --list to see all available examples.")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("PyPTO Dynamic Function Examples")
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
            print("All kenel_unordered_input tests passed!")
            print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
