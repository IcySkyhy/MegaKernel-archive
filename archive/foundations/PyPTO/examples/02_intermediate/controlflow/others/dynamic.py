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
Dynamic Shape Examples for PyPTO

This file demonstrates the usage of PyPTO's dynamic shape feature, which allows
kernels to handle inputs with shapes that are not known at compile time.

Key Concepts:
  1. Define dynamic dimensions using pypto.DYNAMIC
  2. Only make necessary dimensions dynamic; keep others as concrete values or Ellipsis
  3. Use pypto.view / pypto.assemble with pypto.loop for explicit tiling and
     boundary management on dynamic dimensions

Examples included:
  - dynamic_mul:       Basic dynamic batch dimension with view/assemble tiling
  - dynamic_partial:   Partial dynamic dimensions (only batch is dynamic, others concrete)
  - dynamic_attention: Multi-head attention with dynamic batch size
  - dynamic_multi_dim: Multiple dynamic dimensions in a single kernel

Usage:
    python dynamic.py                                   # Run all examples
    python dynamic.py --list                            # List all available examples
    python dynamic.py dynamic_mul::test_dynamic_mul     # Run a specific case
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


# ============================================================================
# Example 1: Basic Dynamic Batch Dimension (mul with view/assemble)
# ============================================================================
#
# This example shows the fundamental pattern for dynamic shapes:
#   - Define dynamic dimension at MODULE LEVEL (outside any function)
#   - Use pypto.view to slice tiles from the input along the dynamic axis
#   - Use pypto.assemble to write computed tiles back to the output
#   - Use min for boundary management on the last tile
#
# IMPORTANT: Dynamic dimensions MUST be defined at using pypto.DYNAMIC

# Module-level dynamic dimension definition


@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode})
def dynamic_mul_kernel(
    x: pypto.Tensor([pypto.DYNAMIC, 128], pypto.DT_FP16),
    output: pypto.Tensor([pypto.DYNAMIC, 128], pypto.DT_FP16),
    tile_b: int,
):
    batch_size_dyn = x.shape[0]
    # Compute loop count: ceil(batch_size / tile_b)
    b_loop = (batch_size_dyn + tile_b - 1) // tile_b

    for idx in pypto.loop(b_loop):
        b_offset = idx * tile_b
        # Boundary management: clamp end offset to actual batch size
        b_offset_end = min(b_offset + tile_b, batch_size_dyn)
        valid_shape = [b_offset_end - b_offset, 128]

        # View a tile from the input
        x_view = pypto.view(x, [tile_b, 128], [b_offset, 0], valid_shape=valid_shape)
        pypto.set_vec_tile_shapes(1, 128)
        result = pypto.mul(x_view, 2.0)

        # Assemble the result back into the output
        pypto.assemble(result, [b_offset, 0], output)


@pypto.options(pass_options={"enable_slice": True})
def test_dynamic_mul(device_id: int = None):
    """Test dynamic mul with different batch sizes - same kernel, no recompilation."""
    print("=" * 60)
    print("Test: Dynamic Mul (basic view/assemble tiling)")
    print("=" * 60)

    device = f'npu:{device_id}' if global_run_mode == pypto.RunMode.NPU and device_id is not None else 'cpu'

    test_batch_sizes = [8, 16]
    for bs in test_batch_sizes:
        x = torch.randn(bs, 128, dtype=torch.float16, device=device)
        result = torch.zeros(bs, 128, dtype=torch.float16, device=device)
        dynamic_mul_kernel(x, result, bs)

        if global_run_mode == pypto.RunMode.NPU:
            torch.npu.synchronize()

        golden = x * 2.0
        if global_run_mode == pypto.RunMode.NPU:
            assert_allclose(np.array(result.cpu()), np.array(golden.cpu()), rtol=1e-3, atol=1e-3)
        print(f"  batch_size={bs}: Input shape {x.shape} -> Output shape {result.shape}")

    print("✓ Dynamic mul passed for all batch sizes")
    print()


# ============================================================================
# Example 2: Partial Dynamic Dimensions (softmax-style)
# ============================================================================
#
# Pattern: Only make the batch dimension dynamic; keep seqlen, head, dim as
# concrete values. This is the recommended approach - avoid making all
# dimensions dynamic unless truly necessary.

# Module-level: only batch is dynamic


def softmax_core(input_tensor: pypto.Tensor) -> pypto.Tensor:
    """Compute softmax along the last dimension."""
    return pypto.softmax(input_tensor, dim=-1)


@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode})
def softmax_kernel(
    input_tensor: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
    output_tensor: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
):
    tile_b = 1
    bs_dyn, seqlen, head, dim = input_tensor.shape
    b_loop = bs_dyn // tile_b

    pypto.set_vec_tile_shapes(1, 4, 1, 64)

    for idx in pypto.loop(0, b_loop, 1, name="LOOP_L0_bIdx", idx_name="idx"):
        b_offset = idx * tile_b
        b_offset_end = (idx + 1) * tile_b
        # Use slicing to extract a tile (concrete dims stay as-is)
        input_view = input_tensor[b_offset:b_offset_end, :seqlen, :head, :dim]
        softmax_out = softmax_core(input_view)
        output_tensor[b_offset:b_offset_end, ...] = softmax_out


@pypto.options(pass_options={"enable_slice": True})
def test_dynamic_partial(device_id: int = None):
    """Test softmax with partial dynamic dimensions (only batch is dynamic)."""
    print("=" * 60)
    print("Test: Partial Dynamic Dimensions (softmax)")
    print("=" * 60)

    device = f'npu:{device_id}' if global_run_mode == pypto.RunMode.NPU and device_id is not None else 'cpu'

    # Fixed non-batch dimensions
    seqlen, head, dim = 32, 1, 256

    test_batch_sizes = [8, 32]
    for bs in test_batch_sizes:
        shape = (bs, seqlen, head, dim)
        x = torch.rand(shape, dtype=torch.float32, device=device)
        y = torch.zeros(shape, dtype=torch.float32, device=device)

        softmax_kernel(x, y)

        if global_run_mode == pypto.RunMode.NPU:
            torch.npu.synchronize()

        golden = torch.softmax(x, dim=-1)
        if global_run_mode == pypto.RunMode.NPU:
            assert_allclose(np.array(y.cpu()), np.array(golden.cpu()), rtol=1e-3, atol=1e-3)
        print(f"  batch_size={bs}: Input shape {x.shape} -> Output shape {y.shape}")

    print("✓ Partial dynamic softmax passed for all batch sizes")
    print()


# ============================================================================
# Example 3: Dynamic Attention (scaled dot-product attention)
# ============================================================================
#
# A more complex example: scaled dot-product attention where the batch
# dimension is dynamic. Demonstrates view/assemble on multi-dimensional
# tensors with dynamic batch axis.

# Reuse bs_dyn from Example 2


@dataclass
class AttentionConfig:
    """Configuration for attention operations."""

    num_heads: int = 8
    head_dim: int = 64
    scale: Optional[float] = None
    dtype: pypto.DataType = pypto.DT_FP32


def scaled_dot_product_attention_golden(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float, attn_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """PyTorch reference implementation of scaled dot-product attention."""
    scores = torch.matmul(q, k.transpose(-2, -1))
    scores = scores * scale
    if attn_mask is not None:
        scores = scores + attn_mask
    attn_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, v)
    return output


def scaled_dot_product_attention_core(
    q: pypto.Tensor, k: pypto.Tensor, v: pypto.Tensor, scale: float, dtype: pypto.DataType
) -> pypto.Tensor:
    """Core attention computation in PyPTO."""
    k_t = pypto.transpose(k, 2, 3)
    scores = pypto.matmul(q, k_t, out_dtype=dtype)
    scores_scaled = scores * scale
    attn_weights = pypto.softmax(scores_scaled, dim=-1)
    res = pypto.matmul(attn_weights, v, out_dtype=dtype)
    return res


@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode})
def attention_kernel(
    q: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
    k: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
    v: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
    output_tensor: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
    config: AttentionConfig,
    tile: int,
):
    """Scaled dot-product attention with dynamic batch size."""
    bs_dyn = q.shape[0]
    head = config.num_heads
    dim = config.head_dim
    q_len = q.shape[2]
    kv_len = k.shape[2]  # Tile step equals actual batch size
    scale = config.scale if config.scale is not None else (1.0 / (dim**0.5))
    cube_tiling = 64
    pypto.set_cube_tile_shapes([cube_tiling, cube_tiling], [cube_tiling, cube_tiling], [cube_tiling, cube_tiling])

    bs_loop = (bs_dyn + tile - 1) // tile

    for bss_idx in pypto.loop(bs_loop):
        bs_offset = bss_idx * tile
        bs_offset_end = min(bs_offset + tile, bs_dyn)

        # View tiles along the dynamic batch axis
        q_view = pypto.view(
            q, [tile, head, q_len, dim], [bs_offset, 0, 0, 0], valid_shape=[bs_offset_end - bs_offset, head, q_len, dim]
        )
        k_view = pypto.view(
            k,
            [tile, head, kv_len, dim],
            [bs_offset, 0, 0, 0],
            valid_shape=[bs_offset_end - bs_offset, head, kv_len, dim],
        )
        v_view = pypto.view(
            v,
            [tile, head, kv_len, dim],
            [bs_offset, 0, 0, 0],
            valid_shape=[bs_offset_end - bs_offset, head, kv_len, dim],
        )

        pypto.set_vec_tile_shapes(1, 8, 16, 64)
        res = scaled_dot_product_attention_core(q_view, k_view, v_view, scale, config.dtype)
        pypto.assemble(res, [bs_offset, 0, 0, 0], output_tensor)


@pypto.options(pass_options={"enable_slice": True})
def test_dynamic_attention(device_id: int = None):
    """Test attention with dynamic batch sizes."""
    print("=" * 60)
    print("Test: Dynamic Scaled Dot-Product Attention")
    print("=" * 60)

    device = f'npu:{device_id}' if global_run_mode == pypto.RunMode.NPU and device_id is not None else 'cpu'

    num_heads, head_dim = 8, 64
    config = AttentionConfig(num_heads=num_heads, head_dim=head_dim, dtype=pypto.DT_FP32)

    test_cases = [
        (2, 16, 16),
        (4, 32, 32),
        (8, 64, 64),
    ]
    for batch_size, seq_len_q, seq_len_kv in test_cases:
        dtype = torch.float32
        q = torch.randn(batch_size, num_heads, seq_len_q, head_dim, dtype=dtype, device=device)
        k = torch.randn(batch_size, num_heads, seq_len_kv, head_dim, dtype=dtype, device=device)
        v = torch.randn(batch_size, num_heads, seq_len_kv, head_dim, dtype=dtype, device=device)

        out = torch.empty(batch_size, num_heads, seq_len_q, head_dim, dtype=dtype, device=device)
        attention_kernel(q, k, v, out, config, batch_size)

        if global_run_mode == pypto.RunMode.NPU:
            torch.npu.synchronize()

        scale = 1.0 / (head_dim**0.5)
        golden = scaled_dot_product_attention_golden(q, k, v, scale).cpu()

        if global_run_mode == pypto.RunMode.NPU:
            out_cpu = out.cpu()
            max_diff = (out_cpu - golden).abs().max().item()
            print(f"  Batch={batch_size}, SeqQ={seq_len_q}, SeqKV={seq_len_kv}, Max diff: {max_diff:.6f}")
            assert_allclose(np.array(out_cpu), np.array(golden), rtol=3e-3, atol=3e-3)
        print(f"  Input shape: {q.shape} -> Output shape: {out.shape}")

    print("✓ Dynamic attention passed for all test cases")
    print()


# ============================================================================
# Example 4: Multiple Dynamic Dimensions
# ============================================================================
#
# When multiple dimensions are dynamic. Here both batch_size and hidden_size
# are dynamic. This pattern is useful for layer normalization or similar
# operations where both dimensions may vary.


@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode})
def dynamic_add_kernel(
    x: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP16),
    y: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP16),
    output: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP16),
    tile_b: int,
    tile_h: int,
):
    batch_dyn = x.shape[0]
    hidden_dyn = x.shape[1]
    b_loop = (batch_dyn + tile_b - 1) // tile_b

    for b_idx in pypto.loop(b_loop):
        b_offset = b_idx * tile_b
        b_offset_end = min(b_offset + tile_b, batch_dyn)
        valid_b = b_offset_end - b_offset

        h_loop = (hidden_dyn + tile_h - 1) // tile_h

        for h_idx in pypto.loop(h_loop):
            h_offset = h_idx * tile_h
            h_offset_end = min(h_offset + tile_h, hidden_dyn)
            valid_h = h_offset_end - h_offset

            x_view = pypto.view(x, [tile_b, tile_h], [b_offset, h_offset], valid_shape=[valid_b, valid_h])
            y_view = pypto.view(y, [tile_b, tile_h], [b_offset, h_offset], valid_shape=[valid_b, valid_h])

            pypto.set_vec_tile_shapes(tile_b, tile_h)
            result = pypto.add(x_view, y_view)
            pypto.assemble(result, [b_offset, h_offset], output)


@pypto.options(pass_options={"enable_slice": True})
def test_dynamic_multi_dim(device_id: int = None):
    """Test kernel with multiple dynamic dimensions."""
    print("=" * 60)
    print("Test: Multiple Dynamic Dimensions (add)")
    print("=" * 60)

    device = f'npu:{device_id}' if global_run_mode == pypto.RunMode.NPU and device_id is not None else 'cpu'

    test_cases = [
        (8, 64),
        (16, 128),
    ]
    for bs, hs in test_cases:
        x = torch.randn(bs, hs, dtype=torch.float16, device=device)
        y = torch.randn(bs, hs, dtype=torch.float16, device=device)

        result = torch.zeros(bs, hs, dtype=torch.float16, device=device)
        dynamic_add_kernel(x, y, result, bs, hs)

        if global_run_mode == pypto.RunMode.NPU:
            torch.npu.synchronize()

        golden = x + y
        if global_run_mode == pypto.RunMode.NPU:
            assert_allclose(np.array(result.cpu()), np.array(golden.cpu()), rtol=1e-3, atol=1e-3)
        print(f"  batch={bs}, hidden={hs}: Input shapes {x.shape}, {y.shape} -> Output shape {result.shape}")

    print("✓ Multiple dynamic dimensions passed for all test cases")
    print()


# ============================================================================
# Main Function
# ============================================================================


def main():
    """Run dynamic shape examples.

    Usage:
        python dynamic.py                                  # Run all examples
        python dynamic.py dynamic_mul::test_dynamic_mul    # Run a specific case
        python dynamic.py --list                           # List all available examples
    """
    parser = argparse.ArgumentParser(
        description="PyPTO Dynamic Shape Examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                       Run all examples
  %(prog)s dynamic_mul::test_dynamic_mul         Run a specific case
  %(prog)s --list                                List all available examples
        """,
    )
    parser.add_argument(
        'example_id', type=str, nargs='?', help='Example ID to run. If not specified, all examples will run.'
    )
    parser.add_argument('--list', action='store_true', help='List all available examples and exit')
    parser.add_argument(
        '--run_mode', type=str, nargs='?', default='npu', choices=["npu", "sim"], help='Run mode, supports npu and sim.'
    )

    args = parser.parse_args()

    # Define available examples
    examples = {
        'dynamic_mul::test_dynamic_mul': {
            'name': 'Basic dynamic batch mul',
            'description': 'Demonstrates basic view/assemble tiling with a single dynamic batch dimension',
            'function': test_dynamic_mul,
        },
        'dynamic_partial::test_dynamic_partial': {
            'name': 'Partial dynamic dimensions (softmax)',
            'description': 'Only batch is dynamic; seqlen, head, dim stay concrete',
            'function': test_dynamic_partial,
        },
        'dynamic_attention::test_dynamic_attention': {
            'name': 'Dynamic attention',
            'description': 'Scaled dot-product attention with dynamic batch size, view/assemble on 4D tensors',
            'function': test_dynamic_attention,
        },
        'dynamic_multi_dim::test_dynamic_multi_dim': {
            'name': 'Multiple dynamic dimensions',
            'description': 'Both batch and hidden dimensions are dynamic, nested view/assemble loops',
            'function': test_dynamic_multi_dim,
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
            print(f"Valid example IDs are: {', '.join(sorted(examples.keys()))}")
            print("\nUse --list to see all available examples.")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("PyPTO Dynamic Shape Examples")
    print("=" * 60 + "\n")

    # Get and validate device ID
    device_id = None
    examples_to_run = []

    if args.example_id is not None:
        examples_to_run = [(args.example_id, examples[args.example_id])]
    else:
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
            print("All dynamic shape tests passed!")
            print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
