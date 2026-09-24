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
Multi-Function Module Example for PyPTO

This example demonstrates how to use multiple `@pypto.frontend.jit` functions together
to build complex computation pipelines. It shows:
- Multiple JIT-compiled functions
- Data flow between functions
- Function composition patterns
- Reusing compiled functions
- Switching between different functions at runtime

This pattern is useful for building modular neural network components.
"""

import argparse
from dataclasses import dataclass
import os
import sys

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
class ModuleConfig:
    """Configuration for multi-function module."""

    hidden_size: int = 128
    intermediate_size: int = 256
    dtype: pypto.DataType = pypto.DT_BF16
    use_dynamic_shape: bool = False


# Reference implementations for verification
def layer_norm_golden(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float) -> torch.Tensor:
    """PyTorch reference for layer norm."""
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    normalized = (x - mean) / torch.sqrt(var + eps)
    return normalized * gamma + beta


def gelu_golden(x: torch.Tensor) -> torch.Tensor:
    """PyTorch reference for GELU."""
    return torch.nn.functional.gelu(x)


# Function 1: Layer Normalization
def layernorm_core(x: pypto.Tensor, gamma: pypto.Tensor, beta: pypto.Tensor, eps: float = 1e-6) -> pypto.Tensor:
    # Compute mean
    hidden_size = x.shape[-1]
    mean = pypto.sum(x, dim=-1, keepdim=True)
    mean = mean / hidden_size

    centered = x - mean

    squared = centered * centered
    var = pypto.sum(squared, dim=-1, keepdim=True)
    var = var / hidden_size

    var_eps = var + eps
    std = pypto.sqrt(var_eps)
    normalized = centered / std

    scaled = normalized * gamma
    return scaled + beta


@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode})
def layer_norm_kernel(x: pypto.Tensor(), gamma: pypto.Tensor(), beta: pypto.Tensor(), out: pypto.Tensor()):
    """Layer Normalization."""
    pypto.set_vec_tile_shapes(64, 128)

    out[:] = layernorm_core(x, gamma, beta)


# Function 2: Linear Projection
@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode})
def linear_projection_kernel(x: pypto.Tensor(), weight: pypto.Tensor(), out: pypto.Tensor()):
    bias = None

    pypto.set_cube_tile_shapes([64, 64], [64, 64], [64, 64])
    # Matrix multiplication
    if bias is not None:
        out[:] = pypto.add(pypto.matmul(x, weight, out_dtype=x.dtype), bias)
    else:
        out[:] = pypto.matmul(x, weight, out_dtype=x.dtype)


# Function 3: GELU Activation
@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode})
def gelu_activation_kernel(x: pypto.Tensor(), out: pypto.Tensor()):
    # Configure tiling
    tile_shapes = [32] * len(x.shape)
    pypto.set_vec_tile_shapes(*tile_shapes)

    # GELU approximation: x * sigmoid(1.702 * x)
    coeff = 1.702
    x_scaled = x * coeff

    out[:] = x * pypto.sigmoid(x_scaled)


# Function 4: Residual Connection
@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode})
def residual_add_kernel(x: pypto.tensor(), residual: pypto.tensor(), out: pypto.tensor()):
    pypto.set_vec_tile_shapes(64, 128)
    out[:] = pypto.add(x, residual)


# Function 5: Attention (simplified)
@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode})
def attention_kernel(q: pypto.tensor(), k: pypto.tensor(), v: pypto.tensor(), out: pypto.tensor()):
    pypto.set_cube_tile_shapes([64, 64], [64, 64], [64, 64])

    # Q @ K^T
    k_t = pypto.transpose(k, [0, 1, 3, 2])
    scores = pypto.matmul(q, k_t, out_dtype=out.dtype)

    # Scale
    scale = q.shape[-1] ** -0.5
    scores_scaled = pypto.mul(scores, scale)

    # Softmax
    attn_weights = pypto.softmax(scores_scaled, dim=-1)

    # Apply to values
    out[:] = pypto.matmul(attn_weights, v, out_dtype=out.dtype)


@pypto.options(pass_options={"enable_slice": True})
def test_sequential_functions(device_id: int = None, dynamic: bool = False) -> None:
    """Test multiple functions in sequence."""
    print("=" * 60)
    print("Test: Sequential Functions")
    print("=" * 60)

    # Get current device ID (set in main)
    device = f'npu:{device_id}' if global_run_mode == pypto.RunMode.NPU and device_id is not None else 'cpu'

    atol_val = 1e-1

    batch_size, hidden_size = 32, 128

    # Create tensors
    x = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16, device=device)
    gamma = torch.ones(hidden_size, dtype=torch.bfloat16, device=device)
    beta = torch.zeros(hidden_size, dtype=torch.bfloat16, device=device)

    # Step 1: Layer normalization
    normed = torch.empty(x.shape, dtype=torch.bfloat16, device=device)
    layer_norm_kernel(x, gamma, beta, normed)

    # Step 2: GELU activation
    activated = torch.empty(normed.shape, dtype=torch.bfloat16, device=device)
    gelu_activation_kernel(normed, activated)

    # Verify
    expected_normed = layer_norm_golden(x, gamma, beta, 1e-6)
    expected_activated = gelu_golden(expected_normed)

    max_diff_norm = (normed - expected_normed).abs().max().item()
    max_diff_act = (activated - expected_activated).abs().max().item()

    print(f"Input shape: {x.shape}")
    if global_run_mode == pypto.RunMode.NPU:
        print(f"Normalized max diff: {max_diff_norm:.6f}")
        print(f"Activated max diff: {max_diff_act:.6f}")
        assert max_diff_norm < atol_val, "Layer norm mismatch!"
        assert max_diff_act < atol_val, "GELU mismatch!"
    print("✓ Sequential functions passed")
    print()


@pypto.options(pass_options={"enable_slice": True})
def test_residual_connection(device_id: int = None, dynamic: bool = False) -> None:
    """Test residual connection pattern."""
    print("=" * 60)
    print("Test: Residual Connection")
    print("=" * 60)

    # Get current device ID (set in main)
    device = f'npu:{device_id}' if global_run_mode == pypto.RunMode.NPU and device_id is not None else 'cpu'

    batch_size, hidden_size = 32, 128

    # Create tensors
    x = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16, device=device)
    residual = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16, device=device)

    # Apply residual connection
    out = torch.empty(x.shape, dtype=torch.bfloat16, device=device)
    residual_add_kernel(x, residual, out)

    # Verify
    expected = x + residual
    max_diff = (out - expected).abs().max().item()

    print(f"Input shape: {x.shape}")
    print(f"Residual shape: {residual.shape}")
    print(f"Output shape: {out.shape}")
    if global_run_mode == pypto.RunMode.NPU:
        print(f"Max difference: {max_diff:.6f}")
        assert max_diff < 1e-2, "Residual connection mismatch!"
    print("✓ Residual connection passed")
    print()


@pypto.options(pass_options={"enable_slice": True})
def test_transformer_block(device_id: int = None, dynamic: bool = False) -> None:
    """Test a complete transformer block using multiple functions."""
    print("=" * 60)
    print("Test: Transformer Block (Multi-Function)")
    print("=" * 60)

    # Get current device ID (set in main)
    device = f'npu:{device_id}' if global_run_mode == pypto.RunMode.NPU and device_id is not None else 'cpu'

    batch_size, hidden_size, intermediate_size = 32, 128, 256

    # Create input
    x = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16, device=device)

    # Layer norm parameters
    gamma = torch.ones(hidden_size, dtype=torch.bfloat16, device=device)
    beta = torch.zeros(hidden_size, dtype=torch.bfloat16, device=device)

    # FFN weights (gate/up: 128->256, down: 256->128)
    gate_weight = torch.randn(hidden_size, intermediate_size, dtype=torch.bfloat16, device=device)
    up_weight = torch.randn(hidden_size, intermediate_size, dtype=torch.bfloat16, device=device)
    down_weight = torch.randn(intermediate_size, hidden_size, dtype=torch.bfloat16, device=device)

    # Intermediate tensors
    normed = torch.zeros(batch_size, hidden_size, dtype=torch.bfloat16, device=device)
    gate = torch.zeros(batch_size, intermediate_size, dtype=torch.bfloat16, device=device)
    up = torch.zeros(batch_size, intermediate_size, dtype=torch.bfloat16, device=device)
    activated = torch.zeros(batch_size, intermediate_size, dtype=torch.bfloat16, device=device)
    ffn_out = torch.zeros(batch_size, hidden_size, dtype=torch.bfloat16, device=device)
    output = torch.zeros(batch_size, hidden_size, dtype=torch.bfloat16, device=device)
    # Transformer block computation:
    # 1. Layer normalization
    normed = torch.empty(x.shape, dtype=torch.bfloat16, device=device)
    layer_norm_kernel(x, gamma, beta, normed)
    if global_run_mode == pypto.RunMode.NPU:
        torch.npu.synchronize()

    # 2. FFN: Gate and Up projections
    linear_projection_kernel(normed, gate_weight, gate)
    if global_run_mode == pypto.RunMode.NPU:
        torch.npu.synchronize()
    linear_projection_kernel(normed, up_weight, up)
    if global_run_mode == pypto.RunMode.NPU:
        torch.npu.synchronize()

    # 3. GELU activation on gate
    activated = torch.empty(gate.shape, dtype=torch.bfloat16, device=device)
    gelu_activation_kernel(gate, activated)
    if global_run_mode == pypto.RunMode.NPU:
        torch.npu.synchronize()

    # 4. Multiply with up (SwiGLU-like)
    activated = activated * up  # PyTorch operation for simplicity

    # 5. Down projection
    linear_projection_kernel(activated, down_weight, ffn_out)

    # 6. Residual connection
    output = torch.empty(x.shape, dtype=torch.bfloat16, device=device)
    residual_add_kernel(x, ffn_out, output)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
    print("✓ Transformer block (multi-function) completed")
    print()


@pypto.options(pass_options={"enable_slice": True})
def test_function_reuse(device_id: int = None, dynamic: bool = True) -> None:
    """Test reusing the same function multiple times."""
    print("=" * 60)
    print("Test: Function Reuse")
    print("=" * 60)

    # Get current device ID (set in main)
    device = f'npu:{device_id}' if global_run_mode == pypto.RunMode.NPU and device_id is not None else 'cpu'

    batch_size, hidden_size = 32, 128

    # Create multiple inputs
    x1 = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16, device=device)
    x2 = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16, device=device)
    x3 = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16, device=device)

    gamma = torch.ones(hidden_size, dtype=torch.bfloat16, device=device)
    beta = torch.zeros(hidden_size, dtype=torch.bfloat16, device=device)

    # Outputs
    out1 = torch.zeros(batch_size, hidden_size, dtype=torch.bfloat16, device=device)
    out2 = torch.zeros(batch_size, hidden_size, dtype=torch.bfloat16, device=device)
    out3 = torch.zeros(batch_size, hidden_size, dtype=torch.bfloat16, device=device)

    # Reuse the same function with different inputs
    layer_norm_kernel(x1, gamma, beta, out1)
    if global_run_mode == pypto.RunMode.NPU:
        torch.npu.synchronize()
    layer_norm_kernel(x2, gamma, beta, out2)
    if global_run_mode == pypto.RunMode.NPU:
        torch.npu.synchronize()
    layer_norm_kernel(x3, gamma, beta, out3)
    if global_run_mode == pypto.RunMode.NPU:
        torch.npu.synchronize()

    # Verify
    expected1 = layer_norm_golden(x1, gamma, beta, 1e-6)
    expected2 = layer_norm_golden(x2, gamma, beta, 1e-6)
    expected3 = layer_norm_golden(x3, gamma, beta, 1e-6)

    max_diff1 = (out1 - expected1).abs().max().item()
    max_diff2 = (out2 - expected2).abs().max().item()
    max_diff3 = (out3 - expected3).abs().max().item()

    print("Function reused 3 times with different inputs")
    if global_run_mode == pypto.RunMode.NPU:
        print(f"Max diff 1: {max_diff1:.6f}")
        print(f"Max diff 2: {max_diff2:.6f}")
        print(f"Max diff 3: {max_diff3:.6f}")
        assert max_diff1 < 1e-1 and max_diff2 < 1e-1 and max_diff3 < 1e-1, "Function reuse mismatch!"
    print("✓ Function reuse passed")
    print()


def main():
    """Run multi-function module examples.

    Usage:
        python multi_function_module.py          # Run all examples
        python multi_function_module.py 1         # Run example 1 only
        python multi_function_module.py --list   # List all available examples
    """
    parser = argparse.ArgumentParser(
        description="PyPTO Multi-Function Module Examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s              Run all examples
  %(prog)s function_reuse::test_function_reuse
            Run example function_reuse::test_function_reuse
  %(prog)s --list       List all available examples
        """,
    )
    parser.add_argument(
        'example_id', type=str, nargs='?', help='Example ID to run (1-4). If not specified, all examples will run.'
    )
    parser.add_argument('--list', action='store_true', help='List all available examples and exit')
    parser.add_argument(
        '--run_mode', type=str, nargs='?', default='npu', choices=["npu", "sim"], help='Run mode, supports npu and sim.'
    )

    args = parser.parse_args()

    # Define available examples
    examples = {
        'sequential_functions::test_sequential_functions': {
            'name': 'Sequential Functions',
            'description': 'Using multiple functions in sequence',
            'function': test_sequential_functions,
            'requires_npu': True,
        },
        'residual_connection::test_residual_connection': {
            'name': 'Residual Connection',
            'description': 'Residual connection pattern',
            'function': test_residual_connection,
            'requires_npu': True,
        },
        'transformer_block::test_transformer_block': {
            'name': 'Transformer Block',
            'description': 'Complete transformer block with multiple functions',
            'function': test_transformer_block,
            'requires_npu': True,
        },
        'function_reuse::test_function_reuse': {
            'name': 'Function Reuse',
            'description': 'Reusing the same function with different inputs',
            'function': test_function_reuse,
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
    print("PyPTO Multi-Function Module Examples")
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
            print("All multi-function module tests passed!")
            print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
