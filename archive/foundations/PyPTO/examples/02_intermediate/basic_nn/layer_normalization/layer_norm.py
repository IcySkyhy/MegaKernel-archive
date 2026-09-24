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
Layer Normalization Example for PyPTO

This example demonstrates:
- Standard Layer Normalization (LayerNorm)
- RMS Normalization (RMSNorm)
- Static and dynamic batch size support
- Residual connections

Layer normalization is a key component in transformer architectures.
"""

import argparse
from dataclasses import dataclass
import os
import sys
from typing import Literal

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
class NormConfig:
    """Configuration for normalization operations."""

    norm_type: Literal["layernorm", "rmsnorm"] = "layernorm"
    eps: float = 1e-6
    dtype: pypto.DataType = pypto.DT_BF16
    use_dynamic_shape: bool = False


def layernorm_golden(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float) -> torch.Tensor:
    """PyTorch reference implementation of LayerNorm."""
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    normalized = (x - mean) / torch.sqrt(var + eps)
    return normalized * gamma + beta


def layernorm_core(
    x: pypto.Tensor, gamma: pypto.Tensor, beta: pypto.Tensor, eps: float, hidden_size: int
) -> pypto.Tensor:
    # Compute mean
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
def layer_norm_kernel(
    x: pypto.Tensor(), gamma: pypto.Tensor(), beta: pypto.Tensor(), output: pypto.Tensor(), config: NormConfig
):
    hidden_size = x.shape[1]
    eps = config.eps
    pypto.set_vec_tile_shapes(64, 128)
    out = layernorm_core(x, gamma, beta, eps, hidden_size)
    pypto.assemble(out, [0, 0], output)


@pypto.options(pass_options={"enable_slice": True})
def test_layer_norm(device_id=None, dynamic: bool = False):
    """Test LayerNorm."""
    print("=" * 60)
    print("Test: LayerNorm")
    print("=" * 60)

    device = f'npu:{device_id}' if global_run_mode == pypto.RunMode.NPU and device_id is not None else 'cpu'

    batch_size, hidden_size = 32, 128
    shape = (batch_size, hidden_size)

    x_torch = torch.randn(shape, dtype=torch.bfloat16, device=device)
    gamma_torch = torch.ones(hidden_size, dtype=torch.bfloat16, device=device)
    beta_torch = torch.zeros(hidden_size, dtype=torch.bfloat16, device=device)
    config = NormConfig(norm_type="layernorm", dtype=pypto.DT_BF16)
    out_torch = torch.empty(shape, dtype=torch.bfloat16, device=device)
    layer_norm_kernel(x_torch, gamma_torch, beta_torch, out_torch, config)

    expected = layernorm_golden(x_torch, gamma_torch, beta_torch, config.eps)
    max_diff = (out_torch - expected).abs().max().item()

    print(f"Input shape: {x_torch.shape}")
    print(f"Output shape: {out_torch.shape}")
    print(f"Max difference: {max_diff:.6f}")
    if global_run_mode == pypto.RunMode.NPU:
        assert max_diff < 1e-1, "Result mismatch!"
    print("✓ LayerNorm passed")
    print()


def rmsnorm_golden(x: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    """PyTorch reference implementation of RMSNorm."""
    rms = torch.sqrt((x**2).mean(dim=-1, keepdim=True) + eps)
    return (x / rms) * gamma


def rms_norm_core(x: pypto.Tensor, gamma: pypto.Tensor, eps: float, hidden_size: float) -> pypto.Tensor:
    # Compute RMS: sqrt(mean(x^2) + eps)
    squared = x * x
    mean_sq = pypto.sum(squared, dim=-1, keepdim=True)
    mean_sq = mean_sq / hidden_size
    rms = pypto.sqrt((mean_sq + eps))
    normalized = x / rms
    return normalized * gamma


@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode})
def rms_norm_kernel(x: pypto.Tensor(), gamma: pypto.Tensor(), output: pypto.Tensor(), config: NormConfig):
    hidden_size = x.shape[1]
    eps = config.eps
    pypto.set_vec_tile_shapes(64, 128)
    out = rms_norm_core(x, gamma, eps, hidden_size)
    pypto.assemble(out, [0, 0], output)


@pypto.options(pass_options={"enable_slice": True})
def test_rms_norm(device_id=None, dynamic: bool = False) -> None:
    """Test RMSNorm."""
    print("=" * 60)
    print("Test: RMSNorm")
    print("=" * 60)

    device = f'npu:{device_id}' if global_run_mode == pypto.RunMode.NPU and device_id is not None else 'cpu'

    batch_size, hidden_size = 32, 128
    shape = (batch_size, hidden_size)

    x_torch = torch.randn(shape, dtype=torch.bfloat16, device=device)
    gamma_torch = torch.ones(hidden_size, dtype=torch.bfloat16, device=device)
    config = NormConfig(norm_type="rmsnorm", dtype=pypto.DT_BF16)
    out_torch = torch.empty(shape, dtype=torch.bfloat16, device=device)

    rms_norm_kernel(x_torch, gamma_torch, out_torch, config)

    expected = rmsnorm_golden(x_torch, gamma_torch, config.eps)
    max_diff = (out_torch - expected).abs().max().item()

    print(f"Input shape: {x_torch.shape}")
    print(f"Output shape: {out_torch.shape}")
    print(f"Max difference: {max_diff:.6f}")
    if global_run_mode == pypto.RunMode.NPU:
        assert max_diff < 1e-1, "Result mismatch!"
    print("✓ RMSNorm passed")
    print()


def main():
    """Run layer normalization examples.

    Usage:
        python layer_norm.py          # Run all examples
        python layer_norm.py 1         # Run example 1 only
        python layer_norm.py --list   # List all available examples
    """
    parser = argparse.ArgumentParser(
        description="PyPTO Layer Normalization Examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s              Run all examples
  %(prog)s layer_norm::test_layer_norm
            Run example layer_norm::test_layer_norm
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

    examples = {
        'layer_norm::test_layer_norm': {
            'name': 'LayerNorm',
            'description': 'Standard Layer Normalization',
            'function': test_layer_norm,
        },
        'rms_norm::test_rms_norm': {'name': 'RMSNorm', 'description': 'RMS Normalization', 'function': test_rms_norm},
    }

    if args.list:
        print("\n" + "=" * 60)
        print("Available Examples")
        print("=" * 60 + "\n")
        for ex_id, ex_info in sorted(examples.items()):
            print(f"  ID: {ex_id}")
            print(f"    name: {ex_info['name']}")
            print(f"    description: {ex_info['description']}\n")
        return

    if args.example_id is not None:
        if args.example_id not in examples:
            print(f"ERROR: Invalid example ID: {args.example_id}")
            print(f"Valid example IDs are: {', '.join(map(str, sorted(examples.keys())))}")
            print("\nUse --list to see all available examples.")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("PyPTO Layer Normalization Examples")
    print("=" * 60 + "\n")

    device_id = None
    examples_to_run = []

    if args.example_id is not None:
        example = examples.get(args.example_id)
        if example is None:
            raise ValueError(f"Invalid example ID: {args.example_id}")
        examples_to_run = [(args.example_id, example)]
    else:
        examples_to_run = list(examples.items())

    if args.run_mode == "npu":
        device_id = get_device_id()
        if device_id is None:
            return
        torch.npu.set_device(device_id)
        print("Running examples that require NPU hardware...")
        print("Make sure CANN environment is configured and NPU is available\n")

    try:
        for ex_id, ex_info in examples_to_run:
            print(f"Running Example {ex_id}: {ex_info['name']}")
            ex_info['function'](device_id)

        if len(examples_to_run) > 1:
            print("=" * 60)
            print("All layer normalization tests passed!")
            print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
