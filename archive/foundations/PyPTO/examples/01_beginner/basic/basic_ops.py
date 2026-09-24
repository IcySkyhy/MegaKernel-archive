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
Basic Operations Quick-Start for PyPTO
"""

import argparse
import os
import sys

import torch

import pypto

runtime_options = {"run_mode": pypto.RunMode.NPU}


@pypto.frontend.jit(runtime_options=runtime_options)
def add_kernel(
    a: pypto.Tensor[[...], pypto.DT_FP16],
    b: pypto.Tensor[[...], pypto.DT_FP16],
    out: pypto.Tensor[[...], pypto.DT_FP16],
):
    pypto.set_vec_tile_shapes(32, 32)
    out[:] = (a + b) * 2.0


@pypto.options(pass_options={"enable_slice": True})
def test_add(device):
    shape = (64, 64)

    a = torch.randn(shape, dtype=torch.float16, device=device)
    b = torch.randn(shape, dtype=torch.float16, device=device)
    out = torch.zeros(shape, dtype=torch.float16, device=device)

    add_kernel(a, b, out)
    torch.testing.assert_close(out, (a + b) * 2.0, atol=1e-3, rtol=1e-3)


@pypto.frontend.jit(runtime_options=runtime_options)
def erfc_kernel(x: pypto.Tensor[[...], pypto.DT_FP32], out: pypto.Tensor[[...], pypto.DT_FP32]):
    pypto.set_vec_tile_shapes(32, 32)
    out[:] = pypto.erfc(x)


@pypto.options(pass_options={"enable_slice": True})
def test_erfc(device):
    shape = (64, 64)

    x = torch.randn(shape, dtype=torch.float32, device=device)
    out = torch.zeros(shape, dtype=torch.float32, device=device)

    erfc_kernel(x, out)
    torch.testing.assert_close(out, torch.erfc(x), atol=1e-3, rtol=1e-3)


@pypto.frontend.jit(runtime_options=runtime_options)
def matmul_kernel(
    a: pypto.Tensor[[...], pypto.DT_BF16],
    b: pypto.Tensor[[...], pypto.DT_BF16],
    out: pypto.Tensor[[...], pypto.DT_BF16],
):
    pypto.set_cube_tile_shapes([32, 32], [64, 64], [64, 64])
    out.move(pypto.matmul(a, b, a.dtype))


@pypto.options(pass_options={"enable_slice": True})
def test_matmul(device):
    m, k, n = 64, 128, 64

    a = torch.randn(m, k, dtype=torch.bfloat16, device=device)
    b = torch.randn(k, n, dtype=torch.bfloat16, device=device)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=device)
    matmul_kernel(a, b, out)

    torch.testing.assert_close(out, torch.matmul(a, b), atol=1e-3, rtol=1e-3)


@pypto.frontend.jit(runtime_options=runtime_options)
def sum_kernel(a: pypto.Tensor[[...], pypto.DT_FP32], out: pypto.Tensor[[...], pypto.DT_FP32]):
    pypto.set_vec_tile_shapes(8, 8)
    out[:] = pypto.sum(a, dim=-1, keepdim=False)


@pypto.options(pass_options={"enable_slice": True})
def test_sum(device):
    a = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device=device)
    out = torch.empty((2), dtype=torch.float32, device=device)

    sum_kernel(a, out)

    torch.testing.assert_close(out, torch.sum(a, dim=-1), atol=1e-3, rtol=1e-3)


def ceildiv(x, y):
    return (x + y - 1) // y


@pypto.frontend.jit(runtime_options=runtime_options)
def dynamic_add_kernel(
    # `pypto.DYNAMIC` marks a dynamic dimension support. Static and dynamic dimensions can be mixed.
    x: pypto.Tensor[[pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP16],
    output: pypto.Tensor[[pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP16],
    # Logical block processed by each loop iteration.
    block_m: int,
    block_n: int,
    # Hardware compute tile.
    tile_m: int,
    tile_n: int,
):
    pypto.set_vec_tile_shapes(tile_m, tile_n)

    # `pypto.loop` generates loops for dynamic iteration counts.
    # It is also recommended for large static loops to reduce compile time.
    # `break` and `continue` are not supported.
    for m in pypto.loop(ceildiv(x.shape[0], block_m)):
        for n in pypto.loop(ceildiv(x.shape[1], block_n)):
            # Pypto computes on fixed-size blocks. `view` creates a logical BLOCK_M × BLOCK_N block,
            # while automatically tracking the valid region for boundary blocks.
            #
            # `shape` and `valid_shape` are symbolic compile-time values and can be inspected during compilation.
            # eg: `print(tile.shape)`, `print(tile.valid_shape)`
            tile = pypto.view(x, shape=[block_m, block_n], offsets=[m * block_m, n * block_n])
            tile = tile * 2
            pypto.assemble(tile, [m * block_m, n * block_n], output)


@pypto.options(pass_options={"enable_slice": True})
def test_dynamic_add(device):
    m, n = 512, 512
    block_m, block_n = 128, 128
    tile_m, tile_n = 32, 32

    x = torch.randn((m, n), dtype=torch.float16, device=device)
    out = torch.empty((m, n), dtype=torch.float16, device=device)

    dynamic_add_kernel(x, out, block_m, block_n, tile_m, tile_n)
    if "npu" in device:
        torch.testing.assert_close(out, x * 2.0, atol=1e-2, rtol=1e-2)
    else:
        print("Warning: dynamic_add_kernel is not supported in sim mode, skip verification")


def device_init(run_mode):
    if run_mode == "sim":
        runtime_options["run_mode"] = pypto.RunMode.SIM
        return "cpu"
    else:
        try:
            import torch_npu  # noqa: F401
        except ImportError:
            print("torch_npu is not installed, please install it first")
            sys.exit(1)

        device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
        torch.npu.set_device(device_id)

        runtime_options["run_mode"] = pypto.RunMode.NPU
        return f"npu:{device_id}"


def main():
    examples = {
        "add": test_add,
        "erfc": test_erfc,
        "matmul": test_matmul,
        "sum": test_sum,
        "dynamic_add": test_dynamic_add,
    }

    parser = argparse.ArgumentParser(description="PyPTO Basic Operations Quick-Start")
    parser.add_argument(
        "-m",
        "--run_mode",
        choices=["npu", "sim"],
        default="npu",
        help="Execution mode (default: npu)",
    )
    parser.add_argument(
        "-t",
        "--tests",
        nargs="*",
        choices=examples.keys(),
        metavar="TEST",
        help="Test cases to run (default: all). Choices: %(choices)s",
    )
    args = parser.parse_args()

    if args.tests:
        selected = {test: examples[test] for test in args.tests}
    else:
        selected = examples

    device = device_init(args.run_mode)
    for name, test in selected.items():
        print(f"Running test_{name} ...")
        test(device)
        print(f"✓ Test {name} completed successfully")


if __name__ == "__main__":
    main()
