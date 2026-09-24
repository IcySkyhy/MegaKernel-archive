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
Hello World Example for PyPTO
"""

import argparse
import os
import sys

import torch

import pypto

runtime_options = {"run_mode": pypto.RunMode.NPU}


@pypto.jit(runtime_options=runtime_options)
def add_kernel(x: pypto.Tensor[...], y: pypto.Tensor[...], out: pypto.Tensor[...]):
    """Simple add kernel, use pypto.Tensor[...] to auto infer shape and dtype."""

    # set vector tile shapes, it'll use by the following `vector` operations,
    # so the rank must match tensor `x` and `y`
    pypto.set_vec_tile_shapes(32, 32)

    # pypto kernel does not support return value, `[:]` is just a syntax sugar to present
    # write to output tensor, can also use `pypto.assemble(x + y, [0, 0], out)`
    out[:] = x + y


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
    parser = argparse.ArgumentParser(description="PyPTO add kernel")
    parser.add_argument(
        "-m",
        "--run_mode",
        choices=["npu", "sim"],
        default="npu",
        help="Execution mode (default: npu)",
    )
    args = parser.parse_args()

    shape = (64, 64)
    device = device_init(args.run_mode)

    x = torch.randn(shape, dtype=torch.float, device=device)
    y = torch.randn(shape, dtype=torch.float, device=device)
    out = torch.empty(shape, dtype=torch.float, device=device)

    add_kernel(x, y, out)

    torch.testing.assert_close(x + y, out, atol=1e-3, rtol=1e-3)
    print("✓ Test add_kernel completed successfully")


if __name__ == "__main__":
    main()
