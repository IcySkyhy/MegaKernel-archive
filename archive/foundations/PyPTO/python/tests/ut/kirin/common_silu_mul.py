#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Test silu_mul codegen - common functions for Kirin9030 and KirinX90
"""

import numpy as np
import pytest
import torch

from kirin.common import check_nan, compare_cos
import pypto


def silu_mul_golden(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x) * y


def create_silu_mul_kernels(soc_version):
    """Factory function to create silu_mul kernels with specified soc_version."""

    @pypto.frontend.jit(codegen_options={"soc_version": soc_version}, runtime_options={"run_mode": pypto.RunMode.SIM})
    def silu_mul_fp16(
        x: pypto.Tensor([64, 3072], pypto.DT_FP16),
        y: pypto.Tensor([64, 3072], pypto.DT_FP16),
        output: pypto.Tensor([64, 3072], pypto.DT_FP16),
    ):
        pypto.set_vec_tile_shapes(32, 256)
        sigmoid_x = pypto.sigmoid(x)
        silu_x = pypto.mul(x, sigmoid_x)
        output[:] = pypto.mul(silu_x, y)

    kernels = {"silu_mul_fp16": silu_mul_fp16}
    return kernels


TEST_CASES = [
    # shape: input tensor shape (x and y have same shape)
    # torch_dtype: torch data type (float16)
    # marks: pytest marks
    # - shape: input tensor shape
    # - torch_dtype: torch data type (float16)
    # - marks: pytest marks (e.g., pytest.mark.skip())
    pytest.param(
        (64, 3072),
        torch.float16,
        marks=[],
        id="001",
    ),
]


def run_silu_mul_test(kernels, shape, dtype):
    x = torch.rand(shape, dtype=dtype, device="cpu")
    y = torch.rand(shape, dtype=dtype, device="cpu")
    output = torch.rand(shape, dtype=dtype, device="cpu")

    golden = silu_mul_golden(x, y)

    kernels["silu_mul_fp16"](x, y, output)

    check_nan(output)
    cos_value = abs(compare_cos(np.array(output.cpu()), np.array(golden.cpu())))
    if cos_value < 0.9999:
        raise AssertionError(f"cos_value {cos_value} < 0.9999")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
