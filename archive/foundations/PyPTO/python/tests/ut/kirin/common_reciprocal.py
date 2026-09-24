#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED.
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Test reciprocal codegen - common functions for Kirin9030 and KirinX90
"""

import numpy as np
import pytest
import torch

from kirin.common import check_nan, compare_cos
import pypto


def make_reciprocal_kernel(soc_version, name, dtype, tile_shapes):
    @pypto.frontend.jit(codegen_options={"soc_version": soc_version}, runtime_options={"run_mode": pypto.RunMode.SIM})
    def kernel(
        a: pypto.Tensor([...], dtype),
        out: pypto.Tensor([...], dtype),
    ):
        pypto.set_vec_tile_shapes(*tile_shapes)
        out[:] = pypto.reciprocal(a)

    kernel.__name__ = name
    return kernel


TEST_CASES = [
    # kernel_name: name of the kernel
    # torch_dtype: torch data type (float16, float32, etc.)
    # pypto_dtype: pypto data type
    # tile_shapes: tile shape for pypto kernel
    # shape: input tensor shape
    # marks: pytest marks
    # - kernel_name: name of the kernel in create_reciprocal_kernels dict
    # - torch_dtype: torch data type (float16, float32, etc.)
    # - pypto_dtype: pypto data type (DT_FP16, DT_FP32, etc.)
    # - tile_shapes: tile shape for pypto kernel
    # - shape: input tensor shape
    pytest.param("reciprocal_kernel_001", torch.float16, pypto.DT_FP16, (50,), (112,), marks=[], id="001"),
    pytest.param(
        "reciprocal_kernel_002", torch.float16, pypto.DT_FP16, (100,), (100,), marks=[pytest.mark.skip()], id="002"
    ),
    pytest.param(
        "reciprocal_kernel_003", torch.float32, pypto.DT_FP32, (2, 32), (4, 128), marks=[pytest.mark.skip()], id="003"
    ),
    pytest.param(
        "reciprocal_kernel_004", torch.float32, pypto.DT_FP32, (1, 130), (4, 130), marks=[pytest.mark.skip()], id="004"
    ),
    pytest.param(
        "reciprocal_kernel_005",
        torch.float16,
        pypto.DT_FP16,
        (1, 2, 32),
        (2, 4, 160),
        marks=[pytest.mark.skip()],
        id="005",
    ),
    pytest.param(
        "reciprocal_kernel_006",
        torch.float32,
        pypto.DT_FP32,
        (1, 2, 140),
        (2, 4, 140),
        marks=[pytest.mark.skip()],
        id="006",
    ),
    pytest.param(
        "reciprocal_kernel_007",
        torch.float16,
        pypto.DT_FP16,
        (1, 5, 32),
        (2, 5, 152),
        marks=[pytest.mark.skip()],
        id="007",
    ),
    pytest.param(
        "reciprocal_kernel_008",
        torch.float32,
        pypto.DT_FP32,
        (1, 3, 170),
        (2, 3, 170),
        marks=[pytest.mark.skip()],
        id="008",
    ),
    pytest.param(
        "reciprocal_kernel_009",
        torch.float16,
        pypto.DT_FP16,
        (2, 1, 2, 16),
        (5, 2, 4, 176),
        marks=[pytest.mark.skip()],
        id="009",
    ),
    pytest.param(
        "reciprocal_kernel_010",
        torch.float32,
        pypto.DT_FP32,
        (1, 1, 1, 130),
        (5, 2, 4, 130),
        marks=[pytest.mark.skip()],
        id="010",
    ),
    pytest.param(
        "reciprocal_kernel_011",
        torch.float16,
        pypto.DT_FP16,
        (1, 1, 5, 32),
        (2, 3, 5, 134),
        marks=[pytest.mark.skip()],
        id="011",
    ),
    pytest.param(
        "reciprocal_kernel_012",
        torch.float32,
        pypto.DT_FP32,
        (2, 2, 3, 32),
        (4, 2, 6, 135),
        marks=[pytest.mark.skip()],
        id="012",
    ),
    pytest.param(
        "reciprocal_kernel_013",
        torch.float16,
        pypto.DT_FP16,
        (1, 1, 4, 130),
        (6, 2, 4, 130),
        marks=[pytest.mark.skip()],
        id="013",
    ),
    pytest.param(
        "reciprocal_kernel_014",
        torch.float32,
        pypto.DT_FP32,
        (1, 2, 1, 139),
        (3, 2, 3, 139),
        marks=[pytest.mark.skip()],
        id="014",
    ),
    pytest.param(
        "reciprocal_kernel_015",
        torch.float16,
        pypto.DT_FP16,
        (3, 3, 5, 32),
        (6, 3, 5, 141),
        marks=[pytest.mark.skip()],
        id="015",
    ),
]


def run_reciprocal_test(kernels, kernel_name, dtype, shape):
    """Run a single reciprocal kernel test with given kernels dict."""
    device = "cpu"

    a = 0.1 + 0.9 * torch.rand(shape, dtype=dtype, device=device)
    out = torch.empty(shape, dtype=dtype, device=device)

    kernels[kernel_name](a, out)

    golden_out = torch.reciprocal(a)
    check_nan(out, name=kernel_name)
    cos_value = abs(compare_cos(np.array(out.cpu()), np.array(golden_out.cpu())))
    if cos_value < 0.9999:
        raise AssertionError(f"{kernel_name}: cos_value {cos_value} < 0.9999")


def create_reciprocal_kernels(soc_version):
    return {p.values[0]: make_reciprocal_kernel(soc_version, p.values[0], p.values[2], p.values[3]) for p in TEST_CASES}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
