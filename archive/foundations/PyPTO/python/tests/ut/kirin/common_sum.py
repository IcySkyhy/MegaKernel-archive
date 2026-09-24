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
Test sum codegen - common functions for Kirin9030 and KirinX90
"""

import numpy as np
import pytest
import torch

from kirin.common import check_nan, compare_cos
import pypto


def make_sum_kernel(soc_version, name, dtype, tile_shapes, dim):
    @pypto.frontend.jit(codegen_options={"soc_version": soc_version}, runtime_options={"run_mode": pypto.RunMode.SIM})
    def kernel(
        input_tensor: pypto.Tensor([...], dtype),
        out_tensor: pypto.Tensor([...], dtype),
    ):
        pypto.set_vec_tile_shapes(*tile_shapes)
        out_tensor[:] = pypto.sum(input_tensor, dim)

    kernel.__name__ = name
    return kernel


TEST_CASES = [
    # kernel_name: name of the kernel
    # torch_dtype: torch data type (float32)
    # pypto_dtype: pypto data type
    # tile_shapes: tile shape for pypto kernel
    # shape_input: input tensor shape
    # shape_out: output tensor shape (result of sum along dim)
    # dim: dimension to sum along (-1, 0, 1, or 2)
    # marks: pytest marks
    # - kernel_name: name of the kernel in create_sum_kernels dict
    # - torch_dtype: torch data type (float32)
    # - pypto_dtype: pypto data type (DT_FP32, etc.)
    # - tile_shapes: tile shape for pypto kernel
    # - shape_input: input tensor shape
    # - shape_out: output tensor shape (result of sum along dim)
    # - dim: dimension to sum along (-1, 0, 1, or 2)
    pytest.param("sum_kernel_fp32_001", torch.float32, pypto.DT_FP32, (48,), (112,), (1,), -1, marks=[], id="001"),
    pytest.param(
        "sum_kernel_fp32_002",
        torch.float32,
        pypto.DT_FP32,
        (96,),
        (100,),
        (1,),
        -1,
        marks=[pytest.mark.skip()],
        id="002",
    ),
    pytest.param(
        "sum_kernel_fp32_003",
        torch.float32,
        pypto.DT_FP32,
        (2, 32),
        (4, 128),
        (4, 1),
        -1,
        marks=[pytest.mark.skip()],
        id="003",
    ),
    pytest.param(
        "sum_kernel_fp32_004",
        torch.float32,
        pypto.DT_FP32,
        (1, 128),
        (4, 130),
        (4, 1),
        -1,
        marks=[pytest.mark.skip()],
        id="004",
    ),
    pytest.param(
        "sum_kernel_fp32_005",
        torch.float32,
        pypto.DT_FP32,
        (1, 2, 32),
        (2, 4, 160),
        (2, 4, 1),
        -1,
        marks=[pytest.mark.skip()],
        id="005",
    ),
    pytest.param(
        "sum_kernel_fp32_006",
        torch.float32,
        pypto.DT_FP32,
        (1, 2, 128),
        (2, 4, 140),
        (2, 4, 1),
        -1,
        marks=[pytest.mark.skip()],
        id="006",
    ),
    pytest.param(
        "sum_kernel_fp32_007",
        torch.float32,
        pypto.DT_FP32,
        (1, 5, 32),
        (2, 5, 152),
        (2, 5, 1),
        -1,
        marks=[pytest.mark.skip()],
        id="007",
    ),
    pytest.param(
        "sum_kernel_fp32_008",
        torch.float32,
        pypto.DT_FP32,
        (1, 3, 168),
        (2, 3, 170),
        (2, 3, 1),
        -1,
        marks=[pytest.mark.skip()],
        id="008",
    ),
    pytest.param(
        "sum_kernel_fp32_009",
        torch.float32,
        pypto.DT_FP32,
        (2, 1, 2, 16),
        (5, 2, 4, 176),
        (5, 2, 4, 1),
        -1,
        marks=[pytest.mark.skip()],
        id="009",
    ),
    pytest.param(
        "sum_kernel_fp32_010",
        torch.float32,
        pypto.DT_FP32,
        (1, 1, 1, 128),
        (5, 2, 4, 130),
        (5, 2, 4, 1),
        -1,
        marks=[pytest.mark.skip()],
        id="010",
    ),
    pytest.param(
        "sum_kernel_fp32_011",
        torch.float32,
        pypto.DT_FP32,
        (1, 1, 5, 32),
        (2, 3, 5, 134),
        (2, 3, 5, 1),
        -1,
        marks=[pytest.mark.skip()],
        id="011",
    ),
    pytest.param(
        "sum_kernel_fp32_012",
        torch.float32,
        pypto.DT_FP32,
        (2, 2, 3, 32),
        (4, 2, 6, 135),
        (4, 2, 6, 1),
        -1,
        marks=[pytest.mark.skip()],
        id="012",
    ),
    pytest.param(
        "sum_kernel_fp32_013",
        torch.float32,
        pypto.DT_FP32,
        (1, 1, 4, 128),
        (6, 2, 4, 130),
        (6, 2, 4, 1),
        -1,
        marks=[pytest.mark.skip()],
        id="013",
    ),
    pytest.param(
        "sum_kernel_fp32_014",
        torch.float32,
        pypto.DT_FP32,
        (1, 2, 1, 136),
        (3, 2, 3, 139),
        (3, 2, 3, 1),
        -1,
        marks=[pytest.mark.skip()],
        id="014",
    ),
    pytest.param(
        "sum_kernel_fp32_015",
        torch.float32,
        pypto.DT_FP32,
        (3, 3, 5, 32),
        (6, 3, 5, 141),
        (6, 3, 5, 1),
        -1,
        marks=[pytest.mark.skip()],
        id="015",
    ),
    pytest.param(
        "sum_kernel_fp32_015_dim0",
        torch.float32,
        pypto.DT_FP32,
        (3, 3, 5, 32),
        (6, 3, 5, 141),
        (1, 3, 5, 141),
        0,
        marks=[pytest.mark.skip()],
        id="016",
    ),
    pytest.param(
        "sum_kernel_fp32_015_dim1",
        torch.float32,
        pypto.DT_FP32,
        (3, 3, 5, 32),
        (6, 3, 5, 141),
        (6, 1, 5, 141),
        1,
        marks=[pytest.mark.skip()],
        id="017",
    ),
    pytest.param(
        "sum_kernel_fp32_015_dim2",
        torch.float32,
        pypto.DT_FP32,
        (3, 3, 5, 32),
        (6, 3, 5, 141),
        (6, 3, 1, 141),
        2,
        marks=[pytest.mark.skip()],
        id="018",
    ),
]


def _compute_golden_sum(input_tensor, dim):
    return torch.sum(input_tensor, dim=dim, keepdim=True)


def run_sum_test(kernels, kernel_name, dtype, shape_input, shape_out, dim):
    """Run a single sum kernel test with given kernels dict."""
    device = "cpu"

    input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
    out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

    kernels[kernel_name](input_tensor, out_tensor)

    golden_out = _compute_golden_sum(input_tensor, dim)

    check_nan(out_tensor, name=kernel_name)
    cos_value = abs(compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu())))
    if cos_value < 0.9999:
        raise AssertionError(f"{kernel_name}: cos_value {cos_value} < 0.9999")


def create_sum_kernels(soc_version):
    return {
        p.values[0]: make_sum_kernel(soc_version, p.values[0], p.values[2], p.values[3], p.values[6])
        for p in TEST_CASES
    }


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
