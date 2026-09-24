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
Test max codegen - common functions for Kirin9030 and KirinX90
"""

import numpy as np
import pytest
import torch

from kirin.common import check_nan, compare_cos
import pypto


def make_max_kernel(soc_version, name, dtype, tile_shapes):
    @pypto.frontend.jit(codegen_options={"soc_version": soc_version}, runtime_options={"run_mode": pypto.RunMode.SIM})
    def kernel(
        a: pypto.Tensor([...], dtype),
        b: pypto.Tensor([...], dtype),
        out: pypto.Tensor([...], dtype),
    ):
        pypto.set_vec_tile_shapes(*tile_shapes)
        out[:] = pypto.maximum(a, b)

    kernel.__name__ = name
    return kernel


TEST_CASES = [
    # kernel_name: name of the kernel
    # torch_dtype: torch data type (int16, int32, float16, float32)
    # pypto_dtype: pypto data type
    # tile_shape: tile shape for pypto kernel
    # shape_a: first input tensor shape
    # shape_b: second input tensor shape (or None)
    # scalar_val: scalar value for max (or None)
    # marks: pytest marks
    pytest.param(
        "max_kernel_int16_001",
        torch.int16,
        pypto.DT_INT16,
        (1, 1, 16, 16),
        (2, 2, 32, 32),
        (2, 2, 32, 32),
        None,
        marks=[],
        id="001",
    ),
    pytest.param(
        "max_kernel_int16_002",
        torch.int16,
        pypto.DT_INT16,
        (1, 2, 8, 8),
        (2, 4, 16, 1),
        (2, 1, 16, 16),
        None,
        marks=[pytest.mark.skip()],
        id="002",
    ),
    pytest.param(
        "max_kernel_int16_003",
        torch.int16,
        pypto.DT_INT16,
        (1, 1, 12, 12),
        (2, 1, 24, 24),
        (2, 3, 24, 24),
        None,
        marks=[pytest.mark.skip()],
        id="003",
    ),
    pytest.param(
        "max_kernel_int16_004",
        torch.int16,
        pypto.DT_INT16,
        (1, 1, 20, 20),
        (2, 2, 40, 40),
        (2, 2, 1, 40),
        None,
        marks=[pytest.mark.skip()],
        id="004",
    ),
    pytest.param(
        "max_kernel_int16_005",
        torch.int16,
        pypto.DT_INT16,
        (1, 1, 8, 8),
        (2, 3, 16, 16),
        (2, 3, 16, 16),
        None,
        marks=[pytest.mark.skip()],
        id="005",
    ),
    pytest.param(
        "max_kernel_int16_006",
        torch.int16,
        pypto.DT_INT16,
        (4, 1),
        (8, 4),
        None,
        50,
        marks=[pytest.mark.skip()],
        id="006",
    ),
    pytest.param(
        "max_kernel_int16_007",
        torch.int16,
        pypto.DT_INT16,
        (2, 1, 32, 32),
        (4, 1, 32, 32),
        (4, 1, 32, 32),
        None,
        marks=[pytest.mark.skip()],
        id="007",
    ),
    pytest.param(
        "max_kernel_int16_008",
        torch.int16,
        pypto.DT_INT16,
        (1, 4, 16, 16),
        (1, 8, 16, 16),
        (1, 8, 16, 16),
        None,
        marks=[pytest.mark.skip()],
        id="008",
    ),
    pytest.param(
        "max_kernel_int16_009",
        torch.int16,
        pypto.DT_INT16,
        (2, 2, 24, 48),
        (2, 2, 48, 48),
        (2, 2, 48, 48),
        None,
        marks=[pytest.mark.skip()],
        id="009",
    ),
    pytest.param(
        "max_kernel_int16_010",
        torch.int16,
        pypto.DT_INT16,
        (1, 4, 32, 32),
        (1, 4, 32, 64),
        (1, 4, 32, 64),
        None,
        marks=[pytest.mark.skip()],
        id="010",
    ),
    pytest.param(
        "max_kernel_int32_001",
        torch.int32,
        pypto.DT_INT32,
        (1, 4, 16, 16),
        (2, 4, 16, 1),
        (2, 1, 16, 16),
        None,
        marks=[pytest.mark.skip()],
        id="011",
    ),
    pytest.param(
        "max_kernel_int32_002",
        torch.int32,
        pypto.DT_INT32,
        (1, 2, 16, 32),
        (2, 1, 32, 32),
        (2, 2, 1, 32),
        None,
        marks=[pytest.mark.skip()],
        id="012",
    ),
    pytest.param(
        "max_kernel_int32_003",
        torch.int32,
        pypto.DT_INT32,
        (1, 3, 24, 24),
        (2, 3, 24, 1),
        (1, 3, 24, 48),
        None,
        marks=[pytest.mark.skip()],
        id="013",
    ),
    pytest.param(
        "max_kernel_int32_004",
        torch.int32,
        pypto.DT_INT32,
        (1, 4, 8, 16),
        (1, 4, 1, 16),
        (1, 1, 16, 16),
        None,
        marks=[pytest.mark.skip()],
        id="014",
    ),
    pytest.param(
        "max_kernel_int32_005",
        torch.int32,
        pypto.DT_INT32,
        (2, 1, 32, 32),
        (2, 2, 32, 1),
        (2, 2, 1, 64),
        None,
        marks=[pytest.mark.skip()],
        id="015",
    ),
    pytest.param(
        "max_kernel_int32_006",
        torch.int32,
        pypto.DT_INT32,
        (1, 1, 24, 32),
        (1, 1, 48, 64),
        (1, 1, 48, 64),
        None,
        marks=[pytest.mark.skip()],
        id="016",
    ),
    pytest.param(
        "max_kernel_int32_007",
        torch.int32,
        pypto.DT_INT32,
        (1, 2, 16, 48),
        (2, 4, 32, 48),
        (2, 4, 32, 48),
        None,
        marks=[pytest.mark.skip()],
        id="017",
    ),
    pytest.param(
        "max_kernel_int32_008",
        torch.int32,
        pypto.DT_INT32,
        (1, 2, 32, 32),
        (2, 4, 32, 64),
        (2, 4, 32, 64),
        None,
        marks=[pytest.mark.skip()],
        id="018",
    ),
    pytest.param(
        "max_kernel_int32_009",
        torch.int32,
        pypto.DT_INT32,
        (2, 2, 16, 32),
        (4, 2, 32, 64),
        (4, 2, 32, 64),
        None,
        marks=[pytest.mark.skip()],
        id="019",
    ),
    pytest.param(
        "max_kernel_int32_010",
        torch.int32,
        pypto.DT_INT32,
        (1, 2, 16, 32),
        (1, 4, 32, 64),
        (1, 4, 32, 64),
        None,
        marks=[pytest.mark.skip()],
        id="020",
    ),
    pytest.param(
        "max_kernel_fp16_001",
        torch.float16,
        pypto.DT_FP16,
        (50,),
        (112,),
        (112,),
        None,
        marks=[pytest.mark.skip()],
        id="021",
    ),
    pytest.param(
        "max_kernel_fp16_002",
        torch.float16,
        pypto.DT_FP16,
        (32,),
        (64,),
        None,
        0.5,
        marks=[pytest.mark.skip()],
        id="022",
    ),
    pytest.param(
        "max_kernel_fp16_003",
        torch.float16,
        pypto.DT_FP16,
        (16,),
        (32,),
        (32,),
        None,
        marks=[pytest.mark.skip()],
        id="023",
    ),
    pytest.param(
        "max_kernel_fp16_004",
        torch.float16,
        pypto.DT_FP16,
        (16, 8),
        (16, 16),
        None,
        0.5,
        marks=[pytest.mark.skip()],
        id="024",
    ),
    pytest.param(
        "max_kernel_fp16_005",
        torch.float16,
        pypto.DT_FP16,
        (2, 40),
        (4, 80),
        (4, 80),
        None,
        marks=[pytest.mark.skip()],
        id="025",
    ),
    pytest.param(
        "max_kernel_fp16_006",
        torch.float16,
        pypto.DT_FP16,
        (1, 48),
        (2, 96),
        (1, 96),
        None,
        marks=[pytest.mark.skip()],
        id="026",
    ),
    pytest.param(
        "max_kernel_fp16_007",
        torch.float16,
        pypto.DT_FP16,
        (2, 16),
        (5, 1),
        (5, 32),
        None,
        marks=[pytest.mark.skip()],
        id="027",
    ),
    pytest.param(
        "max_kernel_fp16_008",
        torch.float16,
        pypto.DT_FP16,
        (2, 64),
        (3, 128),
        (3, 128),
        None,
        marks=[pytest.mark.skip()],
        id="028",
    ),
    pytest.param(
        "max_kernel_fp16_009",
        torch.float16,
        pypto.DT_FP16,
        (32, 64),
        (64, 1),
        (64, 64),
        None,
        marks=[pytest.mark.skip()],
        id="029",
    ),
    pytest.param(
        "max_kernel_fp16_010",
        torch.float16,
        pypto.DT_FP16,
        (64, 32),
        (1, 64),
        (64, 64),
        None,
        marks=[pytest.mark.skip()],
        id="030",
    ),
    pytest.param(
        "max_kernel_fp16_011",
        torch.float16,
        pypto.DT_FP16,
        (1, 32, 32),
        (2, 64, 64),
        (2, 64, 64),
        None,
        marks=[pytest.mark.skip()],
        id="031",
    ),
    pytest.param(
        "max_kernel_fp16_012",
        torch.float16,
        pypto.DT_FP16,
        (1, 1, 24),
        (2, 1, 48),
        (2, 3, 48),
        None,
        marks=[pytest.mark.skip()],
        id="032",
    ),
    pytest.param(
        "max_kernel_fp16_013",
        torch.float16,
        pypto.DT_FP16,
        (1, 32, 24),
        (3, 64, 1),
        (3, 64, 48),
        None,
        marks=[pytest.mark.skip()],
        id="033",
    ),
    pytest.param(
        "max_kernel_fp16_014",
        torch.float16,
        pypto.DT_FP16,
        (1, 48, 48),
        (2, 48, 48),
        (2, 1, 48),
        None,
        marks=[pytest.mark.skip()],
        id="034",
    ),
    pytest.param(
        "max_kernel_fp16_015",
        torch.float16,
        pypto.DT_FP16,
        (2, 32, 48),
        (2, 64, 48),
        (2, 64, 48),
        None,
        marks=[pytest.mark.skip()],
        id="035",
    ),
    pytest.param(
        "max_kernel_fp16_016",
        torch.float16,
        pypto.DT_FP16,
        (3, 32, 32),
        (3, 32, 64),
        (3, 32, 64),
        None,
        marks=[pytest.mark.skip()],
        id="036",
    ),
    pytest.param(
        "max_kernel_fp16_017",
        torch.float16,
        pypto.DT_FP16,
        (1, 16, 64),
        (2, 32, 1),
        (2, 32, 64),
        None,
        marks=[pytest.mark.skip()],
        id="037",
    ),
    pytest.param(
        "max_kernel_fp16_018",
        torch.float16,
        pypto.DT_FP16,
        (16, 16, 16),
        (1, 48, 64),
        (48, 48, 64),
        None,
        marks=[pytest.mark.skip()],
        id="038",
    ),
    pytest.param(
        "max_kernel_fp32_001",
        torch.float32,
        pypto.DT_FP32,
        (16, 16, 8, 8),
        (16, 16, 16, 16),
        None,
        0.5,
        marks=[pytest.mark.skip()],
        id="039",
    ),
    pytest.param(
        "max_kernel_fp32_002",
        torch.float32,
        pypto.DT_FP32,
        (4, 4),
        (8,),
        (8, 8),
        None,
        marks=[pytest.mark.skip()],
        id="040",
    ),
    pytest.param(
        "max_kernel_fp32_003",
        torch.float32,
        pypto.DT_FP32,
        (8, 4, 4),
        (8,),
        (16, 8, 8),
        None,
        marks=[pytest.mark.skip()],
        id="041",
    ),
    pytest.param(
        "max_kernel_fp32_004",
        torch.float32,
        pypto.DT_FP32,
        (2, 8, 4, 4),
        (8,),
        (4, 16, 8, 8),
        None,
        marks=[pytest.mark.skip()],
        id="042",
    ),
    pytest.param(
        "max_kernel_fp32_005",
        torch.float32,
        pypto.DT_FP32,
        (16, 12, 8),
        (24, 16),
        (32, 24, 16),
        None,
        marks=[pytest.mark.skip()],
        id="043",
    ),
    pytest.param(
        "max_kernel_fp32_006",
        torch.float32,
        pypto.DT_FP32,
        (2, 8, 12, 8),
        (24, 16),
        (4, 32, 24, 16),
        None,
        marks=[pytest.mark.skip()],
        id="044",
    ),
    pytest.param(
        "max_kernel_fp32_007",
        torch.float32,
        pypto.DT_FP32,
        (8, 8, 8, 8),
        (32, 32, 16),
        (16, 32, 32, 16),
        None,
        marks=[pytest.mark.skip()],
        id="045",
    ),
    pytest.param(
        "max_kernel_fp32_008",
        torch.float32,
        pypto.DT_FP32,
        (2, 2, 8),
        (1, 1, 16),
        (4, 4, 16),
        None,
        marks=[pytest.mark.skip()],
        id="046",
    ),
    pytest.param(
        "max_kernel_fp32_009",
        torch.float32,
        pypto.DT_FP32,
        (2, 2, 2),
        (1, 1, 1),
        (4, 4, 4),
        None,
        marks=[pytest.mark.skip()],
        id="047",
    ),
    pytest.param(
        "max_kernel_fp32_010",
        torch.float32,
        pypto.DT_FP32,
        (8, 2, 2),
        (16, 1, 1),
        (16, 4, 4),
        None,
        marks=[pytest.mark.skip()],
        id="048",
    ),
    pytest.param(
        "max_kernel_fp32_011",
        torch.float32,
        pypto.DT_FP32,
        (2, 2, 2, 2),
        (1, 1, 1, 1),
        (4, 4, 4, 4),
        None,
        marks=[pytest.mark.skip()],
        id="049",
    ),
    pytest.param(
        "max_kernel_fp32_012",
        torch.float32,
        pypto.DT_FP32,
        (2, 2, 2, 2),
        (1, 1, 4, 4),
        (4, 4, 4, 4),
        None,
        marks=[pytest.mark.skip()],
        id="050",
    ),
    pytest.param(
        "max_kernel_fp32_013",
        torch.float32,
        pypto.DT_FP32,
        (2, 2, 2, 2),
        (1, 4, 1, 4),
        (4, 4, 4, 4),
        None,
        marks=[pytest.mark.skip()],
        id="051",
    ),
    pytest.param(
        "max_kernel_fp32_014",
        torch.float32,
        pypto.DT_FP32,
        (2, 2, 2, 2),
        (1, 4, 4, 1),
        (4, 4, 4, 4),
        None,
        marks=[pytest.mark.skip()],
        id="052",
    ),
    pytest.param(
        "max_kernel_fp32_015",
        torch.float32,
        pypto.DT_FP32,
        (2, 2, 2, 2),
        (4, 1, 1, 4),
        (4, 4, 4, 4),
        None,
        marks=[pytest.mark.skip()],
        id="053",
    ),
    pytest.param(
        "max_kernel_fp32_016",
        torch.float32,
        pypto.DT_FP32,
        (2, 2, 2, 2),
        (4, 1, 4, 1),
        (4, 4, 4, 4),
        None,
        marks=[pytest.mark.skip()],
        id="054",
    ),
    pytest.param(
        "max_kernel_fp32_017",
        torch.float32,
        pypto.DT_FP32,
        (2, 2, 2, 2),
        (4, 4, 1, 1),
        (4, 4, 4, 4),
        None,
        marks=[pytest.mark.skip()],
        id="055",
    ),
]


def run_max_test(kernels, kernel_name, dtype, shape_a, shape_b, scalar_val):
    """Run a single max kernel test with given kernels dict."""
    device = "cpu"
    use_int32_conv = dtype in (torch.int16, torch.int32)

    a = (
        torch.randint(0, 100, shape_a, dtype=dtype, device=device)
        if use_int32_conv
        else torch.rand(shape_a, dtype=dtype, device=device)
    )

    if scalar_val is not None:
        b = torch.full_like(a, scalar_val, dtype=dtype, device=device)
    else:
        b = (
            torch.randint(0, 100, shape_b, dtype=dtype, device=device)
            if use_int32_conv
            else torch.rand(shape_b, dtype=dtype, device=device)
        )

    out_shape = torch.broadcast_shapes(a.shape, b.shape)
    out = torch.zeros(out_shape, dtype=dtype, device=device)

    kernels[kernel_name](a, b, out)

    expect = torch.maximum(a, b)
    out_np = np.array(out.cpu().to(torch.int32) if use_int32_conv else out.cpu())
    expect_np = np.array(expect.cpu().to(torch.int32) if use_int32_conv else expect.cpu())

    check_nan(out, name=kernel_name)
    cos_value = abs(compare_cos(out_np, expect_np))
    if cos_value < 0.9999:
        raise AssertionError(f"{kernel_name}: cos_value {cos_value} < 0.9999")


def create_test_max_module(soc_version):
    """Create a test module for max with specified soc_version."""
    kernels = {p.values[0]: make_max_kernel(soc_version, p.values[0], p.values[2], p.values[3]) for p in TEST_CASES}
    return kernels, lambda: run_max_test(kernels, None, None, None, None, None)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
