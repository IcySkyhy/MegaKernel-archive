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
Test sigmoid codegen - common functions for Kirin9030 and KirinX90
"""

import numpy as np
import pytest
import torch

from kirin.common import check_nan, compare_cos
import pypto


def make_sigmoid_kernel(soc_version, name, dtype, tile_shapes):
    @pypto.frontend.jit(codegen_options={"soc_version": soc_version}, runtime_options={"run_mode": pypto.RunMode.SIM})
    def kernel(
        a: pypto.Tensor([...], dtype),
        out: pypto.Tensor([...], dtype),
    ):
        pypto.set_vec_tile_shapes(*tile_shapes)
        out[:] = pypto.sigmoid(a)

    kernel.__name__ = name
    return kernel


TEST_CASES = [
    # kernel_name: name of the kernel
    # torch_dtype: torch data type (float16, float32, etc.)
    # pypto_dtype: pypto data type
    # tile_shapes: tile shape for pypto kernel
    # shape: input tensor shape
    # marks: pytest marks
    pytest.param(
        "sigmoid_kernel_fp16_001",
        torch.float16,
        pypto.DT_FP16,
        (50,),
        (112,),
        marks=[pytest.mark.skip()],
        id="001"
    ),
    pytest.param(
        "sigmoid_kernel_fp16_002",
        torch.float16,
        pypto.DT_FP16,
        (100,),
        (100,),
        marks=[],
        id="002"
    ),
    pytest.param(
        "sigmoid_kernel_fp16_003",
        torch.float16,
        pypto.DT_FP16,
        (2, 32),
        (4, 128),
        marks=[pytest.mark.skip()],
        id="003"
    ),
    pytest.param(
        "sigmoid_kernel_fp16_004",
        torch.float16,
        pypto.DT_FP16,
        (1, 130),
        (4, 130),
        marks=[pytest.mark.skip()],
        id="004",
    ),
    pytest.param(
        "sigmoid_kernel_fp16_005",
        torch.float16,
        pypto.DT_FP16,
        (1, 2, 32),
        (2, 4, 160),
        marks=[pytest.mark.skip()],
        id="005",
    ),
    pytest.param(
        "sigmoid_kernel_fp16_006",
        torch.float16,
        pypto.DT_FP16,
        (1, 2, 140),
        (2, 4, 140),
        marks=[pytest.mark.skip()],
        id="006",
    ),
    pytest.param(
        "sigmoid_kernel_fp16_007",
        torch.float16,
        pypto.DT_FP16,
        (1, 5, 32),
        (2, 5, 152),
        marks=[pytest.mark.skip()],
        id="007",
    ),
    pytest.param(
        "sigmoid_kernel_fp16_008",
        torch.float16,
        pypto.DT_FP16,
        (1, 3, 170),
        (2, 3, 170),
        marks=[pytest.mark.skip()],
        id="008",
    ),
    pytest.param(
        "sigmoid_kernel_fp16_009",
        torch.float16,
        pypto.DT_FP16,
        (2, 1, 2, 128),
        (5, 2, 4, 176),
        marks=[pytest.mark.skip()],
        id="009",
    ),
    pytest.param(
        "sigmoid_kernel_fp16_010",
        torch.float16,
        pypto.DT_FP16,
        (1, 1, 1, 130),
        (5, 2, 4, 130),
        marks=[pytest.mark.skip()],
        id="010",
    ),
    pytest.param(
        "sigmoid_kernel_fp16_011",
        torch.float16,
        pypto.DT_FP16,
        (1, 1, 5, 32),
        (2, 3, 5, 134),
        marks=[pytest.mark.skip()],
        id="011",
    ),
    pytest.param(
        "sigmoid_kernel_fp16_012",
        torch.float16,
        pypto.DT_FP16,
        (2, 2, 3, 32),
        (4, 2, 6, 135),
        marks=[pytest.mark.skip()],
        id="012",
    ),
    pytest.param(
        "sigmoid_kernel_fp16_013",
        torch.float16,
        pypto.DT_FP16,
        (1, 1, 4, 130),
        (6, 2, 4, 130),
        marks=[pytest.mark.skip()],
        id="013",
    ),
    pytest.param(
        "sigmoid_kernel_fp16_014",
        torch.float16,
        pypto.DT_FP16,
        (1, 2, 1, 139),
        (3, 2, 3, 139),
        marks=[pytest.mark.skip()],
        id="014",
    ),
    pytest.param(
        "sigmoid_kernel_fp16_015",
        torch.float16,
        pypto.DT_FP16,
        (3, 3, 5, 32),
        (6, 3, 5, 141),
        marks=[pytest.mark.skip()],
        id="015",
    ),
    pytest.param(
        "sigmoid_kernel_fp32_001", torch.float32, pypto.DT_FP32, (50,), (112,), marks=[pytest.mark.skip()], id="016"
    ),
    pytest.param(
        "sigmoid_kernel_fp32_002", torch.float32, pypto.DT_FP32, (100,), (100,), marks=[pytest.mark.skip()], id="017"
    ),
    pytest.param(
        "sigmoid_kernel_fp32_003", torch.float32, pypto.DT_FP32, (2, 32), (4, 128), marks=[pytest.mark.skip()], id="018"
    ),
    pytest.param(
        "sigmoid_kernel_fp32_004",
        torch.float32,
        pypto.DT_FP32,
        (1, 130),
        (4, 130),
        marks=[pytest.mark.skip()],
        id="019",
    ),
    pytest.param(
        "sigmoid_kernel_fp32_005",
        torch.float32,
        pypto.DT_FP32,
        (1, 2, 32),
        (2, 4, 160),
        marks=[pytest.mark.skip()],
        id="020",
    ),
    pytest.param(
        "sigmoid_kernel_fp32_006",
        torch.float32,
        pypto.DT_FP32,
        (1, 2, 140),
        (2, 4, 140),
        marks=[pytest.mark.skip()],
        id="021",
    ),
    pytest.param(
        "sigmoid_kernel_fp32_007",
        torch.float32,
        pypto.DT_FP32,
        (1, 5, 32),
        (2, 5, 152),
        marks=[pytest.mark.skip()],
        id="022",
    ),
    pytest.param(
        "sigmoid_kernel_fp32_008",
        torch.float32,
        pypto.DT_FP32,
        (1, 3, 170),
        (2, 3, 170),
        marks=[pytest.mark.skip()],
        id="023",
    ),
    pytest.param(
        "sigmoid_kernel_fp32_009",
        torch.float32,
        pypto.DT_FP32,
        (2, 1, 2, 128),
        (5, 2, 4, 176),
        marks=[pytest.mark.skip()],
        id="024",
    ),
    pytest.param(
        "sigmoid_kernel_fp32_010",
        torch.float32,
        pypto.DT_FP32,
        (1, 1, 1, 130),
        (5, 2, 4, 130),
        marks=[pytest.mark.skip()],
        id="025",
    ),
    pytest.param(
        "sigmoid_kernel_fp32_011",
        torch.float32,
        pypto.DT_FP32,
        (1, 1, 5, 32),
        (2, 3, 5, 134),
        marks=[pytest.mark.skip()],
        id="026",
    ),
    pytest.param(
        "sigmoid_kernel_fp32_012",
        torch.float32,
        pypto.DT_FP32,
        (2, 2, 3, 32),
        (4, 2, 6, 135),
        marks=[pytest.mark.skip()],
        id="027",
    ),
    pytest.param(
        "sigmoid_kernel_fp32_013",
        torch.float32,
        pypto.DT_FP32,
        (1, 1, 4, 130),
        (6, 2, 4, 130),
        marks=[pytest.mark.skip()],
        id="028",
    ),
    pytest.param(
        "sigmoid_kernel_fp32_014",
        torch.float32,
        pypto.DT_FP32,
        (1, 2, 1, 139),
        (3, 2, 3, 139),
        marks=[pytest.mark.skip()],
        id="029",
    ),
    pytest.param(
        "sigmoid_kernel_fp32_015",
        torch.float32,
        pypto.DT_FP32,
        (3, 3, 5, 32),
        (6, 3, 5, 141),
        marks=[pytest.mark.skip()],
        id="030",
    ),
]


def run_sigmoid_test(kernels, kernel_name, dtype, shape):
    """Run a single sigmoid kernel test with given kernels dict."""
    input_ = torch.rand(shape, dtype=dtype, device="cpu")
    output = torch.rand(shape, dtype=dtype, device="cpu")

    kernels[kernel_name](input_, output)

    expect = torch.sigmoid(input_)
    check_nan(output, name=kernel_name)
    cos_value = abs(compare_cos(np.array(output.cpu()), np.array(expect.cpu())))
    if cos_value < 0.9999:
        raise AssertionError(f"{kernel_name}: cos_value {cos_value} < 0.9999")


def create_sigmoid_kernels(soc_version):
    return {p.values[0]: make_sigmoid_kernel(soc_version, p.values[0], p.values[2], p.values[3]) for p in TEST_CASES}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
