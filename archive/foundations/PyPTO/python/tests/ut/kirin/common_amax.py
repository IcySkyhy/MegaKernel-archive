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
Test amax codegen - common functions for Kirin9030 and KirinX90
"""

import numpy as np
import pytest
import torch

from kirin.common import check_nan, compare_cos
import pypto


def make_amax_kernel(soc_version, name, dtype, tile_shapes, dim):
    @pypto.frontend.jit(codegen_options={"soc_version": soc_version}, runtime_options={"run_mode": pypto.RunMode.SIM})
    def kernel(
        input_: pypto.Tensor([...], dtype),
        output: pypto.Tensor([...], dtype),
    ):
        pypto.set_vec_tile_shapes(*tile_shapes)
        output[:] = pypto.amax(input_, dim=dim)

    kernel.__name__ = name
    return kernel


TEST_CASES = [
    # kernel_name: name of the kernel
    # torch_dtype: torch data type (float16, float32)
    # pypto_dtype: pypto data type
    # tile_shapes: tile shape for pypto kernel
    # shape: input tensor shape
    # shape_out: output tensor shape (result of amax along dim)
    # dim: dimension to reduce over
    # marks: pytest marks
    pytest.param("amax_kernel_fp16_001", torch.float16, pypto.DT_FP16, (48,), (112,), (1,), -1, marks=[], id="001"),
    pytest.param(
        "amax_kernel_fp16_002",
        torch.float16,
        pypto.DT_FP16,
        (96,),
        (100,),
        (1,),
        -1,
        marks=[pytest.mark.skip()],
        id="002",
    ),
    pytest.param(
        "amax_kernel_fp16_003",
        torch.float16,
        pypto.DT_FP16,
        (2, 32),
        (4, 128),
        (4, 1),
        -1,
        marks=[pytest.mark.skip()],
        id="003",
    ),
    pytest.param(
        "amax_kernel_fp16_004",
        torch.float16,
        pypto.DT_FP16,
        (1, 128),
        (4, 130),
        (4, 1),
        -1,
        marks=[pytest.mark.skip()],
        id="004",
    ),
    pytest.param(
        "amax_kernel_fp16_005",
        torch.float16,
        pypto.DT_FP16,
        (1, 2, 32),
        (2, 4, 160),
        (2, 4, 1),
        -1,
        marks=[pytest.mark.skip()],
        id="005",
    ),
    pytest.param(
        "amax_kernel_fp16_006",
        torch.float16,
        pypto.DT_FP16,
        (1, 2, 128),
        (2, 4, 140),
        (2, 4, 1),
        -1,
        marks=[pytest.mark.skip()],
        id="006",
    ),
    pytest.param(
        "amax_kernel_fp16_007",
        torch.float16,
        pypto.DT_FP16,
        (1, 5, 32),
        (2, 5, 152),
        (2, 5, 1),
        -1,
        marks=[pytest.mark.skip()],
        id="007",
    ),
    pytest.param(
        "amax_kernel_fp16_008",
        torch.float16,
        pypto.DT_FP16,
        (1, 3, 160),
        (2, 3, 170),
        (2, 3, 1),
        -1,
        marks=[pytest.mark.skip()],
        id="008",
    ),
    pytest.param(
        "amax_kernel_fp16_009",
        torch.float16,
        pypto.DT_FP16,
        (2, 1, 2, 16),
        (5, 2, 4, 176),
        (5, 2, 4, 1),
        -1,
        marks=[pytest.mark.skip()],
        id="009",
    ),
    pytest.param(
        "amax_kernel_fp16_010",
        torch.float16,
        pypto.DT_FP16,
        (1, 1, 1, 128),
        (5, 2, 4, 130),
        (5, 2, 4, 1),
        -1,
        marks=[pytest.mark.skip()],
        id="010",
    ),
    pytest.param(
        "amax_kernel_fp16_011",
        torch.float16,
        pypto.DT_FP16,
        (1, 1, 5, 32),
        (2, 3, 5, 134),
        (2, 3, 5, 1),
        -1,
        marks=[pytest.mark.skip()],
        id="011",
    ),
    pytest.param(
        "amax_kernel_fp16_012",
        torch.float16,
        pypto.DT_FP16,
        (2, 2, 3, 32),
        (4, 2, 6, 135),
        (4, 2, 6, 1),
        -1,
        marks=[pytest.mark.skip()],
        id="012",
    ),
    pytest.param(
        "amax_kernel_fp16_013",
        torch.float16,
        pypto.DT_FP16,
        (1, 1, 4, 128),
        (6, 2, 4, 130),
        (6, 2, 4, 1),
        -1,
        marks=[pytest.mark.skip()],
        id="013",
    ),
    pytest.param(
        "amax_kernel_fp16_014",
        torch.float16,
        pypto.DT_FP16,
        (1, 2, 1, 128),
        (3, 2, 3, 139),
        (3, 2, 3, 1),
        -1,
        marks=[pytest.mark.skip()],
        id="014",
    ),
    pytest.param(
        "amax_kernel_fp16_015",
        torch.float16,
        pypto.DT_FP16,
        (3, 3, 5, 32),
        (6, 3, 5, 141),
        (6, 3, 5, 1),
        -1,
        marks=[pytest.mark.skip()],
        id="015",
    ),
    pytest.param(
        "amax_kernel_fp16_015_dim0",
        torch.float16,
        pypto.DT_FP16,
        (3, 3, 5, 32),
        (6, 3, 5, 141),
        (1, 3, 5, 141),
        0,
        marks=[pytest.mark.skip()],
        id="016",
    ),
    pytest.param(
        "amax_kernel_fp16_015_dim1",
        torch.float16,
        pypto.DT_FP16,
        (3, 3, 5, 32),
        (6, 3, 5, 141),
        (6, 1, 5, 141),
        1,
        marks=[pytest.mark.skip()],
        id="017",
    ),
    pytest.param(
        "amax_kernel_fp16_015_dim2",
        torch.float16,
        pypto.DT_FP16,
        (3, 3, 5, 32),
        (6, 3, 5, 141),
        (6, 3, 1, 141),
        2,
        marks=[pytest.mark.skip()],
        id="018",
    ),
    pytest.param(
        "amax_kernel_fp32_001",
        torch.float32,
        pypto.DT_FP32,
        (48,),
        (112,),
        (1,),
        -1,
        marks=[pytest.mark.skip()],
        id="019",
    ),
    pytest.param(
        "amax_kernel_fp32_002",
        torch.float32,
        pypto.DT_FP32,
        (96,),
        (100,),
        (1,),
        -1,
        marks=[pytest.mark.skip()],
        id="020",
    ),
    pytest.param(
        "amax_kernel_fp32_003",
        torch.float32,
        pypto.DT_FP32,
        (2, 32),
        (4, 128),
        (4, 1),
        -1,
        marks=[pytest.mark.skip()],
        id="021",
    ),
    pytest.param(
        "amax_kernel_fp32_004",
        torch.float32,
        pypto.DT_FP32,
        (1, 128),
        (4, 130),
        (4, 1),
        -1,
        marks=[pytest.mark.skip()],
        id="022",
    ),
    pytest.param(
        "amax_kernel_fp32_005",
        torch.float32,
        pypto.DT_FP32,
        (1, 2, 32),
        (2, 4, 160),
        (2, 4, 1),
        -1,
        marks=[pytest.mark.skip()],
        id="023",
    ),
    pytest.param(
        "amax_kernel_fp32_006",
        torch.float32,
        pypto.DT_FP32,
        (1, 2, 128),
        (2, 4, 140),
        (2, 4, 1),
        -1,
        marks=[pytest.mark.skip()],
        id="024",
    ),
    pytest.param(
        "amax_kernel_fp32_007",
        torch.float32,
        pypto.DT_FP32,
        (1, 5, 32),
        (2, 5, 152),
        (2, 5, 1),
        -1,
        marks=[pytest.mark.skip()],
        id="025",
    ),
    pytest.param(
        "amax_kernel_fp32_008",
        torch.float32,
        pypto.DT_FP32,
        (1, 3, 168),
        (2, 3, 170),
        (2, 3, 1),
        -1,
        marks=[pytest.mark.skip()],
        id="026",
    ),
    pytest.param(
        "amax_kernel_fp32_009",
        torch.float32,
        pypto.DT_FP32,
        (2, 1, 2, 16),
        (5, 2, 4, 176),
        (5, 2, 4, 1),
        -1,
        marks=[pytest.mark.skip()],
        id="027",
    ),
    pytest.param(
        "amax_kernel_fp32_010",
        torch.float32,
        pypto.DT_FP32,
        (1, 1, 1, 128),
        (5, 2, 4, 130),
        (5, 2, 4, 1),
        -1,
        marks=[pytest.mark.skip()],
        id="028",
    ),
    pytest.param(
        "amax_kernel_fp32_011",
        torch.float32,
        pypto.DT_FP32,
        (1, 1, 5, 32),
        (2, 3, 5, 134),
        (2, 3, 5, 1),
        -1,
        marks=[pytest.mark.skip()],
        id="029",
    ),
    pytest.param(
        "amax_kernel_fp32_012",
        torch.float32,
        pypto.DT_FP32,
        (2, 2, 3, 32),
        (4, 2, 6, 135),
        (4, 2, 6, 1),
        -1,
        marks=[pytest.mark.skip()],
        id="030",
    ),
    pytest.param(
        "amax_kernel_fp32_013",
        torch.float32,
        pypto.DT_FP32,
        (1, 1, 4, 128),
        (6, 2, 4, 130),
        (6, 2, 4, 1),
        -1,
        marks=[pytest.mark.skip()],
        id="031",
    ),
    pytest.param(
        "amax_kernel_fp32_014",
        torch.float32,
        pypto.DT_FP32,
        (1, 2, 1, 136),
        (3, 2, 3, 139),
        (3, 2, 3, 1),
        -1,
        marks=[pytest.mark.skip()],
        id="032",
    ),
    pytest.param(
        "amax_kernel_fp32_015",
        torch.float32,
        pypto.DT_FP32,
        (3, 3, 5, 32),
        (6, 3, 5, 141),
        (6, 3, 5, 1),
        -1,
        marks=[pytest.mark.skip()],
        id="033",
    ),
]


def run_amax_test(kernels, kernel_name, dtype, input_shape, output_shape, dim):
    """Run a single amax kernel test with given kernels dict."""
    device = "cpu"

    input_ = torch.rand(input_shape, dtype=dtype, device=device)
    output = torch.empty(output_shape, dtype=dtype, device=device)

    kernels[kernel_name](input_, output)

    expect = torch.amax(input_, dim=dim, keepdim=True)
    check_nan(output, name=kernel_name)
    cos_value = abs(compare_cos(np.array(output.cpu()), np.array(expect.cpu())))
    if cos_value < 0.9999:
        raise AssertionError(f"{kernel_name}: cos_value {cos_value} < 0.9999")


def create_test_amax_module(soc_version):
    """Create a test module for amax with specified soc_version."""
    kernels = {
        p.values[0]: make_amax_kernel(soc_version, p.values[0], p.values[2], p.values[3], p.values[6])
        for p in TEST_CASES
    }
    return kernels, lambda: run_amax_test(kernels, None, None, None, None, None)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
