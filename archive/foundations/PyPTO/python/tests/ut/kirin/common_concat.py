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
Test concat codegen - common functions for Kirin9030 and KirinX90
"""

import numpy as np
import pytest
import torch

from kirin.common import check_nan, compare_cos
import pypto


def make_concat_kernel_2(soc_version, name, dtype, tile_shapes):
    @pypto.frontend.jit(codegen_options={"soc_version": soc_version}, runtime_options={"run_mode": pypto.RunMode.SIM})
    def kernel(
        input0: pypto.Tensor([...], dtype),
        input1: pypto.Tensor([...], dtype),
        output: pypto.Tensor([...], dtype),
        dim: pypto.DT_INT8,
    ):
        pypto.set_vec_tile_shapes(*tile_shapes)
        output.move(pypto.concat([input0, input1], dim))

    kernel.__name__ = name
    return kernel


def make_concat_kernel_3(soc_version, name, dtype, tile_shapes):
    @pypto.frontend.jit(codegen_options={"soc_version": soc_version}, runtime_options={"run_mode": pypto.RunMode.SIM})
    def kernel(
        input0: pypto.Tensor([...], dtype),
        input1: pypto.Tensor([...], dtype),
        input2: pypto.Tensor([...], dtype),
        output: pypto.Tensor([...], dtype),
        dim: pypto.DT_INT8,
    ):
        pypto.set_vec_tile_shapes(*tile_shapes)
        output.move(pypto.concat([input0, input1, input2], dim))

    kernel.__name__ = name
    return kernel


def make_concat_kernel_4(soc_version, name, dtype, tile_shapes):
    @pypto.frontend.jit(codegen_options={"soc_version": soc_version}, runtime_options={"run_mode": pypto.RunMode.SIM})
    def kernel(
        input0: pypto.Tensor([...], dtype),
        input1: pypto.Tensor([...], dtype),
        input2: pypto.Tensor([...], dtype),
        input3: pypto.Tensor([...], dtype),
        output: pypto.Tensor([...], dtype),
        dim: pypto.DT_INT8,
    ):
        pypto.set_vec_tile_shapes(*tile_shapes)
        output.move(pypto.concat([input0, input1, input2, input3], dim))

    kernel.__name__ = name
    return kernel


TEST_CASES = [
    # kernel_name: name of the kernel
    # torch_dtype: torch data type (int8, int16, int32, float16, float32)
    # pypto_dtype: pypto data type
    # tile_shapes: tile shape for pypto kernel
    # input_shapes: list of input tensor shapes
    # output_shape: output tensor shape (concatenation result)
    # dim: dimension along which to concatenate
    # marks: pytest marks
    # - kernel_name: name of the kernel in create_concat_kernels dict
    # - torch_dtype: torch data type (int8, int16, int32, float16, float32)
    # - pypto_dtype: pypto data type (DT_FP16, DT_FP32, DT_INT8, etc.)
    # - tile_shapes: tile shape for pypto kernel
    # - input_shapes: list of input tensor shapes
    # - output_shape: output tensor shape (concatenation result)
    # - dim: dimension along which to concatenate
    pytest.param(
        "concat_kernel_001",
        torch.int16,
        pypto.DT_INT16,
        (8, 256),
        [(4, 128), (7, 128)],
        (11, 128),
        0,
        marks=[],
        id="001",
    ),
    pytest.param(
        "concat_kernel_002",
        torch.int32,
        pypto.DT_INT32,
        (10, 200),
        [(4, 130), (4, 90)],
        (4, 220),
        1,
        marks=[pytest.mark.skip()],
        id="002",
    ),
    pytest.param(
        "concat_kernel_003",
        torch.float16,
        pypto.DT_FP16,
        (5, 32),
        [(15, 31), (20, 31), (16, 31)],
        (51, 31),
        0,
        marks=[pytest.mark.skip()],
        id="003",
    ),
    pytest.param(
        "concat_kernel_004",
        torch.float32,
        pypto.DT_FP32,
        (2, 280),
        [(4, 140), (4, 23), (4, 4)],
        (4, 167),
        1,
        marks=[pytest.mark.skip()],
        id="004",
    ),
    pytest.param(
        "concat_kernel_005",
        torch.int8,
        pypto.DT_INT8,
        (5, 5, 32),
        [(10, 5, 12), (5, 5, 12)],
        (15, 5, 12),
        0,
        marks=[pytest.mark.skip()],
        id="005",
    ),
    pytest.param(
        "concat_kernel_006",
        torch.int16,
        pypto.DT_INT16,
        (5, 5, 400),
        [(7, 3, 170), (7, 20, 170)],
        (7, 23, 170),
        1,
        marks=[pytest.mark.skip()],
        id="006",
    ),
    pytest.param(
        "concat_kernel_007",
        torch.int32,
        pypto.DT_INT32,
        (5, 4, 120),
        [(9, 8, 100), (9, 8, 40)],
        (9, 8, 140),
        2,
        marks=[pytest.mark.skip()],
        id="007",
    ),
    pytest.param(
        "concat_kernel_008",
        torch.float16,
        pypto.DT_FP16,
        (10, 10, 16),
        [(20, 40, 10), (9, 40, 10), (12, 40, 10)],
        (41, 40, 10),
        0,
        marks=[pytest.mark.skip()],
        id="008",
    ),
    pytest.param(
        "concat_kernel_009",
        torch.float32,
        pypto.DT_FP32,
        (16, 5, 5, 16),
        [(32, 3, 5, 14), (21, 3, 5, 14)],
        (53, 3, 5, 14),
        0,
        marks=[pytest.mark.skip()],
        id="009",
    ),
    pytest.param(
        "concat_kernel_010",
        torch.int8,
        pypto.DT_INT8,
        (2, 10, 9, 32),
        [(8, 10, 6, 16), (8, 4, 6, 16)],
        (8, 14, 6, 16),
        1,
        marks=[pytest.mark.skip()],
        id="010",
    ),
    pytest.param(
        "concat_kernel_011",
        torch.int16,
        pypto.DT_INT16,
        (3, 40, 4, 16),
        [(6, 20, 9, 31), (6, 20, 23, 31)],
        (6, 20, 32, 31),
        2,
        marks=[pytest.mark.skip()],
        id="011",
    ),
    pytest.param(
        "concat_kernel_012",
        torch.int32,
        pypto.DT_INT32,
        (3, 3, 30, 8),
        [(6, 9, 21, 10), (6, 9, 21, 13)],
        (6, 9, 21, 23),
        3,
        marks=[pytest.mark.skip()],
        id="012",
    ),
    pytest.param(
        "concat_kernel_013",
        torch.float16,
        pypto.DT_FP16,
        (5, 10, 5, 16),
        [(6, 9, 21, 10), (9, 9, 21, 10), (17, 9, 21, 10)],
        (32, 9, 21, 10),
        0,
        marks=[pytest.mark.skip()],
        id="013",
    ),
    pytest.param(
        "concat_kernel_014",
        torch.float32,
        pypto.DT_FP32,
        (3, 3, 40, 8),
        [(6, 9, 21, 10), (6, 21, 21, 10), (6, 16, 21, 10)],
        (6, 46, 21, 10),
        1,
        marks=[pytest.mark.skip()],
        id="014",
    ),
    pytest.param(
        "concat_kernel_015",
        torch.int8,
        pypto.DT_INT8,
        (5, 5, 12, 32),
        [(6, 9, 21, 10), (6, 9, 14, 10), (6, 9, 19, 10)],
        (6, 9, 54, 10),
        2,
        marks=[pytest.mark.skip()],
        id="015",
    ),
    pytest.param(
        "concat_kernel_016",
        torch.int16,
        pypto.DT_INT16,
        (5, 8, 12, 16),
        [(6, 9, 21, 10), (6, 9, 21, 40), (6, 9, 21, 21), (6, 9, 21, 9)],
        (6, 9, 21, 80),
        3,
        marks=[pytest.mark.skip()],
        id="016",
    ),
]


def run_concat_test(kernels, kernel_name, dtype, input_shapes, output_shape, dim):
    """Run a single concat kernel test with given kernels dict."""
    device = "cpu"

    inputs = [
        torch.randint(-100, 100, shape, dtype=dtype, device=device)
        if dtype in [torch.int8, torch.int16, torch.int32]
        else torch.randn(shape, dtype=dtype, device=device)
        for shape in input_shapes
    ]
    output = torch.empty(output_shape, dtype=dtype, device=device)

    args = list(inputs) + [output, dim]
    kernels[kernel_name](*args)

    golden = torch.cat(inputs, dim=dim)
    check_nan(output, name=kernel_name)
    cos_value = abs(compare_cos(np.array(output.cpu()), np.array(golden.cpu())))
    if cos_value < 0.9999:
        raise AssertionError(f"{kernel_name}: cos_value {cos_value} < 0.9999")


def create_test_concat_module(soc_version):
    """Create a test module for concat with specified soc_version."""

    def _make_kernel(p):
        kernel_name = p.values[0]
        dtype = p.values[2]
        tile_shapes = p.values[3]
        input_shapes = p.values[4]
        num_inputs = len(input_shapes)
        if num_inputs == 2:
            return make_concat_kernel_2(soc_version, kernel_name, dtype, tile_shapes)
        elif num_inputs == 3:
            return make_concat_kernel_3(soc_version, kernel_name, dtype, tile_shapes)
        elif num_inputs == 4:
            return make_concat_kernel_4(soc_version, kernel_name, dtype, tile_shapes)
        else:
            raise ValueError(f"Unsupported number of inputs: {num_inputs}")

    kernels = {p.values[0]: _make_kernel(p) for p in TEST_CASES}
    return kernels, lambda: run_concat_test(kernels, None, None, None, None, None)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
