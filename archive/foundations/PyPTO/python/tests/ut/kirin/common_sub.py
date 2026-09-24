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
Test sub codegen - common functions for Kirin9030 and KirinX90
"""

import pytest
import torch

from kirin.common import check_nan, compare_cos
import pypto


def make_sub_tensor_kernel(soc_version, name, dtype, tile_shapes):
    @pypto.frontend.jit(codegen_options={"soc_version": soc_version}, runtime_options={"run_mode": pypto.RunMode.SIM})
    def kernel(
        input0: pypto.Tensor([...], dtype),
        input1: pypto.Tensor([...], dtype),
        output: pypto.Tensor([...], dtype),
    ):
        pypto.set_vec_tile_shapes(*tile_shapes)
        output[:] = pypto.sub(input0, input1)

    kernel.__name__ = name
    return kernel


def make_sub_scalar_kernel(soc_version, name, dtype, tile_shapes):
    @pypto.frontend.jit(codegen_options={"soc_version": soc_version}, runtime_options={"run_mode": pypto.RunMode.SIM})
    def kernel(
        input0: pypto.Tensor([...], dtype),
        output: pypto.Tensor([...], dtype),
        input1: dtype,
    ):
        pypto.set_vec_tile_shapes(*tile_shapes)
        output[:] = pypto.sub(input0, input1)

    kernel.__name__ = name
    return kernel


TEST_CASES = [
    # kernel_name: name of the kernel
    # torch_dtype: torch data type (int32, int16, float32, float16)
    # pypto_dtype: pypto data type
    # tile_shape: tile shape for pypto kernel
    # shape_in0: first input tensor shape
    # shape_in1: second input tensor shape (or None)
    # scalar_val: scalar value to subtract (or None)
    # marks: pytest marks
    pytest.param("sub_kernel_001", torch.int32, pypto.DT_INT32, (200,), (160,), (160,), None, marks=[], id="001"),
    pytest.param(
        "sub_kernel_002",
        torch.int16,
        pypto.DT_INT16,
        (100,),
        (100,),
        (100,),
        None,
        marks=[pytest.mark.skip()],
        id="002",
    ),
    pytest.param(
        "sub_kernel_003",
        torch.float32,
        pypto.DT_FP32,
        (100,),
        (112,),
        (112,),
        None,
        marks=[pytest.mark.skip()],
        id="003",
    ),
    pytest.param(
        "sub_kernel_004",
        torch.float16,
        pypto.DT_FP16,
        (100,),
        (101,),
        (101,),
        None,
        marks=[pytest.mark.skip()],
        id="004",
    ),
    pytest.param(
        "sub_kernel_005", torch.float32, pypto.DT_FP32, (100,), (112,), None, 2.0, marks=[pytest.mark.skip()], id="005"
    ),
    pytest.param(
        "sub_kernel_006", torch.float16, pypto.DT_FP16, (100,), (101,), None, 2.0, marks=[pytest.mark.skip()], id="006"
    ),
    pytest.param(
        "sub_kernel_007", torch.int32, pypto.DT_INT32, (120,), (160,), (1,), None, marks=[pytest.mark.skip()], id="007"
    ),
    pytest.param(
        "sub_kernel_008", torch.int16, pypto.DT_INT16, (112,), (100,), (1,), None, marks=[pytest.mark.skip()], id="008"
    ),
    pytest.param(
        "sub_kernel_009",
        torch.float32,
        pypto.DT_FP32,
        (64, 32),
        (32, 20),
        None,
        2.0,
        marks=[pytest.mark.skip()],
        id="009",
    ),
    pytest.param(
        "sub_kernel_010",
        torch.float16,
        pypto.DT_FP16,
        (32, 10),
        (31, 19),
        (31, 19),
        None,
        marks=[pytest.mark.skip()],
        id="010",
    ),
    pytest.param(
        "sub_kernel_011",
        torch.int32,
        pypto.DT_INT32,
        (10, 30),
        (31, 19),
        (31, 19),
        None,
        marks=[pytest.mark.skip()],
        id="011",
    ),
    pytest.param(
        "sub_kernel_012",
        torch.int16,
        pypto.DT_INT16,
        (10, 14),
        (31, 19),
        (31, 19),
        None,
        marks=[pytest.mark.skip()],
        id="012",
    ),
    pytest.param(
        "sub_kernel_013",
        torch.int32,
        pypto.DT_INT32,
        (64, 32),
        (32, 20),
        (1, 20),
        None,
        marks=[pytest.mark.skip()],
        id="013",
    ),
    pytest.param(
        "sub_kernel_014",
        torch.int32,
        pypto.DT_INT32,
        (64, 32),
        (1, 20),
        (32, 20),
        None,
        marks=[pytest.mark.skip()],
        id="014",
    ),
    pytest.param(
        "sub_kernel_015",
        torch.int16,
        pypto.DT_INT16,
        (32, 10),
        (31, 21),
        (31, 1),
        None,
        marks=[pytest.mark.skip()],
        id="015",
    ),
    pytest.param(
        "sub_kernel_016",
        torch.float32,
        pypto.DT_FP32,
        (10, 20),
        (31, 1),
        (31, 19),
        None,
        marks=[pytest.mark.skip()],
        id="016",
    ),
    pytest.param(
        "sub_kernel_017",
        torch.float16,
        pypto.DT_FP16,
        (10, 16),
        (1, 19),
        (31, 1),
        None,
        marks=[pytest.mark.skip()],
        id="017",
    ),
    pytest.param(
        "sub_kernel_018",
        torch.float32,
        pypto.DT_FP32,
        (10, 32, 25),
        (10, 32, 23),
        None,
        2.0,
        marks=[pytest.mark.skip()],
        id="018",
    ),
    pytest.param(
        "sub_kernel_019",
        torch.float16,
        pypto.DT_FP16,
        (10, 32, 20),
        (10, 32, 19),
        None,
        2.0,
        marks=[pytest.mark.skip()],
        id="019",
    ),
    pytest.param(
        "sub_kernel_020",
        torch.int32,
        pypto.DT_INT32,
        (25, 20, 25),
        (21, 19, 23),
        (21, 19, 23),
        None,
        marks=[pytest.mark.skip()],
        id="020",
    ),
    pytest.param(
        "sub_kernel_021",
        torch.int16,
        pypto.DT_INT16,
        (10, 32, 25),
        (10, 32, 23),
        (10, 32, 23),
        None,
        marks=[pytest.mark.skip()],
        id="021",
    ),
    pytest.param(
        "sub_kernel_022",
        torch.int32,
        pypto.DT_INT32,
        (10, 25, 30),
        (1, 23, 27),
        (13, 23, 27),
        None,
        marks=[pytest.mark.skip()],
        id="022",
    ),
    pytest.param(
        "sub_kernel_023",
        torch.float32,
        pypto.DT_FP32,
        (23, 10, 30),
        (13, 1, 27),
        (13, 23, 27),
        None,
        marks=[pytest.mark.skip()],
        id="023",
    ),
    pytest.param(
        "sub_kernel_024",
        torch.int16,
        pypto.DT_INT16,
        (23, 25, 15),
        (13, 23, 1),
        (13, 23, 27),
        None,
        marks=[pytest.mark.skip()],
        id="024",
    ),
    pytest.param(
        "sub_kernel_025",
        torch.float16,
        pypto.DT_FP16,
        (10, 10, 30),
        (13, 23, 27),
        (13, 1, 1),
        None,
        marks=[pytest.mark.skip()],
        id="025",
    ),
    pytest.param(
        "sub_kernel_026",
        torch.int16,
        pypto.DT_INT16,
        (10, 25, 10),
        (13, 23, 27),
        (1, 23, 1),
        None,
        marks=[pytest.mark.skip()],
        id="026",
    ),
    pytest.param(
        "sub_kernel_027",
        torch.float16,
        pypto.DT_FP16,
        (23, 10, 10),
        (13, 23, 27),
        (1, 1, 27),
        None,
        marks=[pytest.mark.skip()],
        id="027",
    ),
    pytest.param(
        "sub_kernel_028",
        torch.float32,
        pypto.DT_FP32,
        (23, 20, 17),
        (63, 1, 1),
        (1, 43, 27),
        None,
        marks=[pytest.mark.skip()],
        id="028",
    ),
    pytest.param(
        "sub_kernel_029",
        torch.float32,
        pypto.DT_FP32,
        (5, 20, 15, 12),
        (5, 16, 11, 12),
        None,
        2.0,
        marks=[pytest.mark.skip()],
        id="029",
    ),
    pytest.param(
        "sub_kernel_030",
        torch.float16,
        pypto.DT_FP16,
        (5, 5, 10, 10),
        (5, 5, 6, 7),
        None,
        2.0,
        marks=[pytest.mark.skip()],
        id="030",
    ),
    pytest.param(
        "sub_kernel_031",
        torch.int32,
        pypto.DT_INT32,
        (5, 12, 16, 16),
        (21, 12, 15, 16),
        (21, 12, 15, 16),
        None,
        marks=[pytest.mark.skip()],
        id="031",
    ),
    pytest.param(
        "sub_kernel_032",
        torch.int16,
        pypto.DT_INT16,
        (12, 5, 15, 12),
        (11, 19, 13, 11),
        (11, 19, 13, 11),
        None,
        marks=[pytest.mark.skip()],
        id="032",
    ),
    pytest.param(
        "sub_kernel_033",
        torch.int32,
        pypto.DT_INT32,
        (21, 12, 5, 20),
        (1, 11, 13, 17),
        (21, 11, 13, 17),
        None,
        marks=[pytest.mark.skip()],
        id="033",
    ),
    pytest.param(
        "sub_kernel_034",
        torch.int16,
        pypto.DT_INT16,
        (11, 12, 15, 2),
        (11, 1, 15, 17),
        (11, 11, 15, 17),
        None,
        marks=[pytest.mark.skip()],
        id="034",
    ),
    pytest.param(
        "sub_kernel_035",
        torch.float32,
        pypto.DT_FP32,
        (15, 5, 15, 20),
        (21, 11, 1, 17),
        (21, 11, 13, 17),
        None,
        marks=[pytest.mark.skip()],
        id="035",
    ),
    pytest.param(
        "sub_kernel_036",
        torch.float16,
        pypto.DT_FP16,
        (13, 12, 3, 18),
        (25, 11, 15, 1),
        (25, 11, 15, 17),
        None,
        marks=[pytest.mark.skip()],
        id="036",
    ),
    pytest.param(
        "sub_kernel_037",
        torch.float32,
        pypto.DT_FP32,
        (10, 12, 15, 6),
        (21, 11, 13, 17),
        (1, 1, 13, 17),
        None,
        marks=[pytest.mark.skip()],
        id="037",
    ),
    pytest.param(
        "sub_kernel_038",
        torch.float16,
        pypto.DT_FP16,
        (15, 7, 5, 18),
        (15, 11, 15, 17),
        (1, 11, 1, 17),
        None,
        marks=[pytest.mark.skip()],
        id="038",
    ),
    pytest.param(
        "sub_kernel_039",
        torch.float32,
        pypto.DT_FP32,
        (21, 3, 13, 6),
        (21, 11, 13, 17),
        (1, 11, 13, 1),
        None,
        marks=[pytest.mark.skip()],
        id="039",
    ),
    pytest.param(
        "sub_kernel_040",
        torch.float16,
        pypto.DT_FP16,
        (25, 11, 5, 3),
        (25, 1, 1, 17),
        (25, 11, 15, 17),
        None,
        marks=[pytest.mark.skip()],
        id="040",
    ),
    pytest.param(
        "sub_kernel_041",
        torch.float32,
        pypto.DT_FP32,
        (5, 7, 7, 18),
        (22, 1, 13, 1),
        (22, 16, 13, 18),
        None,
        marks=[pytest.mark.skip()],
        id="041",
    ),
    pytest.param(
        "sub_kernel_042",
        torch.float16,
        pypto.DT_FP16,
        (5, 7, 15, 5),
        (22, 16, 1, 18),
        (22, 16, 13, 1),
        None,
        marks=[pytest.mark.skip()],
        id="042",
    ),
    pytest.param(
        "sub_kernel_043",
        torch.float32,
        pypto.DT_FP32,
        (5, 16, 7, 5),
        (1, 1, 1, 18),
        (22, 16, 13, 18),
        None,
        marks=[pytest.mark.skip()],
        id="043",
    ),
    pytest.param(
        "sub_kernel_044",
        torch.float16,
        pypto.DT_FP16,
        (22, 7, 7, 5),
        (1, 1, 13, 18),
        (22, 16, 13, 1),
        None,
        marks=[pytest.mark.skip()],
        id="044",
    ),
    pytest.param(
        "sub_kernel_045",
        torch.float32,
        pypto.DT_FP32,
        (5, 16, 7, 5),
        (1, 16, 13, 18),
        (22, 16, 1, 1),
        None,
        marks=[pytest.mark.skip()],
        id="045",
    ),
    pytest.param(
        "sub_kernel_046",
        torch.float16,
        pypto.DT_FP16,
        (22, 7, 7, 5),
        (22, 16, 13, 18),
        (22, 1, 1, 1),
        None,
        marks=[pytest.mark.skip()],
        id="046",
    ),
    pytest.param(
        "sub_kernel_047",
        torch.float32,
        pypto.DT_FP32,
        (5, 16, 7, 5),
        (22, 16, 13, 18),
        None,
        2.0,
        marks=[pytest.mark.skip()],
        id="047",
    ),
    pytest.param(
        "sub_kernel_048",
        torch.float32,
        pypto.DT_FP32,
        (11, 6, 6, 10),
        (1, 1, 13, 18),
        (22, 16, 1, 1),
        None,
        marks=[pytest.mark.skip()],
        id="048",
    ),
    pytest.param(
        "sub_kernel_049",
        torch.float16,
        pypto.DT_FP16,
        (1, 2, 16, 32),
        (1, 8, 16, 32),
        (1, 1, 16, 32),
        None,
        marks=[pytest.mark.skip()],
        id="049",
    ),
]


def run_sub_test(kernels, kernel_name, dtype, shape_in0, shape_in1, scalar_val):
    """Run a single sub kernel test with given kernels dict."""
    device = "cpu"
    is_int = dtype in (torch.int16, torch.int32)

    if is_int:
        input0 = torch.randint(-100, 100, shape_in0, dtype=dtype, device=device)
    else:
        input0 = torch.randn(shape_in0, dtype=dtype, device=device)

    if scalar_val is not None:
        input1 = scalar_val
        out_shape = shape_in0
    else:
        if is_int:
            input1 = torch.randint(-100, 100, shape_in1, dtype=dtype, device=device)
        else:
            input1 = torch.randn(shape_in1, dtype=dtype, device=device)
        out_shape = torch.broadcast_shapes(input0.shape, input1.shape)

    output = torch.empty(out_shape, dtype=dtype, device=device)

    if scalar_val is not None:
        kernels[kernel_name](input0, output, input1)
    else:
        kernels[kernel_name](input0, input1, output)

    golden = input0 - input1

    check_nan(output, name=kernel_name)
    cos_value = compare_cos(output.cpu().numpy(), golden.cpu().numpy())
    cos_value = abs(cos_value)
    if cos_value < 0.9999:
        raise AssertionError(f"{kernel_name}: cos_value {cos_value} < 0.9999")


def create_test_sub_module(soc_version):
    """Create a test module for sub with specified soc_version."""
    kernels = {}
    for p in TEST_CASES:
        kernel_name = p.values[0]
        dtype = p.values[2]
        tile_shapes = p.values[3]
        other_shape = p.values[5]
        if other_shape is None:
            kernels[kernel_name] = make_sub_scalar_kernel(soc_version, kernel_name, dtype, tile_shapes)
        else:
            kernels[kernel_name] = make_sub_tensor_kernel(soc_version, kernel_name, dtype, tile_shapes)
    return kernels, lambda: run_sub_test(kernels, None, None, None, None, None)


create_sub_kernels = create_test_sub_module


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
