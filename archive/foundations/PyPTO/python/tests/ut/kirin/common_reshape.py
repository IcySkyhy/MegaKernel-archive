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
Test reshape codegen - common functions for Kirin9030 and KirinX90
"""

import numpy as np
import pytest
import torch

from kirin.common import check_nan, compare_cos
import pypto


def make_reshape_kernel(soc_version, name, dtype, tile_shapes, output_shape, inplace=False):
    @pypto.frontend.jit(codegen_options={"soc_version": soc_version}, runtime_options={"run_mode": pypto.RunMode.SIM})
    def kernel(
        input_tensor: pypto.Tensor([...], dtype),
        out_tensor: pypto.Tensor([...], dtype),
    ):
        pypto.set_vec_tile_shapes(*tile_shapes)
        if inplace:
            input_tensor_tmp = pypto.add(input_tensor, 0)
            pypto.reshape(input_tensor_tmp, output_shape, inplace=True)
            out_tensor[:] = pypto.add(input_tensor_tmp, 0)
        else:
            out_tensor[:] = pypto.reshape(input_tensor, output_shape)

    kernel.__name__ = name
    return kernel


TEST_CASES = [
    # kernel_name: name of the kernel
    # torch_dtype: torch data type (float16, float32, int8, int16, int32)
    # pypto_dtype: pypto data type
    # tile_shapes: tile shape for pypto kernel
    # shape_input: input tensor shape
    # shape_out: output tensor shape (reshape result)
    # is_inplace: whether to use inplace reshape
    # marks: pytest marks
    # - kernel_name: name of the kernel in create_reshape_kernels dict
    # - torch_dtype: torch data type (float16, float32, int8, int16, int32)
    # - pypto_dtype: pypto data type (DT_FP16, DT_FP32, DT_INT8, etc.)
    # - tile_shapes: tile shape for pypto kernel
    # - shape_input: input tensor shape
    # - shape_out: output tensor shape (reshape result)
    # - is_inplace: whether to use inplace reshape (True/False)
    pytest.param(
        "reshape_kernel_fp16_001",
        torch.float16,
        pypto.DT_FP16,
        (1, 16),
        (2, 2),
        (1, 2, 1, 2),
        False,
        marks=[],
        id="001",
    ),
    pytest.param(
        "reshape_kernel_fp16_002",
        torch.float16,
        pypto.DT_FP16,
        (2, 16),
        (4, 4),
        (16,),
        False,
        marks=[pytest.mark.skip()],
        id="002",
    ),
    pytest.param(
        "reshape_kernel_fp16_003",
        torch.float16,
        pypto.DT_FP16,
        (1, 1, 16),
        (2, 3, 4),
        (2, 12),
        False,
        marks=[pytest.mark.skip()],
        id="003",
    ),
    pytest.param(
        "reshape_kernel_fp16_004",
        torch.float16,
        pypto.DT_FP16,
        (1, 1, 1, 16),
        (2, 2, 2, 2),
        (4, 4),
        False,
        marks=[pytest.mark.skip()],
        id="004",
    ),
    pytest.param(
        "reshape_kernel_fp16_005",
        torch.float16,
        pypto.DT_FP16,
        (16,),
        (8,),
        (2, 4),
        False,
        marks=[pytest.mark.skip()],
        id="005",
    ),
    pytest.param(
        "reshape_kernel_fp16_006",
        torch.float16,
        pypto.DT_FP16,
        (1, 16),
        (3, 9),
        (3, 3, 3),
        False,
        marks=[pytest.mark.skip()],
        id="006",
    ),
    pytest.param(
        "reshape_kernel_fp16_007",
        torch.float16,
        pypto.DT_FP16,
        (1, 1, 16),
        (2, 2, 6),
        (2, 2, 2, 3),
        False,
        marks=[pytest.mark.skip()],
        id="007",
    ),
    pytest.param(
        "reshape_kernel_fp16_008",
        torch.float16,
        pypto.DT_FP16,
        (2, 2, 2, 16),
        (4, 4, 4, 4),
        (256,),
        False,
        marks=[pytest.mark.skip()],
        id="008",
    ),
    pytest.param(
        "reshape_kernel_fp16_009",
        torch.float16,
        pypto.DT_FP16,
        (16,),
        (12,),
        (2, 2, 3),
        False,
        marks=[pytest.mark.skip()],
        id="009",
    ),
    pytest.param(
        "reshape_kernel_fp16_010",
        torch.float16,
        pypto.DT_FP16,
        (1, 16),
        (2, 6),
        (2, 2, 3, 1),
        False,
        marks=[pytest.mark.skip()],
        id="010",
    ),
    pytest.param(
        "reshape_kernel_fp32_001",
        torch.float32,
        pypto.DT_FP32,
        (1, 8),
        (2, 2),
        (1, 2, 1, 2),
        False,
        marks=[pytest.mark.skip()],
        id="011",
    ),
    pytest.param(
        "reshape_kernel_fp32_002",
        torch.float32,
        pypto.DT_FP32,
        (2, 8),
        (4, 4),
        (16,),
        False,
        marks=[pytest.mark.skip()],
        id="012",
    ),
    pytest.param(
        "reshape_kernel_fp32_003",
        torch.float32,
        pypto.DT_FP32,
        (1, 1, 8),
        (2, 3, 4),
        (2, 12),
        False,
        marks=[pytest.mark.skip()],
        id="013",
    ),
    pytest.param(
        "reshape_kernel_fp32_004",
        torch.float32,
        pypto.DT_FP32,
        (1, 1, 1, 8),
        (2, 2, 2, 2),
        (4, 4),
        False,
        marks=[pytest.mark.skip()],
        id="014",
    ),
    pytest.param(
        "reshape_kernel_fp32_005",
        torch.float32,
        pypto.DT_FP32,
        (8,),
        (8,),
        (2, 4),
        False,
        marks=[pytest.mark.skip()],
        id="015",
    ),
    pytest.param(
        "reshape_kernel_fp32_006",
        torch.float32,
        pypto.DT_FP32,
        (1, 8),
        (3, 9),
        (3, 3, 3),
        False,
        marks=[pytest.mark.skip()],
        id="016",
    ),
    pytest.param(
        "reshape_kernel_fp32_007",
        torch.float32,
        pypto.DT_FP32,
        (1, 1, 8),
        (2, 2, 6),
        (2, 2, 2, 3),
        False,
        marks=[pytest.mark.skip()],
        id="017",
    ),
    pytest.param(
        "reshape_kernel_fp32_008",
        torch.float32,
        pypto.DT_FP32,
        (2, 2, 2, 8),
        (4, 4, 4, 4),
        (256,),
        False,
        marks=[pytest.mark.skip()],
        id="018",
    ),
    pytest.param(
        "reshape_kernel_fp32_009",
        torch.float32,
        pypto.DT_FP32,
        (8,),
        (12,),
        (2, 2, 3),
        False,
        marks=[pytest.mark.skip()],
        id="019",
    ),
    pytest.param(
        "reshape_kernel_fp32_010",
        torch.float32,
        pypto.DT_FP32,
        (1, 8),
        (2, 6),
        (2, 2, 3, 1),
        False,
        marks=[pytest.mark.skip()],
        id="020",
    ),
    pytest.param(
        "reshape_kernel_inplace_fp32_001",
        torch.float32,
        pypto.DT_FP32,
        (1, 8),
        (2, 2),
        (1, 2, 1, 2),
        True,
        marks=[pytest.mark.skip()],
        id="021",
    ),
    pytest.param(
        "reshape_kernel_inplace_fp32_002",
        torch.float32,
        pypto.DT_FP32,
        (2, 8),
        (4, 4),
        (16,),
        True,
        marks=[pytest.mark.skip()],
        id="022",
    ),
    pytest.param(
        "reshape_kernel_inplace_fp32_003",
        torch.float32,
        pypto.DT_FP32,
        (1, 1, 8),
        (2, 3, 4),
        (2, 12),
        True,
        marks=[pytest.mark.skip()],
        id="023",
    ),
    pytest.param(
        "reshape_kernel_inplace_fp32_004",
        torch.float32,
        pypto.DT_FP32,
        (1, 1, 1, 8),
        (2, 2, 2, 2),
        (4, 4),
        True,
        marks=[pytest.mark.skip()],
        id="024",
    ),
    pytest.param(
        "reshape_kernel_inplace_fp32_005",
        torch.float32,
        pypto.DT_FP32,
        (8,),
        (8,),
        (2, 4),
        True,
        marks=[pytest.mark.skip()],
        id="025",
    ),
    pytest.param(
        "reshape_kernel_inplace_fp32_006",
        torch.float32,
        pypto.DT_FP32,
        (1, 8),
        (40, 20),
        (20, 40),
        True,
        marks=[pytest.mark.skip()],
        id="026",
    ),
    pytest.param(
        "reshape_kernel_int8_001",
        torch.int8,
        pypto.DT_INT8,
        (1, 32),
        (2, 2),
        (1, 2, 1, 2),
        False,
        marks=[pytest.mark.skip()],
        id="027",
    ),
    pytest.param(
        "reshape_kernel_int8_002",
        torch.int8,
        pypto.DT_INT8,
        (2, 32),
        (4, 4),
        (16,),
        False,
        marks=[pytest.mark.skip()],
        id="028",
    ),
    pytest.param(
        "reshape_kernel_int8_003",
        torch.int8,
        pypto.DT_INT8,
        (32,),
        (8,),
        (2, 4),
        False,
        marks=[pytest.mark.skip()],
        id="029",
    ),
    pytest.param(
        "reshape_kernel_int8_004",
        torch.int8,
        pypto.DT_INT8,
        (1, 1, 32),
        (2, 3, 4),
        (2, 12),
        False,
        marks=[pytest.mark.skip()],
        id="030",
    ),
    pytest.param(
        "reshape_kernel_int8_005",
        torch.int8,
        pypto.DT_INT8,
        (32,),
        (12,),
        (2, 2, 3),
        False,
        marks=[pytest.mark.skip()],
        id="031",
    ),
    pytest.param(
        "reshape_kernel_int16_001",
        torch.int16,
        pypto.DT_INT16,
        (1, 16),
        (2, 2),
        (1, 2, 1, 2),
        False,
        marks=[pytest.mark.skip()],
        id="032",
    ),
    pytest.param(
        "reshape_kernel_int16_002",
        torch.int16,
        pypto.DT_INT16,
        (2, 16),
        (4, 4),
        (16,),
        False,
        marks=[pytest.mark.skip()],
        id="033",
    ),
    pytest.param(
        "reshape_kernel_int16_003",
        torch.int16,
        pypto.DT_INT16,
        (16,),
        (8,),
        (2, 4),
        False,
        marks=[pytest.mark.skip()],
        id="034",
    ),
    pytest.param(
        "reshape_kernel_int16_004",
        torch.int16,
        pypto.DT_INT16,
        (1, 1, 16),
        (2, 3, 4),
        (2, 12),
        False,
        marks=[pytest.mark.skip()],
        id="035",
    ),
    pytest.param(
        "reshape_kernel_int16_005",
        torch.int16,
        pypto.DT_INT16,
        (16,),
        (12,),
        (2, 2, 3),
        False,
        marks=[pytest.mark.skip()],
        id="036",
    ),
    pytest.param(
        "reshape_kernel_int32_001",
        torch.int32,
        pypto.DT_INT32,
        (1, 8),
        (2, 2),
        (1, 2, 1, 2),
        False,
        marks=[pytest.mark.skip()],
        id="037",
    ),
    pytest.param(
        "reshape_kernel_int32_002",
        torch.int32,
        pypto.DT_INT32,
        (2, 8),
        (4, 4),
        (16,),
        False,
        marks=[pytest.mark.skip()],
        id="038",
    ),
    pytest.param(
        "reshape_kernel_int32_003",
        torch.int32,
        pypto.DT_INT32,
        (8,),
        (8,),
        (2, 4),
        False,
        marks=[pytest.mark.skip()],
        id="039",
    ),
    pytest.param(
        "reshape_kernel_int32_004",
        torch.int32,
        pypto.DT_INT32,
        (1, 1, 8),
        (2, 3, 4),
        (2, 12),
        False,
        marks=[pytest.mark.skip()],
        id="040",
    ),
    pytest.param(
        "reshape_kernel_int32_005",
        torch.int32,
        pypto.DT_INT32,
        (8,),
        (12,),
        (2, 2, 3),
        False,
        marks=[pytest.mark.skip()],
        id="041",
    ),
]


def run_reshape_test(kernels, kernel_name, dtype, shape_input, shape_out, is_inplace):
    """Run a single reshape kernel test with given kernels dict."""
    device = "cpu"
    is_int_dtype = dtype in (torch.int8, torch.int16, torch.int32)

    if is_int_dtype:
        input_tensor = torch.randint(0, 100, shape_input, dtype=dtype, device=device)
        out_tensor = torch.randint(0, 100, shape_out, dtype=dtype, device=device)
    else:
        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

    kernels[kernel_name](input_tensor, out_tensor)

    golden_out = input_tensor.reshape(shape_out)

    out_np = np.array(out_tensor.cpu())
    expect_np = np.array(golden_out.cpu())

    check_nan(out_tensor, name=kernel_name)
    cos_value = abs(compare_cos(out_np, expect_np))
    if cos_value < 0.9999:
        raise AssertionError(f"{kernel_name}: cos_value {cos_value} < 0.9999")


def create_reshape_kernels(soc_version):
    return {
        p.values[0]: make_reshape_kernel(soc_version, p.values[0], p.values[2], p.values[3], p.values[5], p.values[6])
        for p in TEST_CASES
    }


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
