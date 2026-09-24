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
Test view codegen - common functions for Kirin9030 and KirinX90
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as functional

from kirin.common import check_nan, compare_cos
import pypto


def pypto_view_in_torch(x, shape=None, offsets=None, valid_shape=None, dtype=None):
    if dtype is not None:
        x = x.view(dtype)

    if shape is None:
        return x

    if offsets is None:
        offsets = [0] * len(shape)

    take_shape = valid_shape if valid_shape is not None else shape

    indices = []
    for i, s in enumerate(take_shape):
        indices.append(slice(offsets[i], offsets[i] + s))

    y = x[tuple(indices)]

    if list(y.shape) != list(shape):
        padding = []
        for sh, ysh in zip(reversed(shape), reversed(y.shape)):
            pad_after = sh - ysh
            padding.extend([0, pad_after])

        y = functional.pad(y, tuple(padding), mode="constant", value=0)

    return y


def make_view_kernel(soc_version, name, input_shape, tile_shapes, view_shape, offsets, dtype_enum):
    @pypto.frontend.jit(codegen_options={"soc_version": soc_version}, runtime_options={"run_mode": pypto.RunMode.SIM})
    def kernel(
        input_tensor: pypto.Tensor(list(input_shape), dtype_enum),
        out_tensor: pypto.Tensor([...], dtype_enum),
    ):
        pypto.set_vec_tile_shapes(*tile_shapes)
        out_tensor[:] = input_tensor.view(list(view_shape), list(offsets))

    kernel.__name__ = name
    return kernel


TEST_CASES = [
    # kernel_name: name of the kernel
    # torch_dtype: torch data type (float16, float32, int8, int16, int32)
    # pypto_dtype: pypto data type
    # tile_shapes: tile shape for pypto kernel
    # kernel_shape: input tensor shape used for kernel creation
    # test_shape: input tensor shape used for test tensor creation
    # view_shape: output tensor shape (view result)
    # offsets: slice offsets for the view operation
    # marks: pytest marks
    # - kernel_name: name of the kernel in create_view_kernels dict
    # - torch_dtype: torch data type (float16, float32, int8, int16, int32)
    # - pypto_dtype: pypto data type (DT_FP16, DT_FP32, DT_INT8, etc.)
    # - tile_shapes: tile shape for pypto kernel
    # - kernel_shape: input tensor shape used for kernel creation
    # - test_shape: input tensor shape used for test tensor creation
    # - view_shape: output tensor shape (view result)
    # - offsets: slice offsets for the view operation
    pytest.param(
        "view_kernel_fp16", torch.float16, pypto.DT_FP16, (2, 16), [4, 8], (4, 8), (4, 4), [0, 4], marks=[], id="001"
    ),
    pytest.param(
        "view_kernel_fp16_002",
        torch.float16,
        pypto.DT_FP16,
        (2, 16),
        [4, 8],
        (4, 8),
        (2, 4),
        [0, 0],
        marks=[pytest.mark.skip()],
        id="002",
    ),
    pytest.param(
        "view_kernel_fp16_003",
        torch.float16,
        pypto.DT_FP16,
        (1, 2, 16),
        [2, 4, 8],
        (2, 4, 8),
        (2, 4, 4),
        [0, 0, 4],
        marks=[pytest.mark.skip()],
        id="003",
    ),
    pytest.param(
        "view_kernel_fp16_004",
        torch.float16,
        pypto.DT_FP16,
        (1, 2, 16),
        [2, 4, 8],
        (2, 4, 8),
        (2, 2, 4),
        [0, 0, 0],
        marks=[pytest.mark.skip()],
        id="004",
    ),
    pytest.param(
        "view_kernel_fp16_005",
        torch.float16,
        pypto.DT_FP16,
        (16,),
        [8],
        (8,),
        (4,),
        [4],
        marks=[pytest.mark.skip()],
        id="005",
    ),
    pytest.param(
        "view_kernel_fp16_006",
        torch.float16,
        pypto.DT_FP16,
        (1, 16),
        [3, 6],
        (3, 6),
        (3, 3),
        [0, 3],
        marks=[pytest.mark.skip()],
        id="006",
    ),
    pytest.param(
        "view_kernel_fp16_007",
        torch.float16,
        pypto.DT_FP16,
        (1, 1, 16),
        [2, 3, 6],
        (2, 3, 6),
        (2, 3, 3),
        [0, 0, 3],
        marks=[pytest.mark.skip()],
        id="007",
    ),
    pytest.param(
        "view_kernel_fp16_008",
        torch.float16,
        pypto.DT_FP16,
        (1, 1, 1, 16),
        [2, 2, 2, 4],
        (2, 2, 2, 4),
        (2, 2, 2, 2),
        [0, 0, 0, 2],
        marks=[pytest.mark.skip()],
        id="008",
    ),
    pytest.param(
        "view_kernel_fp16_009",
        torch.float16,
        pypto.DT_FP16,
        (16,),
        [6],
        (6,),
        (4,),
        [2],
        marks=[pytest.mark.skip()],
        id="009",
    ),
    pytest.param(
        "view_kernel_fp16_010",
        torch.float16,
        pypto.DT_FP16,
        (1, 16),
        [2, 8],
        (2, 8),
        (2, 4),
        [0, 4],
        marks=[pytest.mark.skip()],
        id="010",
    ),
    pytest.param(
        "view_kernel_fp32",
        torch.float32,
        pypto.DT_FP32,
        (2, 8),
        [4, 8],
        (4, 8),
        (4, 4),
        [0, 4],
        marks=[pytest.mark.skip()],
        id="011",
    ),
    pytest.param(
        "view_kernel_fp32_002",
        torch.float32,
        pypto.DT_FP32,
        (2, 8),
        [4, 8],
        (4, 8),
        (2, 4),
        [0, 0],
        marks=[pytest.mark.skip()],
        id="012",
    ),
    pytest.param(
        "view_kernel_fp32_003",
        torch.float32,
        pypto.DT_FP32,
        (1, 2, 8),
        [2, 4, 8],
        (2, 4, 8),
        (2, 4, 4),
        [0, 0, 4],
        marks=[pytest.mark.skip()],
        id="013",
    ),
    pytest.param(
        "view_kernel_fp32_004",
        torch.float32,
        pypto.DT_FP32,
        (1, 2, 8),
        [2, 4, 8],
        (2, 4, 8),
        (2, 2, 4),
        [0, 0, 0],
        marks=[pytest.mark.skip()],
        id="014",
    ),
    pytest.param(
        "view_kernel_fp32_005",
        torch.float32,
        pypto.DT_FP32,
        (8,),
        [8],
        (8,),
        (4,),
        [4],
        marks=[pytest.mark.skip()],
        id="015",
    ),
    pytest.param(
        "view_kernel_fp32_006",
        torch.float32,
        pypto.DT_FP32,
        (1, 8),
        [3, 6],
        (3, 6),
        (3, 3),
        [0, 3],
        marks=[pytest.mark.skip()],
        id="016",
    ),
    pytest.param(
        "view_kernel_fp32_007",
        torch.float32,
        pypto.DT_FP32,
        (1, 1, 8),
        [2, 3, 6],
        (2, 3, 6),
        (2, 3, 3),
        [0, 0, 3],
        marks=[pytest.mark.skip()],
        id="017",
    ),
    pytest.param(
        "view_kernel_fp32_008",
        torch.float32,
        pypto.DT_FP32,
        (1, 1, 1, 8),
        [2, 2, 2, 4],
        (2, 2, 2, 4),
        (2, 2, 2, 2),
        [0, 0, 0, 2],
        marks=[pytest.mark.skip()],
        id="018",
    ),
    pytest.param(
        "view_kernel_fp32_009",
        torch.float32,
        pypto.DT_FP32,
        (8,),
        [6],
        (6,),
        (4,),
        [2],
        marks=[pytest.mark.skip()],
        id="019",
    ),
    pytest.param(
        "view_kernel_fp32_010",
        torch.float32,
        pypto.DT_FP32,
        (1, 8),
        [2, 8],
        (2, 8),
        (2, 4),
        [0, 4],
        marks=[pytest.mark.skip()],
        id="020",
    ),
    pytest.param(
        "view_kernel_int32",
        torch.int32,
        pypto.DT_INT32,
        (2, 8),
        [4, 8],
        (4, 8),
        (4, 4),
        [0, 4],
        marks=[pytest.mark.skip()],
        id="021",
    ),
    pytest.param(
        "view_kernel_int32_002",
        torch.int32,
        pypto.DT_INT32,
        (8,),
        [8],
        (8,),
        (4,),
        [4],
        marks=[pytest.mark.skip()],
        id="022",
    ),
    pytest.param(
        "view_kernel_int32_003",
        torch.int32,
        pypto.DT_INT32,
        (1, 2, 8),
        [2, 4, 8],
        (2, 4, 8),
        (2, 4, 4),
        [0, 0, 4],
        marks=[pytest.mark.skip()],
        id="023",
    ),
    pytest.param(
        "view_kernel_int8",
        torch.int8,
        pypto.DT_INT8,
        (2, 32),
        [4, 8],
        (4, 8),
        (4, 4),
        [0, 4],
        marks=[pytest.mark.skip()],
        id="024",
    ),
    pytest.param(
        "view_kernel_int8_002",
        torch.int8,
        pypto.DT_INT8,
        (32,),
        [8],
        (8,),
        (4,),
        [4],
        marks=[pytest.mark.skip()],
        id="025",
    ),
    pytest.param(
        "view_kernel_int8_003",
        torch.int8,
        pypto.DT_INT8,
        (1, 2, 32),
        [2, 4, 8],
        (2, 4, 8),
        (2, 4, 4),
        [0, 0, 4],
        marks=[pytest.mark.skip()],
        id="026",
    ),
    pytest.param(
        "view_kernel_int16",
        torch.int16,
        pypto.DT_INT16,
        (2, 16),
        [4, 8],
        (4, 8),
        (4, 4),
        [0, 4],
        marks=[pytest.mark.skip()],
        id="027",
    ),
    pytest.param(
        "view_kernel_int16_002",
        torch.int16,
        pypto.DT_INT16,
        (16,),
        [8],
        (8,),
        (4,),
        [4],
        marks=[pytest.mark.skip()],
        id="028",
    ),
    pytest.param(
        "view_kernel_int16_003",
        torch.int16,
        pypto.DT_INT16,
        (1, 2, 16),
        [2, 4, 8],
        (2, 4, 8),
        (2, 4, 4),
        [0, 0, 4],
        marks=[pytest.mark.skip()],
        id="029",
    ),
]


def _compute_golden_view(input_tensor, shape_out, offsets):
    return pypto_view_in_torch(input_tensor, shape=list(shape_out), offsets=offsets)


def run_view_test(kernels, kernel_name, dtype, shape_input, shape_out, offsets):
    """Run a single view kernel test with given kernels dict."""
    device = "cpu"
    is_int = dtype in (torch.int8, torch.int16, torch.int32)

    if is_int:
        if dtype == torch.int8:
            input_tensor = torch.randint(-128, 127, shape_input, dtype=dtype, device=device)
        elif dtype == torch.int16:
            input_tensor = torch.randint(-32768, 32767, shape_input, dtype=dtype, device=device)
        else:
            input_tensor = torch.randint(0, 100, shape_input, dtype=dtype, device=device)
    else:
        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)

    if is_int:
        if dtype == torch.int8:
            out_tensor = torch.randint(-128, 127, shape_out, dtype=dtype, device=device)
        elif dtype == torch.int16:
            out_tensor = torch.randint(-32768, 32767, shape_out, dtype=dtype, device=device)
        else:
            out_tensor = torch.randint(0, 100, shape_out, dtype=dtype, device=device)
    else:
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

    kernels[kernel_name](input_tensor, out_tensor)

    golden_out = _compute_golden_view(input_tensor, shape_out, offsets)
    out_np = np.array(out_tensor.cpu().to(torch.int32) if is_int else out_tensor.cpu())
    expect_np = np.array(golden_out.cpu().to(torch.int32) if is_int else golden_out.cpu())

    check_nan(out_tensor, name=kernel_name)
    cos_value = abs(compare_cos(out_np, expect_np))
    if cos_value < 0.9999:
        raise AssertionError(f"{kernel_name}: cos_value {cos_value} < 0.9999")


def create_view_kernels(soc_version):
    return {
        p.values[0]: make_view_kernel(
            soc_version, p.values[0], p.values[5], p.values[3], p.values[6], p.values[7], p.values[2]
        )
        for p in TEST_CASES
    }


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
