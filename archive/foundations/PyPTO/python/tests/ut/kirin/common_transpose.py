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
Test transpose codegen - common functions for Kirin9030 and KirinX90
"""

import numpy as np
import pytest
import torch

from kirin.common import check_nan, compare_cos
import pypto


def make_transpose_kernel(soc_version, name, shape, tile_shapes, dim0, dim1, dtype_enum):
    @pypto.frontend.jit(codegen_options={"soc_version": soc_version}, runtime_options={"run_mode": pypto.RunMode.SIM})
    def kernel(
        input_tensor: pypto.Tensor(shape, dtype_enum),
        out_tensor: pypto.Tensor([...], dtype_enum),
    ):
        pypto.set_vec_tile_shapes(*tile_shapes)
        out_tensor[:] = pypto.transpose(input_tensor, dim0, dim1)

    kernel.__name__ = name
    return kernel


TEST_CASES = [
    # Per-entry positional layout (matches the parametrize() arglist in the caller):
    #   [0] kernel_name   : key used to look up the compiled kernel in KERNELS.
    #                       Base kernels (no _NNN suffix) are rank-only [...] and
    #                       are created once and shared across their 2D test cases
    #                       (001/002, 011/012, 021/022, 026/027); suffixed-name
    #                       kernels are shape-locked and 1:1 with one test case.
    #   [1] torch_dtype   : torch tensor element type used at runtime.
    #   [2] pypto_dtype   : matching pypto DataType enum for the kernel annotation.
    #   [3] tile_shapes   : (Tuple[int, ...]) forwarded to pypto.set_vec_tile_shapes.
    #   [4] kernel_shape  : list[int] (shape-locked) or [...] (rank-only) — baked
    #                       into the kernel's input Tensor annotation.
    #   [5] test_shape    : actual torch tensor shape supplied by the test.
    #   [6] dim0, [7] dim1: axes to transpose.
    #   id="NNN"          : pytest test id, zero-padded to 3 digits.
    pytest.param(
        "transpose_kernel_fp16", torch.float16, pypto.DT_FP16, (1, 16), [...], (2, 3), 0, 1, marks=[], id="001"
    ),
    pytest.param(
        "transpose_kernel_fp16",
        torch.float16,
        pypto.DT_FP16,
        (1, 16),
        [...],
        (3, 4),
        0,
        1,
        marks=[pytest.mark.skip()],
        id="002",
    ),
    pytest.param(
        "transpose_kernel_fp16_003",
        torch.float16,
        pypto.DT_FP16,
        (1, 2, 16),
        [3, 4, 5],
        (3, 4, 5),
        1,
        2,
        marks=[pytest.mark.skip()],
        id="003",
    ),
    pytest.param(
        "transpose_kernel_fp16_004",
        torch.float16,
        pypto.DT_FP16,
        (1, 3, 16),
        [2, 3, 4],
        (2, 3, 4),
        0,
        2,
        marks=[pytest.mark.skip()],
        id="004",
    ),
    pytest.param(
        "transpose_kernel_fp16_005",
        torch.float16,
        pypto.DT_FP16,
        (1, 1, 2, 16),
        [2, 3, 4, 5],
        (2, 3, 4, 5),
        2,
        3,
        marks=[pytest.mark.skip()],
        id="005",
    ),
    pytest.param(
        "transpose_kernel_fp16_006",
        torch.float16,
        pypto.DT_FP16,
        (1, 3, 1, 16),
        [2, 3, 4, 5],
        (2, 3, 4, 5),
        1,
        2,
        marks=[pytest.mark.skip()],
        id="006",
    ),
    pytest.param(
        "transpose_kernel_fp16_007",
        torch.float16,
        pypto.DT_FP16,
        (1, 1, 1, 2, 16),
        [2, 3, 4, 5, 6],
        (2, 3, 4, 5, 6),
        3,
        4,
        marks=[pytest.mark.skip()],
        id="007",
    ),
    pytest.param(
        "transpose_kernel_fp16_008",
        torch.float16,
        pypto.DT_FP16,
        (2, 16),
        [4, 5],
        (4, 5),
        0,
        1,
        marks=[pytest.mark.skip()],
        id="008",
    ),
    pytest.param(
        "transpose_kernel_fp16_009",
        torch.float16,
        pypto.DT_FP16,
        (1, 5, 16),
        [2, 5, 6],
        (2, 5, 6),
        1,
        2,
        marks=[pytest.mark.skip()],
        id="009",
    ),
    pytest.param(
        "transpose_kernel_fp16_010",
        torch.float16,
        pypto.DT_FP16,
        (1, 2, 2, 16),
        [3, 2, 4, 5],
        (3, 2, 4, 5),
        2,
        3,
        marks=[pytest.mark.skip()],
        id="010",
    ),
    pytest.param(
        "transpose_kernel_fp32",
        torch.float32,
        pypto.DT_FP32,
        (1, 8),
        [...],
        (2, 3),
        0,
        1,
        marks=[pytest.mark.skip()],
        id="011",
    ),
    pytest.param(
        "transpose_kernel_fp32",
        torch.float32,
        pypto.DT_FP32,
        (1, 8),
        [...],
        (3, 4),
        0,
        1,
        marks=[pytest.mark.skip()],
        id="012",
    ),
    pytest.param(
        "transpose_kernel_fp32_003",
        torch.float32,
        pypto.DT_FP32,
        (1, 2, 8),
        [3, 4, 5],
        (3, 4, 5),
        1,
        2,
        marks=[pytest.mark.skip()],
        id="013",
    ),
    pytest.param(
        "transpose_kernel_fp32_004",
        torch.float32,
        pypto.DT_FP32,
        (1, 3, 8),
        [2, 3, 4],
        (2, 3, 4),
        0,
        2,
        marks=[pytest.mark.skip()],
        id="014",
    ),
    pytest.param(
        "transpose_kernel_fp32_005",
        torch.float32,
        pypto.DT_FP32,
        (1, 1, 2, 8),
        [2, 3, 4, 5],
        (2, 3, 4, 5),
        2,
        3,
        marks=[pytest.mark.skip()],
        id="015",
    ),
    pytest.param(
        "transpose_kernel_fp32_006",
        torch.float32,
        pypto.DT_FP32,
        (1, 3, 1, 8),
        [2, 3, 4, 5],
        (2, 3, 4, 5),
        1,
        2,
        marks=[pytest.mark.skip()],
        id="016",
    ),
    pytest.param(
        "transpose_kernel_fp32_007",
        torch.float32,
        pypto.DT_FP32,
        (1, 1, 1, 2, 8),
        [2, 3, 4, 5, 6],
        (2, 3, 4, 5, 6),
        3,
        4,
        marks=[pytest.mark.skip()],
        id="017",
    ),
    pytest.param(
        "transpose_kernel_fp32_008",
        torch.float32,
        pypto.DT_FP32,
        (2, 8),
        [4, 5],
        (4, 5),
        0,
        1,
        marks=[pytest.mark.skip()],
        id="018",
    ),
    pytest.param(
        "transpose_kernel_fp32_009",
        torch.float32,
        pypto.DT_FP32,
        (1, 5, 8),
        [2, 5, 6],
        (2, 5, 6),
        1,
        2,
        marks=[pytest.mark.skip()],
        id="019",
    ),
    pytest.param(
        "transpose_kernel_fp32_010",
        torch.float32,
        pypto.DT_FP32,
        (1, 2, 2, 8),
        [3, 2, 4, 5],
        (3, 2, 4, 5),
        2,
        3,
        marks=[pytest.mark.skip()],
        id="020",
    ),
    pytest.param(
        "transpose_kernel_int32",
        torch.int32,
        pypto.DT_INT32,
        (1, 8),
        [...],
        (2, 3),
        0,
        1,
        marks=[pytest.mark.skip()],
        id="021",
    ),
    pytest.param(
        "transpose_kernel_int32",
        torch.int32,
        pypto.DT_INT32,
        (1, 8),
        [...],
        (3, 4),
        0,
        1,
        marks=[pytest.mark.skip()],
        id="022",
    ),
    pytest.param(
        "transpose_kernel_int32_003",
        torch.int32,
        pypto.DT_INT32,
        (1, 2, 8),
        [3, 4, 5],
        (3, 4, 5),
        1,
        2,
        marks=[pytest.mark.skip()],
        id="023",
    ),
    pytest.param(
        "transpose_kernel_int32_004",
        torch.int32,
        pypto.DT_INT32,
        (1, 1, 2, 8),
        [2, 3, 4, 5],
        (2, 3, 4, 5),
        2,
        3,
        marks=[pytest.mark.skip()],
        id="024",
    ),
    pytest.param(
        "transpose_kernel_int32_005",
        torch.int32,
        pypto.DT_INT32,
        (2, 8),
        [4, 5],
        (4, 5),
        0,
        1,
        marks=[pytest.mark.skip()],
        id="025",
    ),
    pytest.param(
        "transpose_kernel_int16",
        torch.int16,
        pypto.DT_INT16,
        (1, 16),
        [...],
        (2, 3),
        0,
        1,
        marks=[pytest.mark.skip()],
        id="026",
    ),
    pytest.param(
        "transpose_kernel_int16",
        torch.int16,
        pypto.DT_INT16,
        (1, 16),
        [...],
        (3, 4),
        0,
        1,
        marks=[pytest.mark.skip()],
        id="027",
    ),
    pytest.param(
        "transpose_kernel_int16_003",
        torch.int16,
        pypto.DT_INT16,
        (1, 2, 16),
        [3, 4, 5],
        (3, 4, 5),
        1,
        2,
        marks=[pytest.mark.skip()],
        id="028",
    ),
    pytest.param(
        "transpose_kernel_int16_004",
        torch.int16,
        pypto.DT_INT16,
        (1, 3, 16),
        [2, 3, 4],
        (2, 3, 4),
        0,
        2,
        marks=[pytest.mark.skip()],
        id="029",
    ),
    pytest.param(
        "transpose_kernel_int16_005",
        torch.int16,
        pypto.DT_INT16,
        (1, 1, 2, 16),
        [2, 3, 4, 5],
        (2, 3, 4, 5),
        2,
        3,
        marks=[pytest.mark.skip()],
        id="030",
    ),
    pytest.param(
        "transpose_kernel_int16_006",
        torch.int16,
        pypto.DT_INT16,
        (1, 3, 1, 16),
        [2, 3, 4, 5],
        (2, 3, 4, 5),
        1,
        2,
        marks=[pytest.mark.skip()],
        id="031",
    ),
    pytest.param(
        "transpose_kernel_int16_007",
        torch.int16,
        pypto.DT_INT16,
        (1, 1, 1, 2, 16),
        [2, 3, 4, 5, 6],
        (2, 3, 4, 5, 6),
        3,
        4,
        marks=[pytest.mark.skip()],
        id="032",
    ),
    pytest.param(
        "transpose_kernel_int16_008",
        torch.int16,
        pypto.DT_INT16,
        (2, 16),
        [4, 5],
        (4, 5),
        0,
        1,
        marks=[pytest.mark.skip()],
        id="033",
    ),
    pytest.param(
        "transpose_kernel_int16_009",
        torch.int16,
        pypto.DT_INT16,
        (1, 5, 16),
        [2, 5, 6],
        (2, 5, 6),
        1,
        2,
        marks=[pytest.mark.skip()],
        id="034",
    ),
    pytest.param(
        "transpose_kernel_int16_010",
        torch.int16,
        pypto.DT_INT16,
        (1, 2, 2, 16),
        [3, 2, 4, 5],
        (3, 2, 4, 5),
        2,
        3,
        marks=[pytest.mark.skip()],
        id="035",
    ),
    pytest.param(
        "transpose_kernel_int16_011",
        torch.int16,
        pypto.DT_INT16,
        (2, 2, 2, 16),
        [3, 2, 4, 5],
        (3, 2, 4, 5),
        2,
        3,
        marks=[pytest.mark.skip()],
        id="036",
    ),
]


def _compute_golden_transpose(input_tensor, dim0, dim1):
    return torch.transpose(input_tensor, dim0, dim1)


def run_transpose_test(kernels, kernel_name, dtype, shape, dim0, dim1):
    device = "cpu"
    is_int = dtype in (torch.int16, torch.int32)

    if is_int:
        input_tensor = torch.randint(0, 100, shape, dtype=dtype, device=device)
    else:
        input_tensor = torch.rand(shape, dtype=dtype, device=device)

    perm = list(range(len(shape)))
    perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
    out_shape = tuple(shape[i] for i in perm)

    if is_int:
        out_tensor = torch.randint(0, 100, out_shape, dtype=dtype, device=device)
    else:
        out_tensor = torch.rand(out_shape, dtype=dtype, device=device)

    kernels[kernel_name](input_tensor, out_tensor)

    golden_out = _compute_golden_transpose(input_tensor, dim0, dim1)
    out_np = np.array(out_tensor.cpu().to(torch.int32) if is_int else out_tensor.cpu())
    expect_np = np.array(golden_out.cpu().to(torch.int32) if is_int else golden_out.cpu())

    check_nan(out_tensor, name=kernel_name)
    cos_value = abs(compare_cos(out_np, expect_np))
    if cos_value < 0.9999:
        raise AssertionError(f"{kernel_name}: cos_value {cos_value} < 0.9999")


def create_transpose_kernels(soc_version):
    kernels = {}
    for p in TEST_CASES:
        kernel_name = p.values[0]
        if kernel_name in kernels:
            continue
        kernels[kernel_name] = make_transpose_kernel(
            soc_version,
            kernel_name,
            p.values[4],
            p.values[3],
            p.values[6],
            p.values[7],
            p.values[2],
        )
    return kernels


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
