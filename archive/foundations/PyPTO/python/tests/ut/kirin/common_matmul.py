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
Test matmul codegen - common functions for Kirin9030 and KirinX90
"""

import numpy as np
import pytest
import torch

from kirin.common import check_nan, compare_cos
import pypto

DTYPE_MAP = {
    torch.float16: pypto.DT_FP16,
    torch.float32: pypto.DT_FP32,
    torch.int8: pypto.DT_INT8,
    torch.int32: pypto.DT_INT32,
}


def get_matmul_kernel(
    soc_version,
    name,
    in_dtype,
    out_dtype,
    cube_tile_shape,
    has_bias=False,
    bias_shape=None,
    a_trans=False,
    b_trans=False,
):
    # X90 硬件要求 bias 以 32 位 fp32 格式传入（低 16 位有效，高 16 位无效）
    bias_dtype = pypto.DT_FP32 if (soc_version == "KirinX90" and out_dtype == pypto.DT_FP16) else out_dtype
    if has_bias:

        @pypto.frontend.jit(
            codegen_options={"soc_version": soc_version}, runtime_options={"run_mode": pypto.RunMode.SIM}
        )
        def matmul_kernel(
            a: pypto.Tensor([...], in_dtype),
            b: pypto.Tensor([...], in_dtype),
            bias: pypto.Tensor([...], bias_dtype),
            c: pypto.Tensor([...], out_dtype),
        ):
            pypto.set_cube_tile_shapes(cube_tile_shape[0], cube_tile_shape[1], cube_tile_shape[2])
            extend_params = {"bias_tensor": bias}
            c[:] = pypto.matmul(
                a, b, out_dtype, a_trans=a_trans, b_trans=b_trans, c_matrix_nz=False, extend_params=extend_params
            )

        matmul_kernel.__name__ = name
        return matmul_kernel
    else:

        @pypto.frontend.jit(
            codegen_options={"soc_version": soc_version}, runtime_options={"run_mode": pypto.RunMode.SIM}
        )
        def matmul_kernel(
            a: pypto.Tensor([...], in_dtype),
            b: pypto.Tensor([...], in_dtype),
            c: pypto.Tensor([...], out_dtype),
        ):
            pypto.set_cube_tile_shapes(cube_tile_shape[0], cube_tile_shape[1], cube_tile_shape[2])
            c[:] = pypto.matmul(a, b, out_dtype, a_trans=a_trans, b_trans=b_trans)

        matmul_kernel.__name__ = name
        return matmul_kernel


TEST_CASES = [
    # kernel_name: name of the kernel
    # torch_in_dtype: torch data type for input tensor a
    # torch_out_dtype: torch data type for output tensor c
    # pypto_in_dtype: pypto data type for input tensor a
    # pypto_out_dtype: pypto data type for output tensor c
    # shape_a: input tensor a shape
    # shape_b: input tensor b shape
    # shape_c: output tensor c shape
    # cube_tile_shape: cube tile shapes for matmul
    # has_bias: whether to use bias
    # bias_shape: bias tensor shape (or None)
    # a_trans: whether to transpose a
    # b_trans: whether to transpose b
    # marks: pytest marks
    pytest.param(
        "matmul_fp16_001",
        torch.float16,
        torch.float16,
        pypto.DT_FP16,
        pypto.DT_FP16,
        (16, 16),
        (16, 16),
        (16, 16),
        ([16, 16], [16, 16], [16, 16]),
        False,
        None,
        False,
        False,
        marks=[],
        id="001",
    ),
    pytest.param(
        "matmul_fp16_002",
        torch.float16,
        torch.float16,
        pypto.DT_FP16,
        pypto.DT_FP16,
        (128, 130),
        (130, 32),
        (128, 32),
        ([16, 16], [32, 32], [16, 16]),
        True,
        (1, 32),
        False,
        False,
        marks=[pytest.mark.skip()],
        id="002",
    ),
    pytest.param(
        "matmul_fp16_003",
        torch.float16,
        torch.float16,
        pypto.DT_FP16,
        pypto.DT_FP16,
        (30, 150),
        (150, 60),
        (30, 60),
        ([32, 32], [32, 32], [64, 64]),
        False,
        None,
        False,
        False,
        marks=[pytest.mark.skip()],
        id="003",
    ),
    pytest.param(
        "matmul_fp16_004",
        torch.float16,
        torch.float16,
        pypto.DT_FP16,
        pypto.DT_FP16,
        (128, 32),
        (32, 100),
        (128, 100),
        ([64, 64], [32, 32], [64, 64]),
        True,
        (1, 100),
        False,
        False,
        marks=[pytest.mark.skip()],
        id="004",
    ),
    pytest.param(
        "matmul_fp16_005",
        torch.float16,
        torch.float16,
        pypto.DT_FP16,
        pypto.DT_FP16,
        (40, 130),
        (40, 64),
        (130, 64),
        ([32, 32], [32, 32], [64, 64]),
        False,
        None,
        True,
        False,
        marks=[pytest.mark.skip()],
        id="005",
    ),
    pytest.param(
        "matmul_fp16_006",
        torch.float16,
        torch.float16,
        pypto.DT_FP16,
        pypto.DT_FP16,
        (5, 80, 64),
        (5, 64, 1),
        (5, 80, 1),
        ([32, 32], [64, 64], [16, 16]),
        False,
        None,
        False,
        False,
        marks=[pytest.mark.skip()],
        id="006",
    ),
    pytest.param(
        "matmul_fp16_007",
        torch.float16,
        torch.float16,
        pypto.DT_FP16,
        pypto.DT_FP16,
        (16, 1, 64),
        (16, 64, 64),
        (16, 1, 64),
        ([16, 16], [64, 64], [16, 16]),
        False,
        None,
        False,
        True,
        marks=[pytest.mark.skip()],
        id="007",
    ),
    pytest.param(
        "matmul_fp16_008",
        torch.float16,
        torch.float16,
        pypto.DT_FP16,
        pypto.DT_FP16,
        (2, 16, 129, 64),
        (2, 16, 64, 35),
        (2, 16, 129, 35),
        ([16, 16], [64, 64], [32, 32]),
        False,
        None,
        False,
        False,
        marks=[pytest.mark.skip()],
        id="008",
    ),
    pytest.param(
        "matmul_fp16_009",
        torch.float16,
        torch.float16,
        pypto.DT_FP16,
        pypto.DT_FP16,
        (2, 8, 80, 160),
        (2, 8, 160, 30),
        (2, 8, 80, 30),
        ([64, 64], [64, 64], [32, 32]),
        False,
        None,
        False,
        False,
        marks=[pytest.mark.skip()],
        id="009",
    ),
    pytest.param(
        "matmul_fp16_010",
        torch.float16,
        torch.float16,
        pypto.DT_FP16,
        pypto.DT_FP16,
        (1, 4, 60, 80),
        (1, 4, 32, 60),
        (1, 4, 80, 32),
        ([64, 64], [32, 32], [32, 32]),
        False,
        None,
        True,
        True,
        marks=[pytest.mark.skip()],
        id="010",
    ),
    pytest.param(
        "matmul_s8s8_001",
        torch.int8,
        torch.int32,
        pypto.DT_INT8,
        pypto.DT_INT32,
        (16, 32),
        (32, 16),
        (16, 16),
        ([32, 32], [32, 32], [32, 32]),
        False,
        None,
        False,
        False,
        marks=[pytest.mark.skip()],
        id="011",
    ),
    pytest.param(
        "matmul_s8s8_002",
        torch.int8,
        torch.int32,
        pypto.DT_INT8,
        pypto.DT_INT32,
        (16, 32),
        (32, 16),
        (16, 16),
        ([32, 32], [32, 32], [32, 32]),
        True,
        (1, 16),
        False,
        False,
        marks=[pytest.mark.skip()],
        id="012",
    ),
    pytest.param(
        "matmul_s8s8_003",
        torch.int8,
        torch.int32,
        pypto.DT_INT8,
        pypto.DT_INT32,
        (16, 32, 64),
        (16, 64, 16),
        (16, 32, 16),
        ([32, 32], [64, 64], [32, 32]),
        False,
        None,
        False,
        False,
        marks=[pytest.mark.skip()],
        id="013",
    ),
    pytest.param(
        "matmul_s8s8_004",
        torch.int8,
        torch.int32,
        pypto.DT_INT8,
        pypto.DT_INT32,
        (2, 16, 32, 64),
        (2, 16, 64, 32),
        (2, 16, 64, 64),
        ([32, 32], [32, 32], [32, 32]),
        False,
        None,
        True,
        True,
        marks=[pytest.mark.skip()],
        id="014",
    ),
    pytest.param("matmul_s8s8_gemv_001", torch.int8, torch.int32,
                 pypto.DT_INT8, pypto.DT_INT32,
                 (1, 16), (16, 16), (1, 16),
                 ([32, 32], [32, 32], [32, 32]),
                 False, None, False, False, marks=[pytest.mark.skip()], id="015"),
    pytest.param("matmul_s8s8_gemv_002", torch.int8, torch.int32,
                 pypto.DT_INT8, pypto.DT_INT32,
                 (1, 1024), (64, 1024), (1, 64),
                 ([32, 32], [32, 32], [32, 32]),
                 False, None, False, True, marks=[], id="016"),
    pytest.param("matmul_s8s8_gemv_003", torch.int8, torch.int32,
                 pypto.DT_INT8, pypto.DT_INT32,
                 (3, 1, 130), (3, 130, 32), (3, 1, 32),
                 ([16, 16], [32, 32], [32, 32]),
                 False, None, False, False, marks=[pytest.mark.skip()], id="017"),
    pytest.param("matmul_s8s8_gemv_004", torch.int8, torch.int32,
                 pypto.DT_INT8, pypto.DT_INT32,
                 (1024, 1), (1024, 32), (1, 32),
                 ([32, 32], [64, 64], [32, 32]),
                 True, (1, 32), True, False, marks=[pytest.mark.skip()], id="018"),
    pytest.param("matmul_s8s8_gemv_005", torch.int8, torch.int32,
                 pypto.DT_INT8, pypto.DT_INT32,
                 (3, 1, 32), (3, 64, 32), (3, 1, 64),
                 ([32, 32], [32, 32], [32, 32]),
                 False, None, False, True, marks=[pytest.mark.skip()], id="019"),
    pytest.param("matmul_s8s8_gemv_006", torch.int8, torch.int32,
                 pypto.DT_INT8, pypto.DT_INT32,
                 (2, 3, 1, 32), (2, 3, 32, 33), (2, 3, 1, 33),
                 ([32, 32], [32, 32], [32, 32]),
                 False, None, False, False, marks=[pytest.mark.skip()], id="020"),
    pytest.param("matmul_s8s8_gemv_007", torch.int8, torch.int32,
                 pypto.DT_INT8, pypto.DT_INT32,
                 (2, 2, 1, 1024), (2, 2, 1024, 32), (2, 2, 1, 32),
                 ([32, 32], [32, 32], [32, 32]),
                 False, None, False, False, marks=[pytest.mark.skip()], id="021"),
    pytest.param("matmul_s8s8_gemv_008", torch.int8, torch.int32,
                 pypto.DT_INT8, pypto.DT_INT32,
                 (3, 1, 130), (3, 130, 32), (3, 1, 32),
                 ([1, 1], [256, 256], [32, 32]),
                 False, None, False, False, marks=[pytest.mark.skip()], id="022"),
    pytest.param("matmul_s8s8_gemv_009", torch.int8, torch.int32,
                 pypto.DT_INT8, pypto.DT_INT32,
                 (1, 130), (130, 64), (1, 64),
                 ([1, 1], [512, 512], [32, 32]),
                 False, None, False, False, marks=[pytest.mark.skip()], id="023"),
    pytest.param("matmul_s8s8_gemv_010", torch.int8, torch.int32,
                 pypto.DT_INT8, pypto.DT_INT32,
                 (4, 1, 16), (4, 16, 32), (4, 1, 32),
                 ([1, 1], [256, 256], [32, 32]),
                 False, None, False, False, marks=[pytest.mark.skip()], id="024"),
    pytest.param("matmul_s8s8_gemv_011", torch.int8, torch.int32,
                 pypto.DT_INT8, pypto.DT_INT32,
                 (1, 128), (128, 32), (1, 32),
                 ([32, 64], [32, 64], [32, 32]),
                 False, None, False, False, marks=[pytest.mark.skip()], id="025"),
    pytest.param("matmul_s8s8_gemv_012", torch.int8, torch.int32,
                 pypto.DT_INT8, pypto.DT_INT32,
                 (3, 1, 128), (3, 128, 32), (3, 1, 32),
                 ([32, 64], [32, 64], [32, 32]),
                 False, None, False, False, marks=[pytest.mark.skip()], id="026"),
    pytest.param("matmul_s8s8_gemv_013", torch.int8, torch.int32,
                 pypto.DT_INT8, pypto.DT_INT32,
                 (2, 2, 1, 128), (2, 2, 128, 32), (2, 2, 1, 32),
                 ([32, 64], [32, 64], [32, 32]),
                 False, None, False, False, marks=[pytest.mark.skip()], id="027"),

    pytest.param("matmul_fp16_gemv_001", torch.float16, torch.float16,
                 pypto.DT_FP16, pypto.DT_FP16,
                 (1, 16), (16, 16), (1, 16),
                 ([16, 16], [16, 16], [16, 16]),
                 False, None, False, False, marks=[pytest.mark.skip()], id="028"),
    pytest.param("matmul_fp16_gemv_002", torch.float16, torch.float16,
                 pypto.DT_FP16, pypto.DT_FP16,
                 (3, 1, 32), (3, 32, 32), (3, 1, 32),
                 ([16, 16], [16, 16], [16, 16]),
                 False, None, False, False, marks=[pytest.mark.skip()], id="029"),
    pytest.param("matmul_fp16_gemv_003", torch.float16, torch.float16,
                 pypto.DT_FP16, pypto.DT_FP16,
                 (4, 1, 130), (4, 130, 16), (4, 1, 16),
                 ([16, 16], [32, 32], [16, 16]),
                 False, None, False, False, marks=[pytest.mark.skip()], id="030"),
    pytest.param("matmul_fp16_gemv_004", torch.float16, torch.float16,
                 pypto.DT_FP16, pypto.DT_FP16,
                 (1, 16), (16, 130), (1, 130),
                 ([16, 16], [16, 16], [32, 32]),
                 False, None, False, False, marks=[pytest.mark.skip()], id="031"),
    pytest.param("matmul_fp16_gemv_005", torch.float16, torch.float16,
                 pypto.DT_FP16, pypto.DT_FP16,
                 (2, 2, 1, 130), (2, 2, 130, 130), (2, 2, 1, 130),
                 ([32, 32], [32, 32], [32, 32]),
                 False, None, False, False, marks=[pytest.mark.skip()], id="032"),
    pytest.param("matmul_fp16_gemv_006", torch.float16, torch.float16,
                 pypto.DT_FP16, pypto.DT_FP16,
                 (1, 1), (1, 16), (1, 16),
                 ([16, 16], [16, 16], [16, 16]),
                 False, None, False, False, marks=[pytest.mark.skip()], id="033"),
    pytest.param("matmul_fp16_gemv_007", torch.float16, torch.float16,
                 pypto.DT_FP16, pypto.DT_FP16,
                 (1, 16), (16, 1), (1, 1),
                 ([16, 16], [16, 16], [16, 16]),
                 False, None, False, False, marks=[pytest.mark.skip()], id="034"),
    pytest.param("matmul_fp16_gemv_008", torch.float16, torch.float16,
                 pypto.DT_FP16, pypto.DT_FP16,
                 (3, 1, 17), (3, 17, 17), (3, 1, 17),
                 ([16, 16], [16, 16], [16, 16]),
                 False, None, False, False, marks=[pytest.mark.skip()], id="035"),
    pytest.param("matmul_fp16_gemv_009", torch.float16, torch.float16,
                 pypto.DT_FP16, pypto.DT_FP16,
                 (1, 8), (8, 8), (1, 8),
                 ([16, 16], [16, 16], [16, 16]),
                 False, None, False, False, marks=[pytest.mark.skip()], id="036"),
    pytest.param("matmul_fp16_gemv_010", torch.float16, torch.float16,
                 pypto.DT_FP16, pypto.DT_FP16,
                 (2, 3, 1, 64), (2, 3, 64, 64), (2, 3, 1, 64),
                 ([32, 32], [32, 32], [32, 32]),
                 False, None, False, False, marks=[pytest.mark.skip()], id="037"),
    pytest.param("matmul_fp16_gemv_011", torch.float16, torch.float16,
                 pypto.DT_FP16, pypto.DT_FP16,
                 (1, 150), (150, 60), (1, 60),
                 ([32, 32], [32, 32], [32, 32]),
                 False, None, False, False, marks=[pytest.mark.skip()], id="038"),
    pytest.param("matmul_fp16_gemv_012", torch.float16, torch.float16,
                 pypto.DT_FP16, pypto.DT_FP16,
                 (3, 1, 1024), (3, 1024, 32), (3, 1, 32),
                 ([32, 32], [32, 32], [32, 32]),
                 False, None, False, False, marks=[pytest.mark.skip()], id="039"),
    pytest.param("matmul_fp16_gemv_013", torch.float16, torch.float16,
                 pypto.DT_FP16, pypto.DT_FP16,
                 (1, 16), (16, 1024), (1, 1024),
                 ([16, 16], [16, 16], [16, 16]),
                 False, None, False, False, marks=[pytest.mark.skip()], id="040"),
    pytest.param("matmul_fp16_gemv_014", torch.float16, torch.float16,
                 pypto.DT_FP16, pypto.DT_FP16,
                 (4, 1, 64), (4, 64, 64), (4, 1, 64),
                 ([64, 64], [32, 32], [32, 32]),
                 False, None, False, False, marks=[pytest.mark.skip()], id="041"),
    pytest.param("matmul_fp16_gemv_015", torch.float16, torch.float16,
                 pypto.DT_FP16, pypto.DT_FP16,
                 (3, 1, 32), (3, 64, 32), (3, 1, 64),
                 ([16, 16], [16, 16], [16, 16]),
                 False, None, False, True, marks=[pytest.mark.skip()], id="042"),
    pytest.param("matmul_fp16_gemv_016", torch.float16, torch.float16,
                 pypto.DT_FP16, pypto.DT_FP16,
                 (2, 2, 64, 1), (2, 2, 64, 16), (2, 2, 1, 16),
                 ([16, 16], [64, 64], [16, 16]),
                 False, None, True, False, marks=[pytest.mark.skip()], id="043"),
    pytest.param("matmul_fp16_gemv_017", torch.float16, torch.float16,
                 pypto.DT_FP16, pypto.DT_FP16,
                 (128, 1), (16, 128), (1, 16),
                 ([16, 16], [128, 128], [16, 16]),
                 False, None, True, True, marks=[pytest.mark.skip()], id="044"),
    pytest.param("matmul_fp16_gemv_018", torch.float16, torch.float16,
                 pypto.DT_FP16, pypto.DT_FP16,
                 (3, 1, 32), (3, 32, 32), (3, 1, 32),
                 ([16, 16], [16, 16], [16, 16]),
                 True, (1, 32), False, False, marks=[pytest.mark.skip()], id="045"),
    pytest.param("matmul_fp16_gemv_019", torch.float16, torch.float16,
                 pypto.DT_FP16, pypto.DT_FP16,
                 (4, 1, 130), (4, 130, 16), (4, 1, 16),
                 ([16, 16], [32, 32], [16, 16]),
                 True, (1, 16), False, False, marks=[pytest.mark.skip()], id="046"),
    pytest.param("matmul_fp16_gemv_020", torch.float16, torch.float16,
                 pypto.DT_FP16, pypto.DT_FP16,
                 (3, 1, 32), (3, 64, 32), (3, 1, 64),
                 ([16, 16], [16, 16], [16, 16]),
                 True, (1, 64), False, True, marks=[pytest.mark.skip()], id="047"),
    pytest.param("matmul_fp16_gemv_021", torch.float16, torch.float16,
                 pypto.DT_FP16, pypto.DT_FP16,
                 (2, 1, 32), (2, 32, 32), (2, 1, 32),
                 ([16, 16], [16, 16], [16, 16]),
                 False, None, False, False, marks=[pytest.mark.skip()], id="048"),
    pytest.param("matmul_fp16_gemv_022", torch.float16, torch.float16,
                 pypto.DT_FP16, pypto.DT_FP16,
                 (2, 2, 1, 16), (2, 2, 16, 16), (2, 2, 1, 16),
                 ([16, 16], [16, 16], [16, 16]),
                 False, None, False, False, marks=[pytest.mark.skip()], id="049"),
    pytest.param("matmul_fp16_gemv_023", torch.float16, torch.float16,
                 pypto.DT_FP16, pypto.DT_FP16,
                 (1, 130), (130, 32), (1, 32),
                 ([1, 1], [256, 256], [16, 16]),
                 False, None, False, False, marks=[pytest.mark.skip()], id="050"),
    pytest.param("matmul_fp16_gemv_024", torch.float16, torch.float16,
                 pypto.DT_FP16, pypto.DT_FP16,
                 (5, 1, 130), (5, 130, 64), (5, 1, 64),
                 ([1, 1], [512, 512], [16, 16]),
                 False, None, False, False, marks=[pytest.mark.skip()], id="051"),
    pytest.param("matmul_fp16_gemv_025", torch.float16, torch.float16,
                 pypto.DT_FP16, pypto.DT_FP16,
                 (1, 16), (16, 32), (1, 32),
                 ([1, 1], [256, 256], [16, 16]),
                 False, None, False, False, marks=[pytest.mark.skip()], id="052"),
    pytest.param("matmul_fp16_gemv_026", torch.float16, torch.float16,
                 pypto.DT_FP16, pypto.DT_FP16,
                 (1, 128), (128, 32), (1, 32),
                 ([32, 64], [32, 64], [16, 16]),
                 False, None, False, False, marks=[pytest.mark.skip()], id="053"),
    pytest.param("matmul_fp16_gemv_027", torch.float16, torch.float16,
                 pypto.DT_FP16, pypto.DT_FP16,
                 (3, 1, 128), (3, 128, 32), (3, 1, 32),
                 ([32, 64], [32, 64], [16, 16]),
                 False, None, False, False, marks=[pytest.mark.skip()], id="054"),
    pytest.param("matmul_fp16_gemv_028", torch.float16, torch.float16,
                 pypto.DT_FP16, pypto.DT_FP16,
                 (2, 2, 1, 128), (2, 2, 128, 32), (2, 2, 1, 32),
                 ([32, 64], [32, 64], [16, 16]),
                 False, None, False, False, marks=[pytest.mark.skip()], id="055"),
]


def run_matmul_test(
    kernels,
    kernel_name,
    torch_in_dtype,
    torch_out_dtype,
    shape_a,
    shape_b,
    shape_c,
    cube_tile_shape,
    has_bias,
    bias_shape,
    a_trans,
    b_trans,
    soc_version,
):
    """Run a single matmul kernel test with given kernels dict."""
    device = "cpu"

    if torch_in_dtype == torch.float16:
        a = torch.rand(shape_a, dtype=torch_in_dtype, device=device)
        b = torch.rand(shape_b, dtype=torch_in_dtype, device=device)
        c = torch.rand(shape_c, dtype=torch_out_dtype, device=device)
    else:
        a = torch.randint(-128, 127, shape_a, dtype=torch.int8)
        b = torch.randint(-128, 127, shape_b, dtype=torch.int8)
        c = torch.zeros(shape_c).to(dtype=torch.int32)

    a_new = a
    if a_trans:
        if a.dim() == 2:
            a_new = a.permute(1, 0)
        elif a.dim() == 3:
            a_new = a.permute(0, 2, 1)
        else:
            a_new = a.permute(0, 1, 3, 2)

    b_new = b
    if b_trans:
        if b.dim() == 2:
            b_new = b.permute(1, 0)
        elif b.dim() == 3:
            b_new = b.permute(0, 2, 1)
        else:
            b_new = b.permute(0, 1, 3, 2)

    if has_bias:
        if torch_in_dtype == torch.float16:
            # X90 硬件要求 bias 以 32 位 fp32 格式传入（低 16 位有效，高 16 位无效）
            bias = torch.rand(bias_shape, dtype=torch_out_dtype, device=device)
            gold_bias = bias
            if soc_version == "KirinX90" and torch_out_dtype == torch.float16:
                bias = bias.view(torch.int16).to(torch.int32).bitwise_and(0xFFFF).view(torch.float32)
        else:
            bias = torch.randint(-128, 127, bias_shape, dtype=torch.int32)
            gold_bias = bias

        kernels[kernel_name](a, b, bias, c)

        if torch_out_dtype == torch.int32:
            golden_c = a_new.to(torch.int32) @ b_new.to(torch.int32) + gold_bias
        else:
            golden_c = a_new @ b_new + gold_bias
    else:
        kernels[kernel_name](a, b, c)

        if torch_out_dtype == torch.int32:
            golden_c = a_new.to(torch.int32) @ b_new.to(torch.int32)
        else:
            golden_c = a_new @ b_new

    check_nan(c, name=kernel_name)
    cos_value = compare_cos(np.array(c.cpu()), np.array(golden_c.cpu()))
    cos_value = abs(cos_value)
    if cos_value < 0.9999:
        raise AssertionError(f"{kernel_name}: cos_value {cos_value} < 0.9999")


def create_test_matmul_module(soc_version):
    """Create a test module for matmul with specified soc_version."""
    kernels = {
        p.values[0]: get_matmul_kernel(
            soc_version,
            p.values[0],
            p.values[3],  # pypto_in_dtype
            p.values[4],  # pypto_out_dtype
            p.values[8],  # cube_tile_shape
            p.values[9],  # has_bias
            p.values[10],  # bias_shape
            p.values[11],  # a_trans
            p.values[12],  # b_trans
        )
        for p in TEST_CASES
    }
    return kernels, lambda: run_matmul_test(
        kernels, None, None, None, None, None, None, None, False, None, False, False, soc_version
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
