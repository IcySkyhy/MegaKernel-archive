#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

"""
Generate golden data for tpushpop_cv_nosplit tests.

Kernel computation:
    out[M, N] = matmul(A[M, K], B[K, N]) + bias[M, N]

The Cube core computes matmul and pushes the full CASE_TILE_M x N accumulator
tile to a single Vec core (AIV0) via TPUSH with TileSplitAxis::TILE_NO_SPLIT.
AIV0 pops the tile and adds the bias with TADD.

Files written per test case directory:
    x1_gm.bin   - input matrix A (InT)
    x2_gm.bin   - input matrix B (InT)
    bias_gm.bin - bias matrix    (float32, shape M x N)
    golden.bin  - expected output (float32, shape M x N)
"""

import os
import numpy as np

np.random.seed(42)


class CaseParams:
    def __init__(self, in_dtype, m, k, n, use_bias=True):
        self.in_dtype = in_dtype  # numpy dtype for A and B
        self.m = m
        self.k = k
        self.n = n
        self.use_bias = use_bias


def gen_golden_data(params: CaseParams):
    m, k, n = params.m, params.k, params.n
    in_dtype = params.in_dtype

    x1_gm = np.random.randint(-5, 5, [m, k]).astype(in_dtype)
    x2_gm = np.random.randint(-5, 5, [k, n]).astype(in_dtype)

    # Bias has the same M x N shape as the output tile(element - wise TADD).
    bias_gm = np.random.randint(-3, 3, [m, n]).astype(np.float32)

    # Golden : matmul in float32 + optional bias.
    golden = np.matmul(x1_gm.astype(np.float32), x2_gm.astype(np.float32))
    if params.use_bias:
        golden = golden + bias_gm
    golden = golden.astype(np.float32)

    x1_gm.tofile("./x1_gm.bin")
    x2_gm.tofile("./x2_gm.bin")
    bias_gm.tofile("./bias_gm.bin")
    golden.tofile("./golden.bin")


if __name__ == "__main__":
    # Case names must match the test suite and test case names in main.cpp so
    # that GetGoldenDir() resolves to the correct directory.
    case_name_list = [
        "TPushPopCVNoSplitTest.case1_half_single_tile",
        "TPushPopCVNoSplitTest.case2_half_two_tiles",
        "TPushPopCVNoSplitTest.case3_float_single_tile",
        "TPushPopCVNoSplitTest.case4_float_static_acc_valid_shape_strip",
        "TPushPopCVNoSplitTest.case5_float_dynamic_acc_valid_shape_strip",
    ]

    # Parameters mirror kernel instantiations.
    # key = 1 : half->float, TOTAL_M = 16, K = 32, N = 32
    # key = 2 : half->float, TOTAL_M = 32, K = 32, N = 32
    # key = 3 : float->float, TOTAL_M = 16, K = 32, N = 32
    # key = 4 : float->float, M = 32, K = 32, N = 128, static Acc ValidRow strip H = 16, no bias
    # key = 5 : float->float, M = 32, K = 32, N = 128, dynamic Acc valid shape H = 16, no bias
    case_params_list = [
        CaseParams(np.float16, m=16, k=32, n=32),
        CaseParams(np.float16, m=32, k=32, n=32),
        CaseParams(np.float32, m=16, k=32, n=32),
        CaseParams(np.float32, m=32, k=32, n=128, use_bias=False),
        CaseParams(np.float32, m=32, k=32, n=128, use_bias=False),
    ]

    for case_name, params in zip(case_name_list, case_params_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(params)
        os.chdir(original_dir)
