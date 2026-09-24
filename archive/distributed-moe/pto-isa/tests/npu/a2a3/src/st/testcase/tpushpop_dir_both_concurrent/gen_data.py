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

import os
import numpy as np

np.random.seed(19)


def gen_golden_data(case_name, case_params):
    """
    DIR_BOTH concurrent test: the two directions are IN FLIGHT AT THE SAME TIME.

    Unlike tpushpop_dir_both, the cube's matmul takes both operands from GM, so the
    C2V push does not depend on the V2C pop and is issued first. The vector still
    pushes V2C then pops C2V. Both tiles are tileIndex 0 -> both address slot 0.

    Computation:
      C = A + B                       [M, K]   (vector -> cube, V2C; never consumed here)
      E = A @ D                       [M, N]   (cube -> vector, C2V; GM operands only)
      G = E - F                       [M, N]   (golden output)
    """
    m, k, n, dtype = case_params
    srcA = np.random.uniform(-2, 2, [m, k]).astype(dtype)
    srcB = np.random.uniform(-2, 2, [m, k]).astype(dtype)
    srcD = np.random.uniform(-2, 2, [k, n]).astype(dtype)
    srcF = np.random.uniform(-2, 2, [m, n]).astype(dtype)

    srcA.tofile("./srcA_gm.bin")
    srcB.tofile("./srcB_gm.bin")
    srcD.tofile("./srcD_gm.bin")
    srcF.tofile("./srcF_gm.bin")

    e_mat = np.matmul(srcA, srcD).astype(dtype)
    golden = (e_mat - srcF).astype(dtype)
    golden.tofile("./golden.bin")

    # What the CUBE must receive over V2C, put through the same second matmul the kernel
    # applies to it. Checking this alongside the vector's output is what makes the slot-0
    # aliasing observable -- a single-sided check can still pass on an affected build.
    c_mat = (srcA + srcB).astype(dtype)
    golden_cube = np.matmul(c_mat, srcD).astype(dtype)
    golden_cube.tofile("./golden_cube.bin")


if __name__ == "__main__":
    case_name_list = [
        "TPushPopDirBothConcurrentTest.case1_float_dir_both_concurrent",
        "TPushPopDirBothConcurrentTest.case2_float_dir_both_concurrent_left_right",
    ]

    case_params_list = [
        (128, 64, 128, np.float32),
        (128, 64, 128, np.float32),
    ]

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_name, case_params_list[i])
        os.chdir(original_dir)
