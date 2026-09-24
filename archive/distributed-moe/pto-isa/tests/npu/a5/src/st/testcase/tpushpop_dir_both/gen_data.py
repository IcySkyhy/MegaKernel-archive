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
    """Generate input data and golden output.

    Flow: C = A + B, E = C @ D, golden = E - F
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

    C = (srcA + srcB).astype(dtype)
    E = np.matmul(C, srcD).astype(dtype)
    golden = (E - srcF).astype(dtype)
    golden.tofile("./golden.bin")


if __name__ == "__main__":
    case_name_list = [
        "TPushPopDirBothTest.case1_float_dir_both",
        "TPushPopDirBothTest.case2_float_dir_both_left_right",
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
