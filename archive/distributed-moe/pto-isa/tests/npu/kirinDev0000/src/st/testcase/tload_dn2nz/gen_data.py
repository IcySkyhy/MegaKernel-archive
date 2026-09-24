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

np.random.seed(42)


def gen_golden_data(case_name, param):
    m, n = param
    src_type = np.float16

    # Create a known M×N matrix
    matrix = np.arange(1, m * n + 1, dtype=np.int32).reshape(m, n).astype(src_type)

    # GM stores the matrix in col-major (DN) format: matrix.transpose() flattened row-major = [N, M] row-major
    gm_data = matrix.transpose().copy()

    # Golden: the same matrix in row-major (ND) format
    golden = matrix.copy()

    gm_data.tofile("./x1_gm.bin")
    golden.tofile("./golden.bin")


if __name__ == "__main__":
    case_name_list = ["TLOADDN2NZTest.case1", "TLOADDN2NZTest.case2", "TLOADDN2NZTest.case3", "TLOADDN2NZTest.case4"]

    case_params_list = [(16, 16), (32, 32), (32, 64), (64, 96)]

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)

        gen_golden_data(case_name, case_params_list[i])

        os.chdir(original_dir)
