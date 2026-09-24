#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------
import os
import numpy as np

np.random.seed(42)


def nd_to_zn(data, rows, cols, k0):
    """Convert ND (row-major) [rows, cols] layout to ZN fractal layout.

    ZN layout: [rows/k0, cols/16, 16, k0] — the Right/NT cube operand. Each output
    fractal is the transpose of a k0 x 16 source slice:
        D[k1][n1][j][i] = S[k1*k0 + i][n1*16 + j]
    Requires rows % k0 == 0 and cols % 16 == 0.
    """
    a = data.reshape(rows // k0, k0, cols // 16, 16)
    zn = a.transpose(0, 2, 3, 1).reshape(-1)
    return zn


def gen_golden(case_name, dtype_name, rows, cols):
    if dtype_name == "hif8":
        k0 = 32
        dt = np.uint8
    elif dtype_name == "half":
        k0 = 16
        dt = np.uint16
    elif dtype_name == "b32":
        k0 = 8
        dt = np.uint32
    else:
        raise ValueError(f"unknown dtype {dtype_name}")
    input_arr = np.random.randint(0, 1 << (8 * dt().itemsize), size=(rows, cols), dtype=dt)
    input_arr.tofile("input_arr.bin")
    golden = nd_to_zn(input_arr, rows, cols, k0)
    golden.tofile("golden.bin")


class CaseParams:
    def __init__(self, dtype_name, rows, cols):
        self.dtype_name = dtype_name
        self.rows = rows
        self.cols = cols


if __name__ == "__main__":
    case_name_list = [
        "TMovNd2ZnTest.case_hif8_32x32",
        "TMovNd2ZnTest.case_hif8_32x64",
        "TMovNd2ZnTest.case_hif8_64x64",
        "TMovNd2ZnTest.case_hif8_128x128",
        "TMovNd2ZnTest.case_half_32x32",
        "TMovNd2ZnTest.case_half_32x64",
        "TMovNd2ZnTest.case_half_64x64",
        "TMovNd2ZnTest.case_half_128x128",
        "TMovNd2ZnTest.case_b32_32x32",
        "TMovNd2ZnTest.case_b32_32x64",
        "TMovNd2ZnTest.case_b32_64x64",
        "TMovNd2ZnTest.case_b32_128x128",
    ]

    case_params_list = [
        CaseParams("hif8", 32, 32),
        CaseParams("hif8", 32, 64),
        CaseParams("hif8", 64, 64),
        CaseParams("hif8", 128, 128),
        CaseParams("half", 32, 32),
        CaseParams("half", 32, 64),
        CaseParams("half", 64, 64),
        CaseParams("half", 128, 128),
        CaseParams("b32", 32, 32),
        CaseParams("b32", 32, 64),
        CaseParams("b32", 64, 64),
        CaseParams("b32", 128, 128),
    ]

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)

        original_dir = os.getcwd()
        os.chdir(case_name)

        gen_golden(case_name, case_params_list[i].dtype_name, case_params_list[i].rows, case_params_list[i].cols)

        os.chdir(original_dir)
