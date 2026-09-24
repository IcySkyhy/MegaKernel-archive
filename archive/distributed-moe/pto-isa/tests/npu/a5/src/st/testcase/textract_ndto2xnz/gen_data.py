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

C0_BYTES = 32
N0 = 16


def nd_to_nz_bytes(win, rows, cols, esize):
    c0 = C0_BYTES // esize
    c1 = cols // c0
    n1 = rows // N0
    nz = win.reshape(n1, N0, c1, c0, esize).transpose(2, 0, 1, 3, 4).reshape(-1)
    return nz


def gen_golden(src_rows, src_cols, esize, wins):
    src = np.random.randint(0, 256, size=(src_rows, src_cols, esize), dtype=np.uint8)
    src.tofile("input_arr.bin")
    for idx, (rows, cols, ir, ic) in enumerate(wins):
        win = src[ir : ir + rows, ic : ic + cols, :]
        if rows == 1 and cols == 1:
            golden = win.reshape(-1)
        else:
            golden = nd_to_nz_bytes(win, rows, cols, esize)
        golden.tofile(f"golden{idx}.bin")


class Case:
    def __init__(self, name, esize, src_rows, src_cols, wins):
        self.name = name
        self.esize = esize
        self.src_rows = src_rows
        self.src_cols = src_cols
        self.wins = wins


# win entry: (rows, cols, indexRow, indexCol)
CASES = [
    Case("case_half_aligned", 2, 64, 128, [(32, 64, 0, 0), (32, 64, 32, 64)]),
    Case("case_half_unaligned", 2, 64, 128, [(32, 64, 0, 0), (32, 64, 16, 8)]),
    Case("case_float_aligned", 4, 64, 128, [(32, 64, 0, 0), (32, 64, 32, 64)]),
    Case("case_float_unaligned", 4, 64, 128, [(32, 64, 0, 0), (32, 64, 16, 4)]),
    Case("case_bf16_aligned", 2, 64, 128, [(32, 64, 0, 0), (32, 64, 32, 64)]),
    Case("case_int8_aligned", 1, 64, 128, [(32, 64, 0, 0), (32, 64, 32, 64)]),
    Case("case_int8_unaligned", 1, 64, 128, [(32, 64, 0, 0), (32, 64, 16, 16)]),
    Case("case_int32_aligned", 4, 64, 128, [(32, 64, 0, 0), (32, 64, 32, 64)]),
    Case("case_hif8_aligned", 1, 64, 128, [(32, 64, 0, 0), (32, 64, 32, 64)]),
    Case("case_fp8e4m3_aligned", 1, 64, 128, [(32, 64, 0, 0), (32, 64, 32, 64)]),
    Case("case_fp8e5m2_aligned", 1, 64, 128, [(32, 64, 0, 0), (32, 64, 32, 64)]),
    Case("case_fp8e8m0_aligned", 1, 64, 128, [(32, 64, 0, 0), (32, 64, 32, 64)]),
    Case("case_half_1x1", 2, 64, 128, [(1, 1, 0, 0), (1, 1, 5, 7)]),
    Case("case_float_1x1", 4, 64, 128, [(1, 1, 0, 0), (1, 1, 10, 3)]),
    Case("case_int8_1x1", 1, 64, 128, [(1, 1, 0, 0), (1, 1, 20, 17)]),
]


if __name__ == "__main__":
    original_dir = os.getcwd()
    for case in CASES:
        case_dir = f"TExtractNd2xNzTest.{case.name}"
        if not os.path.exists(case_dir):
            os.makedirs(case_dir)
        os.chdir(case_dir)
        gen_golden(case.src_rows, case.src_cols, case.esize, case.wins)
        os.chdir(original_dir)
