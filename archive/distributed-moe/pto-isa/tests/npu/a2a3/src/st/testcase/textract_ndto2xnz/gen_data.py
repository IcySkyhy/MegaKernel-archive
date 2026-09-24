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
    for idx, win_spec in enumerate(wins):
        rows, cols, ir, ic = win_spec[:4]
        trows, tcols = (win_spec[4], win_spec[5]) if len(win_spec) >= 6 else (rows, cols)
        if rows == 1 and cols == 1:
            golden = src[ir : ir + rows, ic : ic + cols, :].reshape(-1)
        else:
            padded = np.zeros((trows, tcols, esize), dtype=np.uint8)
            padded[:rows, :cols, :] = src[ir : ir + rows, ic : ic + cols, :]
            golden = nd_to_nz_bytes(padded, trows, tcols, esize)
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
    Case("case_half", 2, 64, 128, [(32, 64, 0, 0), (32, 64, 32, 64)]),
    Case("case_float", 4, 64, 128, [(32, 64, 0, 0), (32, 64, 32, 64)]),
    Case("case_bf16", 2, 64, 128, [(32, 64, 0, 0), (32, 64, 32, 64)]),
    Case("case_int8", 1, 64, 128, [(32, 64, 0, 0), (32, 64, 32, 64)]),
    Case("case_int32", 4, 64, 128, [(32, 64, 0, 0), (32, 64, 32, 64)]),
    Case("case_half_idx", 2, 64, 128, [(32, 64, 16, 16), (32, 64, 0, 48)]),
    Case("case_int8_idx", 1, 64, 128, [(32, 64, 16, 32), (32, 64, 0, 0)]),
    Case("case_half_unaligned", 2, 64, 128, [(32, 64, 0, 0), (32, 64, 16, 8)]),
    Case("case_int8_unaligned", 1, 64, 128, [(32, 64, 0, 0), (32, 64, 16, 8)]),
    Case("case_int8_odd", 1, 64, 128, [(32, 64, 0, 1), (32, 64, 0, 3)]),
    Case("case_int8_oddvalid", 1, 64, 128, [(32, 63, 0, 0, 32, 64), (32, 61, 0, 2, 32, 64)]),
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
