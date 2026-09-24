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
from dataclasses import dataclass

import numpy as np


def nd_to_zn(arr, rows, cols, dtype_size):
    inner_row = 32 // dtype_size
    inner_col = 16
    return arr.reshape(rows // inner_row, inner_row, cols // inner_col, inner_col).transpose(0, 2, 3, 1)


def rand_data(dtype, shape):
    if dtype == np.int8:
        return np.random.randint(-128, 127, size=shape).astype(dtype)
    if dtype == np.int32:
        return np.random.randint(-128, 128, size=shape).astype(dtype)
    if dtype in (np.uint16, np.int16):
        return np.random.randint(0, 65536, size=shape, dtype=np.uint16)
    if dtype == np.uint8:
        return np.random.randint(0, 256, size=shape, dtype=np.uint8)
    return np.random.uniform(-10, 10, size=shape).astype(dtype)


def run_case(name, gen_fn, *args):
    os.makedirs(name, exist_ok=True)
    orig = os.getcwd()
    os.chdir(name)
    gen_fn(*args)
    os.chdir(orig)


def gen_zn(dtype, rows, cols):
    ds = np.dtype(dtype).itemsize
    arr = rand_data(dtype, (rows, cols))
    nd_to_zn(arr, rows, cols, ds).tofile("input_arr.bin")
    nd_to_zn(arr, rows, cols, ds).tofile("golden_output.bin")


def gen_zn_fp4(rows, cols):
    byte_rows = rows // 2
    arr = np.random.randint(0, 256, size=(byte_rows, cols), dtype=np.uint8)
    arr.tofile("input_arr.bin")
    arr.tofile("golden_output.bin")


@dataclass
class ZnOffsetParams:
    dtype: object
    valid_row: int
    valid_col: int
    src_cols: int
    dst_rows: int
    dst_cols: int
    idx_row: int
    idx_col: int


def gen_zn_offset(p):
    ds = np.dtype(p.dtype).itemsize
    bg = rand_data(p.dtype, (p.dst_rows, p.dst_cols))

    data = rand_data(p.dtype, (p.valid_row, p.valid_col))
    src_region = np.zeros((p.valid_row, p.src_cols), dtype=p.dtype)
    src_region[:, : p.valid_col] = data

    bg_zn = nd_to_zn(bg, p.dst_rows, p.dst_cols, ds).flatten()
    src_zn = nd_to_zn(src_region, p.valid_row, p.src_cols, ds).flatten()
    np.concatenate([bg_zn, src_zn]).tofile("input_arr.bin")

    result = bg.copy()
    r_end = p.idx_row + p.valid_row
    c_end = p.idx_col + p.valid_col
    result[p.idx_row : r_end, p.idx_col : c_end] = data
    nd_to_zn(result, p.dst_rows, p.dst_cols, ds).tofile("golden_output.bin")


if __name__ == "__main__":
    cases = [
        ("TInsertZNTest.case_zn_1", gen_zn, np.float16, 16, 16),
        ("TInsertZNTest.case_zn_2", gen_zn, np.float16, 16, 32),
        ("TInsertZNTest.case_zn_3", gen_zn, np.float32, 8, 16),
        ("TInsertZNTest.case_zn_4", gen_zn, np.float32, 16, 32),
        ("TInsertZNTest.case_zn_5", gen_zn, np.int32, 8, 16),
        ("TInsertZNTest.case_zn_6", gen_zn, np.int8, 32, 32),
        ("TInsertZNTest.case_zn_7", gen_zn, np.float16, 32, 64),
        ("TInsertZNTest.case_zn_8", gen_zn, np.uint16, 16, 32),
        ("TInsertZNTest.case_zn_9", gen_zn, np.uint8, 32, 32),
        ("TInsertZNTest.case_zn_10", gen_zn, np.uint8, 32, 32),
        ("TInsertZNTest.case_zn_11", gen_zn, np.uint8, 32, 64),
        ("TInsertZNTest.case_zn_12", gen_zn, np.uint8, 32, 32),
        ("TInsertZNTest.case_zn_fp4_1", gen_zn_fp4, 64, 32),
        ("TInsertZNTest.case_zn_fp4_2", gen_zn_fp4, 64, 32),
        ("TInsertZNTest.case_zn_offset_1", gen_zn_offset, ZnOffsetParams(np.float16, 16, 16, 16, 32, 32, 0, 0)),
        ("TInsertZNTest.case_zn_offset_2", gen_zn_offset, ZnOffsetParams(np.float16, 16, 16, 16, 32, 32, 16, 16)),
        ("TInsertZNTest.case_zn_offset_3", gen_zn_offset, ZnOffsetParams(np.float32, 8, 16, 16, 16, 32, 8, 16)),
        ("TInsertZNTest.case_zn_offset_4", gen_zn_offset, ZnOffsetParams(np.float16, 16, 15, 16, 16, 32, 0, 0)),
        ("TInsertZNTest.case_zn_offset_5", gen_zn_offset, ZnOffsetParams(np.int8, 32, 32, 32, 64, 64, 32, 32)),
        ("TInsertZNTest.case_zn_offset_6", gen_zn_offset, ZnOffsetParams(np.float16, 16, 32, 32, 32, 64, 0, 32)),
        ("TInsertZNTest.case_zn_offset_7", gen_zn_offset, ZnOffsetParams(np.float16, 16, 16, 16, 32, 32, 16, 0)),
        ("TInsertZNTest.case_zn_offset_8", gen_zn_offset, ZnOffsetParams(np.float32, 8, 15, 16, 16, 32, 8, 16)),
        ("TInsertZNTest.case_zn_offset_9", gen_zn_offset, ZnOffsetParams(np.float16, 32, 32, 32, 64, 64, 0, 32)),
        ("TInsertZNTest.case_zn_offset_10", gen_zn_offset, ZnOffsetParams(np.float16, 16, 16, 16, 16, 48, 0, 8)),
        ("TInsertZNTest.case_zn_offset_11", gen_zn_offset, ZnOffsetParams(np.float32, 8, 16, 16, 8, 48, 0, 24)),
        ("TInsertZNTest.case_zn_offset_12", gen_zn_offset, ZnOffsetParams(np.float16, 16, 16, 16, 32, 48, 16, 8)),
        ("TInsertZNTest.case_zn_offset_13", gen_zn_offset, ZnOffsetParams(np.float32, 8, 13, 16, 8, 48, 0, 24)),
    ]

    for name, gen_fn, *args in cases:
        run_case(name, gen_fn, *args)
