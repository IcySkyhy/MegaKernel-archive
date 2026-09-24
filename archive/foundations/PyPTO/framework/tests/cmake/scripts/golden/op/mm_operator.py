#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Matmul Operator 相关用例 Golden 生成逻辑.

本脚本有 2 种执行模式:
1. CI批跑时, 由 cmake/scripts/golden_ctrl.py 调用, 为避免日志过多, 此时 logging 级别为 logging.INFO;
2. 单独调试时, 本脚本单独被调用, 此时 logging 级别为 logging.DEBUG;
"""

import logging
from pathlib import Path
import sys
from typing import List

from ml_dtypes import bfloat16
import numpy as np

if __name__ == "__main__":
    """ 单独调试时配置 """
    # 日志级别
    logging.basicConfig(
        format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s', level=logging.DEBUG
    )
    # 系统 import 路径
    g_src_root: Path = Path(Path(__file__).parent, "../../../../../").resolve()
    logging.debug("SrcRoot: %s", g_src_root)
    g_ctrl_path: Path = Path(g_src_root, "cmake/scripts")
    if str(g_ctrl_path) not in sys.path:
        sys.path.append(str(g_ctrl_path))
    from golden_register import GoldenRegister  # 单独调试 import 失败, 需确认上文中 '系统 import 路径' 配置正确
else:
    from golden_register import GoldenRegister

dtype_f32 = np.float32
dtype_f16 = np.float16
dtype_bf16 = bfloat16
dtype_s8 = np.int8
dtype_s32 = np.int32


def gen_mm_data(m, k, n, dtype, out_dtype, output_dir: Path):
    shape_a = [m, k]
    shape_b = [k, n]
    shape_c = [m, n]
    logging.debug("shape a is %s", shape_a)
    logging.debug("shape b is %s", shape_b)
    logging.debug("shape c is %s", shape_c)
    logging.debug(f"input dtype: {dtype}, output dtype: {out_dtype}")

    a_path = Path(output_dir, 'a.bin')
    b_path = Path(output_dir, "b.bin")
    c_path = Path(output_dir, "c_golden.bin")

    if dtype == dtype_s8:
        a = np.random.randint(-4, 5, shape_a).astype(dtype)
        b = np.random.randint(-4, 5, shape_b).astype(dtype)
        c = np.matmul(a.astype(dtype_s32), b.astype(dtype_s32)).astype(dtype_s32)
        if out_dtype != dtype_s32:
            c = c.astype(out_dtype)
    else:
        a = np.random.uniform(-1, 1, shape_a).astype(dtype)
        b = np.random.uniform(-1, 1, shape_b).astype(dtype)
        c = np.matmul(a.astype(dtype_f32), b.astype(dtype_f32))
        if out_dtype != dtype_f32:
            c = c.astype(out_dtype)
    a.tofile(a_path)
    b.tofile(b_path)
    c.tofile(c_path)


def gen_mm_data_trans(mm_size, dtype, out_dtype, output_dir: Path):
    m = mm_size[0]
    k = mm_size[1]
    n = mm_size[2]
    shape_a = [m, k]
    shape_b = [k, n]
    shape_c = [m, n]
    logging.debug("shape a is %s", shape_a)
    logging.debug("shape b is %s", shape_b)
    logging.debug("shape c is %s", shape_c)
    logging.debug(f"input dtype: {dtype}, output dtype: {out_dtype}")

    a_path = Path(output_dir, 'a.bin')
    b_path = Path(output_dir, "b.bin")
    c_path = Path(output_dir, "c_golden.bin")

    if dtype == dtype_s8:
        a = np.random.randint(-4, 5, shape_a).astype(dtype)
        b = np.random.randint(-4, 5, shape_b).astype(dtype)
        c = np.matmul(a.astype(dtype_s32), b.astype(dtype_s32)).astype(dtype_s32)
        if out_dtype != dtype_s32:
            c = c.astype(out_dtype)
    else:
        a = np.random.uniform(-2, 2, shape_a).astype(dtype)
        b = np.random.uniform(-2, 2, shape_b).astype(dtype)
        c = np.matmul(a.astype(dtype_f32), b.astype(dtype_f32))
        if out_dtype != dtype_f32:
            c = c.astype(out_dtype)
    a.tofile(a_path)
    b = b.transpose(1, 0)
    b.tofile(b_path)
    c.tofile(c_path)


@GoldenRegister.reg_golden_func(
    case_names=[
        "MatmulOnBoardTest.test_mm_float32_64_64_64",
        "MatmulOnBoardTest.test_mm_float32_64_128_128",
        "MatmulOnBoardTest.test_mm_float32_128_128_128",
        "MatmulOnBoardTest.test_mm_float32_32_128_128",
        "MatmulOnBoardTest.test_mm_float32_32_128_64",
        "MatmulOnBoardTest.test_mm_int8_32_128_64",
        "MatmulOnBoardTest.test_mm_int8_32_128_64_bt",
        "MatmulOnBoardTest.test_mm_float_32_128_128",
        "MatmulOnBoardTest.test_mm_float_32_128_128_bt",
        "MatmulOnBoardTest.test_mm_float32_32_192_64",
        "MatmulOnBoardTest.test_mm_float32_2_128_128",
        "MatmulOnBoardTest.test_mm_float32_256_256_256",
        "MatmulOnBoardTest.test_mm_float32_32_512_576",
        "MatmulOnBoardTest.test_mm_float32_32_7168_1536",
        "MatmulOnBoardTest.test_mm_float32_32_1536_6144",
        "MatmulOnBoardTest.test_mm_float32_32_7168_576",
        "MatmulOnBoardTest.test_mm_float16_64_128_128",
        "MatmulOnBoardTest.test_mm_float16_64_256_128",
        "MatmulOnBoardTest.test_mm_float16_32_7168_1536",
        "MatmulOnBoardTest.test_mm_float16_32_1536_6144",
        "MatmulOnBoardTest.test_mm_float16_32_7168_576",
        "MatmulOnBoardTest.test_mm_float16_4_7168_1536",
        "MatmulOnBoardTest.test_mm_float16_4_1536_6144",
        "MatmulOnBoardTest.test_mm_float16_16_7168_2048",
        "MatmulOnBoardTest.test_mm_bfloat16_64_128_128",
        "MatmulOnBoardTest.test_mm_bfloat16_f32_64_128_128",
        "MatmulOnBoardTest.test_mm_unalign_float32_2_128_128",
        "MatmulOnBoardTest.test_mm_unalign_float32_16_35_32",
        "MatmulOnBoardTest.test_mm_unalign_float32_16_32_35",
        "MatmulOnBoardTest.test_mm_float32_64_64_64_bt",
        "MatmulOnBoardTest.test_mm_unalign_float32_8_576_256_bt",
        "MatmulOnBoardTest.test_mm_unalign_float32_8_64_64_bt",
        "MatmulOnBoardTest.test_mm_int8_32_16384_7168",
        "MatmulOnBoardTest.test_mm_int8_32_2048_256",
        "MatmulOnBoardTest.test_mm_float32_64_64_64_acc",
        "MatmulOnBoardTest.test_mm_float32_32_7168_1536_acc",
        "MatmulOnBoardTest.test_mm_float32_32_512_128_acc",
        "MatmulOnBoardTest.test_mm_float32_32_1024_512_acc",
        "CostModelTest.test_mm_float32_64_64_64_bt",
    ]
)
def gen_mm_op_data(case_name: str, output: Path) -> bool:
    a_path = Path(output, "a.bin")
    b_path = Path(output, "b.bin")
    c_path = Path(output, "c_golden.bin")

    complete = a_path.exists() and b_path.exists() and c_path.exists()

    if complete:
        logging.debug("Case(%s), Golden complete.", case_name)
        return True
    else:
        if case_name == "MatmulOnBoardTest.test_mm_float32_64_64_64":
            m, k, n = 64, 64, 64
            gen_mm_data(m, k, n, dtype_f16, dtype_f32, output)
        elif case_name == "MatmulOnBoardTest.test_mm_float32_64_64_64_acc":
            m, k, n = 64, 64, 64
            gen_mm_data(m, k, n, dtype_f16, dtype_f32, output)
        elif case_name == "MatmulOnBoardTest.test_mm_float32_32_7168_1536_acc":
            m, k, n = 32, 7168, 1536
            gen_mm_data(m, k, n, dtype_f16, dtype_f32, output)
        elif case_name == "MatmulOnBoardTest.test_mm_float32_32_512_128_acc":
            m, k, n = 32, 512, 128
            gen_mm_data(m, k, n, dtype_f16, dtype_f32, output)
        elif case_name == "MatmulOnBoardTest.test_mm_float32_32_1024_512_acc":
            m, k, n = 32, 1024, 512
            gen_mm_data(m, k, n, dtype_f16, dtype_f32, output)
        elif case_name == "MatmulOnBoardTest.test_mm_float32_64_128_128":
            m, k, n = 64, 128, 128
            gen_mm_data(m, k, n, dtype_f16, dtype_f32, output)
        elif case_name == "MatmulOnBoardTest.test_mm_float32_128_128_128":
            m, k, n = 128, 128, 128
            gen_mm_data(m, k, n, dtype_f16, dtype_f32, output)
        elif case_name == "MatmulOnBoardTest.test_mm_float32_32_128_128":
            m, k, n = 32, 128, 128
            gen_mm_data(m, k, n, dtype_f16, dtype_f32, output)
        elif case_name == "MatmulOnBoardTest.test_mm_float32_32_128_64":
            m, k, n = 32, 128, 64
            gen_mm_data(m, k, n, dtype_f16, dtype_f32, output)
        elif case_name == "MatmulOnBoardTest.test_mm_int8_32_128_64":
            m, k, n = 32, 128, 64
            gen_mm_data(m, k, n, dtype_s8, dtype_s32, output)
        elif case_name == "MatmulOnBoardTest.test_mm_int8_32_128_64_bt":
            mm_size = np.array([32, 128, 64])
            gen_mm_data_trans(mm_size, dtype_s8, dtype_s32, output)
        elif case_name == "MatmulOnBoardTest.test_mm_float_32_128_128_bt":
            mm_size = np.array([32, 128, 128])
            gen_mm_data_trans(mm_size, dtype_f32, dtype_f32, output)
        elif case_name == "MatmulOnBoardTest.test_mm_float_32_128_128":
            m, k, n = 32, 128, 128
            gen_mm_data(m, k, n, dtype_f32, dtype_f32, output)
        elif case_name == "MatmulOnBoardTest.test_mm_float32_32_192_64":
            m, k, n = 32, 192, 64
            gen_mm_data(m, k, n, dtype_f16, dtype_f32, output)
        elif case_name == "MatmulOnBoardTest.test_mm_float32_2_128_128":
            m, k, n = 2, 128, 128
            gen_mm_data(m, k, n, dtype_f16, dtype_f32, output)
        elif case_name == "MatmulOnBoardTest.test_mm_float32_256_256_256":
            m, k, n = 256, 256, 256
            gen_mm_data(m, k, n, dtype_f16, dtype_f32, output)
        elif case_name == "MatmulOnBoardTest.test_mm_float32_32_512_576":
            m, k, n = 32, 512, 576
            gen_mm_data(m, k, n, dtype_f16, dtype_f32, output)
        elif case_name == "MatmulOnBoardTest.test_mm_float32_32_7168_1536":
            m, k, n = 32, 7168, 1536
            gen_mm_data(m, k, n, dtype_f16, dtype_f32, output)
        elif case_name == "MatmulOnBoardTest.test_mm_float32_32_1536_6144":
            m, k, n = 32, 1536, 6144
            gen_mm_data(m, k, n, dtype_f16, dtype_f32, output)
        elif case_name == "MatmulOnBoardTest.test_mm_float32_32_7168_576":
            m, k, n = 32, 7168, 576
            gen_mm_data(m, k, n, dtype_f16, dtype_f32, output)
        elif case_name == "MatmulOnBoardTest.test_mm_float16_64_128_128":
            m, k, n = 64, 128, 128
            gen_mm_data(m, k, n, dtype_f16, dtype_f16, output)
        elif case_name == "MatmulOnBoardTest.test_mm_float16_64_256_128":
            m, k, n = 64, 256, 128
            gen_mm_data(m, k, n, dtype_f16, dtype_f16, output)
        elif case_name == "MatmulOnBoardTest.test_mm_float16_32_7168_1536":
            m, k, n = 32, 7168, 1536
            gen_mm_data(m, k, n, dtype_f16, dtype_f16, output)
        elif case_name == "MatmulOnBoardTest.test_mm_float16_32_1536_6144":
            m, k, n = 32, 1536, 6144
            gen_mm_data(m, k, n, dtype_f16, dtype_f16, output)
        elif case_name == "MatmulOnBoardTest.test_mm_float16_32_7168_576":
            m, k, n = 32, 7168, 576
            gen_mm_data(m, k, n, dtype_f16, dtype_f16, output)
        elif case_name == "MatmulOnBoardTest.test_mm_float16_4_7168_1536":
            m, k, n = 4, 7168, 1536
            gen_mm_data(m, k, n, dtype_f16, dtype_f16, output)
        elif case_name == "MatmulOnBoardTest.test_mm_float16_4_1536_6144":
            m, k, n = 4, 1536, 6144
            gen_mm_data(m, k, n, dtype_f16, dtype_f16, output)
        elif case_name == "MatmulOnBoardTest.test_mm_float16_16_7168_2048":
            m, k, n = 16, 7168, 2048
            gen_mm_data(m, k, n, dtype_f16, dtype_f16, output)
        elif case_name == "MatmulOnBoardTest.test_mm_float16_16_7168_1024":
            m, k, n = 16, 7168, 1024
            gen_mm_data(m, k, n, dtype_f16, dtype_f16, output)
        elif case_name == "MatmulOnBoardTest.test_mm_bfloat16_64_128_128":
            m, k, n = 64, 128, 128
            gen_mm_data(m, k, n, dtype_bf16, dtype_bf16, output)
        elif case_name == "MatmulOnBoardTest.test_mm_bfloat16_f32_64_128_128":
            m, k, n = 64, 128, 128
            gen_mm_data(m, k, n, dtype_bf16, dtype_f32, output)
        elif case_name == "MatmulOnBoardTest.test_mm_unalign_float32_2_128_128":
            m, k, n = 2, 128, 128
            gen_mm_data(m, k, n, dtype_f16, dtype_f32, output)
        elif case_name == "MatmulOnBoardTest.test_mm_unalign_float32_16_35_32":
            m, k, n = 16, 35, 32
            gen_mm_data(m, k, n, dtype_f16, dtype_f32, output)
        elif case_name == "MatmulOnBoardTest.test_mm_unalign_float32_16_32_35":
            m, k, n = 16, 32, 35
            gen_mm_data(m, k, n, dtype_f16, dtype_f32, output)
        elif case_name == "MatmulOnBoardTest.test_mm_float32_64_64_64_bt":
            mm_size = np.array([64, 64, 64])
            gen_mm_data_trans(mm_size, dtype_f16, dtype_f32, output)
        elif case_name == "MatmulOnBoardTest.test_mm_unalign_float32_8_576_256_bt":
            mm_size = np.array([8, 576, 256])
            gen_mm_data_trans(mm_size, dtype_f16, dtype_f32, output)
        elif case_name == "MatmulOnBoardTest.test_mm_unalign_float32_8_64_64_bt":
            mm_size = np.array([8, 64, 64])
            gen_mm_data_trans(mm_size, dtype_f16, dtype_f32, output)
        elif case_name == "MatmulOnBoardTest.test_mm_int8_32_16384_7168":
            m, k, n = 32, 16384, 7168
            gen_mm_data(m, k, n, dtype_s8, dtype_s32, output)
        elif case_name == "CostModelTest.test_mm_float32_64_64_64_bt":
            mm_size = np.array([64, 64, 64])
            gen_mm_data_trans(mm_size, dtype_f16, dtype_f32, output)
        else:
            logging.error("Can't get func to gen golden, Case(%s)", case_name)
            return False
    return True


def main() -> bool:
    """
    单独调试 入口函数
    """
    # 用例名称
    case_name_list: List[str] = [
        "MatmulOnBoardTest.test_mm_float32_64_64_64",
    ]
    # 函数调用
    ret: bool = True
    for cs in case_name_list:
        output: Path = Path(g_src_root, "build/output/bin/golden", cs).resolve()
        output.mkdir(parents=True, exist_ok=True)
        ret = gen_mm_op_data(case_name=cs, output=output)
    return ret


if __name__ == "__main__":
    exit(0 if main() else 1)
