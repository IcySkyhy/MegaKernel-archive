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
"""Gather Operator 相关用例 Golden 生成逻辑.

本脚本有 2 种执行模式:
1. CI批跑时, 由 cmake/scripts/golden_ctrl.py 调用, 为避免日志过多, 此时 logging 级别为 logging.INFO;
2. 单独调试时, 本脚本单独被调用, 此时 logging 级别为 logging.DEBUG;
"""

import logging
from pathlib import Path
import sys
from typing import List

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


def gen_gather_data(axis, b, s, s2, d, dtype, indices_dtype, output_dir: Path):
    shape_params = [s2, d]
    shape_indices = [b, s]
    shape_res = [b, s, d]
    logging.debug("shape params is %s", shape_params)
    logging.debug("shape indices is %s", shape_indices)
    logging.debug("shape res is %s", shape_res)

    x_path = Path(output_dir, 'x.bin')
    indices_path = Path(output_dir, 'indices.bin')
    y_path = Path(output_dir, 'y_golden.bin')

    x = np.random.uniform(-10, 10, shape_params).astype(dtype)
    x.tofile(x_path)
    indices = np.random.randint(0, shape_params[axis], size=shape_indices).astype(indices_dtype)
    indices.tofile(indices_path)

    # numpy
    y = np.empty(shape_res).astype(dtype)
    for _b in range(b):
        for _s in range(s):
            index = indices[_b][_s]
            y[_b][_s][:] = x[index][:]
    y.tofile(y_path)

    """
    # tf
    import tensorflow as tf

    x_tensor = tf.convert_to_tensor(x)
    indices_tensor = tf.convert_to_tensor(indices)
    result_tf = tf.gather(x_tensor, indices_tensor, axis=axis)
    y_tf = result_tf.numpy()

    result = np.allclose(y, y_tf, rtol=1e-3, atol=1e-3)
    logging.debug("====== golden precise: %s", result")
    """


# src0: [s2, d], src1: [s], axis: 0, output: [s, d]
def gen_gather_data_2d(axis, s, s2, d, dtype, indices_dtype, output_dir: Path):
    shape_params = [s2, d]
    shape_indices = [s]
    shape_res = [s, d]
    logging.debug("shape params is %s", shape_params)
    logging.debug("shape indices is %s", shape_indices)
    logging.debug("shape res is %s", shape_res)

    x_path = Path(output_dir, 'x.bin')
    indices_path = Path(output_dir, 'indices.bin')
    y_path = Path(output_dir, 'y_golden.bin')

    x = np.random.uniform(-10, 10, shape_params).astype(dtype)
    x.tofile(x_path)
    indices = np.random.randint(0, shape_params[axis], size=shape_indices).astype(indices_dtype)
    indices.tofile(indices_path)

    # numpy
    y = np.empty(shape_res).astype(dtype)
    for _s in range(s):
        index = indices[_s]
        y[_s][:] = x[index][:]
    y.tofile(y_path)


@GoldenRegister.reg_golden_func(
    case_names=[
        # Gather
        "GatherOnBoardTest.test_gather_float_32_64_1_32",
        "GatherOnBoardTest.test_gather_float_32_65_1_33",
        "GatherOnBoardTest.test_gather_float_64_256_1_64",
        "GatherOnBoardTest.test_gather_float_1_64_32_1",
        "GatherOnBoardTest.test_gather_float_64_512_16_64",
        "GatherOnBoardTest.test_gather_float_8_7168_64",
        "GatherOnBoardTest.test_gather_float_8_7169_64",
    ]
)
def gen_gather_op_date(case_name: str, output: Path) -> bool:
    dtype = np.float32
    indices_dtype = np.int32

    x_path = Path(output, 'x.bin')
    indices_path = Path(output, 'indices.bin')
    y_path = Path(output, 'y_golden.bin')
    complete = x_path.exists() and indices_path.exists() and y_path.exists()

    if complete:
        logging.debug("Case(%s), Golden complete.", case_name)
    else:
        if case_name == "GatherOnBoardTest.test_gather_float_32_64_1_32":
            b, s, s2, d = 1, 32, 32, 64
            axis = 0
            gen_gather_data(axis, b, s, s2, d, dtype, indices_dtype, output)
        elif case_name == "GatherOnBoardTest.test_gather_float_64_256_1_64":
            b, s, s2, d = 1, 64, 64, 256
            axis = 0
            gen_gather_data(axis, b, s, s2, d, dtype, indices_dtype, output)
        elif case_name == "GatherOnBoardTest.test_gather_float_32_65_1_33":
            b, s, s2, d = 1, 33, 32, 65
            axis = 0
            gen_gather_data(axis, b, s, s2, d, dtype, indices_dtype, output)
        elif case_name == "GatherOnBoardTest.test_gather_float_1_64_32_1":
            b, s, s2, d = 32, 1, 1, 64
            axis = 0
            gen_gather_data(axis, b, s, s2, d, dtype, indices_dtype, output)
        elif case_name == "GatherOnBoardTest.test_gather_float_64_512_16_64":
            b, s, s2, d = 16, 64, 64, 512
            axis = 0
            gen_gather_data(axis, b, s, s2, d, dtype, indices_dtype, output)
        elif case_name == "GatherOnBoardTest.test_gather_float_8_7168_64":
            s, s2, d = 64, 8, 7168
            axis = 0
            gen_gather_data_2d(axis, s, s2, d, dtype, indices_dtype, output)
        elif case_name == "GatherOnBoardTest.test_gather_float_8_7169_64":
            s, s2, d = 64, 8, 7169
            axis = 0
            gen_gather_data_2d(axis, s, s2, d, dtype, indices_dtype, output)
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
        "GatherOnBoardTest.test_gather_float_32_64_1_32",
    ]
    # 函数调用
    ret: bool = True
    for cs in case_name_list:
        output: Path = Path(g_src_root, "build/output/bin/golden", cs).resolve()
        output.mkdir(parents=True, exist_ok=True)
        ret = gen_gather_op_date(case_name=cs, output=output)
    return ret


if __name__ == "__main__":
    exit(0 if main() else 1)
