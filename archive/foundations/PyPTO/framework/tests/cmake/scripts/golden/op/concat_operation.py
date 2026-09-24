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
"""Concat 相关用例 Golden 生成逻辑.

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


@GoldenRegister.reg_golden_func(
    case_names=[
        "ConcatOnBoardTest.test_concat_exp_dim4_float32",
        "ConcatOnBoardTest.test_exp_concat_dim4_float32",
        "ConcatOnBoardTest.test_concat_dim4_float32",
        "ConcatOnBoardTest.test_concat_dim2_float32_moe",
        "ConcatOnBoardTest.test_concat_sqrt_dim4_float32",
        "ConcatOnBoardTest.test_concat_100_inputs_float32",
        "ConcatOnBoardTest.test_concat_128_inputs_float32",
    ]
)
def concat_operator_func1(case_name: str, output: Path) -> bool:
    in0 = 2
    in1 = 2
    row = 64
    col = 64

    shape_i = [in0, in1, row, col]
    _shape_o = [in0, in1, row, col * 2]

    dtype = np.float32

    if case_name == "ConcatOnBoardTest.test_concat_exp_dim4_float32":
        x_path = Path(output, 'concat_exp_2_2_32_32_operand1.bin')
        y_path = Path(output, 'concat_exp_2_2_32_32_operand2.bin')
        o_path = Path(output, 'concat_exp_2_2_32_32_res.bin')
        complete = x_path.exists() and y_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
            return True
        else:
            operand_1 = np.random.uniform(-1, 1, [2, 2, 32, 32]).astype(dtype)
            operand_2 = np.random.uniform(-1, 1, [2, 2, 32, 32]).astype(dtype)
            operand_1.tofile(x_path)
            operand_2.tofile(y_path)
            res = np.concatenate([operand_1, operand_2], -1)
            res = np.exp(res)
            res.tofile(o_path)
            return True
    elif case_name == "ConcatOnBoardTest.test_exp_concat_dim4_float32":
        x_path = Path(output, 'concat_exp_2_2_32_32_operand1.bin')
        y_path = Path(output, 'concat_exp_2_2_32_32_operand2.bin')
        o_path = Path(output, 'concat_exp_2_2_32_32_res.bin')
        complete = x_path.exists() and y_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
            return True
        else:
            operand_1 = np.random.uniform(-1, 1, [2, 2, 32, 64]).astype(dtype)
            operand_2 = np.random.uniform(-1, 1, [2, 2, 32, 64]).astype(dtype)
            operand_1.tofile(x_path)
            operand_2.tofile(y_path)
            operand_1 = np.exp(operand_1)
            operand_2 = np.exp(operand_2)
            res = np.concatenate([operand_1, operand_2], -1)
            res.tofile(o_path)
            return True
    elif case_name == "ConcatOnBoardTest.test_concat_dim4_float32":
        x_path = Path(output, 'concat_2_2_64_64_operand1.bin')
        y_path = Path(output, 'concat_2_2_64_64_operand2.bin')
        o_path = Path(output, 'concat_2_2_64_64_res.bin')
        complete = x_path.exists() and y_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
            return True
        else:
            operand_1 = np.random.uniform(-1, 1, shape_i).astype(dtype)
            operand_2 = np.random.uniform(-1, 1, shape_i).astype(dtype)
            operand_1.tofile(x_path)
            operand_2.tofile(y_path)
            res = np.concatenate([operand_1, operand_2], -1)
            res.tofile(o_path)
            return True
    elif case_name == "ConcatOnBoardTest.test_concat_dim2_float32_moe":
        x_path = Path(output, 'concat_3_7168_operand1.bin')
        y_path = Path(output, 'concat_64_7168_operand2.bin')
        o_path = Path(output, 'concat_67_7168_res.bin')
        complete = x_path.exists() and y_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
            return True
        else:
            operand_1 = np.random.uniform(-1, 1, [3, 7168]).astype(dtype)
            operand_2 = np.random.uniform(-1, 1, [64, 7168]).astype(dtype)
            operand_1.tofile(x_path)
            operand_2.tofile(y_path)
            res = np.concatenate([operand_1, operand_2], -2)
            res.tofile(o_path)
            return True
    elif case_name == "ConcatOnBoardTest.test_concat_sqrt_dim4_float32":
        x_path = Path(output, 'concat_sqrt_fp32_operand1.bin')
        y_path = Path(output, 'concat_sqrt_fp32_operand2.bin')
        o_path = Path(output, 'concat_sqrt_fp32_res.bin')
        complete = x_path.exists() and y_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
            return True
        else:
            operand_1 = np.random.uniform(0, 100, [2, 2, 32, 64]).astype(dtype)
            operand_2 = np.random.uniform(0, 100, [2, 2, 64, 64]).astype(dtype)
            operand_1.tofile(x_path)
            operand_2.tofile(y_path)
            res = np.concatenate([operand_1, operand_2], 2)
            res = np.sqrt(res)
            res.tofile(o_path)
            return True
    elif case_name == "ConcatOnBoardTest.test_concat_100_inputs_float32":
        inputs = [np.random.uniform(-1, 1, [2, 2, 4, 16]).astype(dtype) for _ in range(100)]
        for index in range(len(inputs)):
            inputs[index].tofile(Path(output, f'concat_100_inputs_{index}_fp32.bin'))
        res = np.concatenate(inputs, -1)
        res.tofile(Path(output, 'concat_100_inputs_res_fp32.bin'))
    elif case_name == "ConcatOnBoardTest.test_concat_128_inputs_float32":
        inputs = [np.random.uniform(-1, 1, [2, 1, 8, 8]).astype(dtype) for _ in range(128)]
        for index in range(len(inputs)):
            inputs[index].tofile(Path(output, f'concat_128_inputs_{index}_fp32.bin'))
        res = np.concatenate(inputs, 2)
        res.tofile(Path(output, 'concat_128_inputs_res_fp32.bin'))
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
        "ConcatOnBoardTest.test_concat_exp_dim4_float32",
    ]
    # 函数调用
    ret: bool = True
    for cs in case_name_list:
        output: Path = Path(g_src_root, "build/output/bin/golden", cs).resolve()
        output.mkdir(parents=True, exist_ok=True)
        ret = concat_operator_func1(case_name=cs, output=output)
    return ret


if __name__ == "__main__":
    exit(0 if main() else 1)
