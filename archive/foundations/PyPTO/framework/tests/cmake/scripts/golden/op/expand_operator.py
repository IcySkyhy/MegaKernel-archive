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


def operation_expand_32_1_to_32_32(output_dir: Path):
    x_path = Path(output_dir, 'expand_x.bin')
    o_path = Path(output_dir, 'expand_res.bin')
    dtype = np.float32
    x = np.random.uniform(-1, 1, [32, 1]).astype(dtype)
    x.tofile(x_path)
    x = np.broadcast_to(x, [32, 32])
    x = np.copy(x)
    x.tofile(o_path)


def operation_expand_32_8_1_to_32_8_32(output_dir: Path):
    x_path = Path(output_dir, 'expand_x.bin')
    o_path = Path(output_dir, 'expand_res.bin')
    dtype = np.float32
    x = np.random.uniform(-1, 1, [32, 8, 1]).astype(dtype)
    x.tofile(x_path)
    x = np.broadcast_to(x, [32, 8, 32])
    x = np.copy(x)
    x.tofile(o_path)


def operation_expand_32_1_to_32_23(output_dir: Path):
    x_path = Path(output_dir, 'expand_x.bin')
    o_path = Path(output_dir, 'expand_res.bin')
    dtype = np.float32
    x = np.random.uniform(-1, 1, [32, 1]).astype(dtype)
    x.tofile(x_path)
    x = np.broadcast_to(x, [32, 23])
    x = np.copy(x)
    x.tofile(o_path)


def operation_expand_1_1_to_1_16384(output_dir: Path):
    x_path = Path(output_dir, 'expand_x.bin')
    o_path = Path(output_dir, 'expand_res.bin')
    dtype = np.float32
    x = np.random.uniform(-1, 1, [1, 1]).astype(dtype)
    x.tofile(x_path)
    x = np.broadcast_to(x, [1, 16384])
    x = np.copy(x)
    x.tofile(o_path)


def operation_expand_32_8_1_to_32_8_23(output_dir: Path):
    x_path = Path(output_dir, 'expand_x.bin')
    o_path = Path(output_dir, 'expand_res.bin')
    dtype = np.float32
    x = np.random.uniform(-1, 1, [32, 8, 1]).astype(dtype)
    x.tofile(x_path)
    x = np.broadcast_to(x, [32, 8, 23])
    x = np.copy(x)
    x.tofile(o_path)


def operation_expand_for_4_dim(output_dir: Path):
    x_path = Path(output_dir, 'expand_x.bin')
    o_path = Path(output_dir, 'expand_res.bin')
    dtype = np.float32
    x = np.random.uniform(-1, 1, [1, 32, 400, 23]).astype(dtype)
    x.tofile(x_path)
    x = np.broadcast_to(x, [8, 32, 400, 23])
    x = np.copy(x)
    x.tofile(o_path)


@GoldenRegister.reg_golden_func(
    case_names=[
        # Expand
        "ExpandOnBoardTest.test_expand_32_1_to_32_32",
        "ExpandOnBoardTest.test_expand_32_8_1_to_32_8_32",
        "ExpandOnBoardTest.test_expand_32_1_to_32_23",
        "ExpandOnBoardTest.test_expand_32_8_1_to_32_8_23",
        "ExpandOnBoardTest.test_expand_for_4_dim",
        "ExpandOnBoardTest.test_expand_1_1_to_1_16384",
    ]
)
def expand_operator_func1(case_name: str, output: Path) -> bool:
    _dtype = np.float32

    x_path = Path(output, 'x.bin')
    res_path = Path(output, 'res.bin')
    complete = x_path.exists() and res_path.exists()

    if complete:
        logging.debug("Case(%s), Golden complete.", case_name)
    else:
        if case_name == "ExpandOnBoardTest.test_expand_32_1_to_32_32":
            operation_expand_32_1_to_32_32(output)
        elif case_name == "ExpandOnBoardTest.test_expand_32_8_1_to_32_8_32":
            operation_expand_32_8_1_to_32_8_32(output)
        elif case_name == "ExpandOnBoardTest.test_expand_32_8_1_to_32_8_23":
            operation_expand_32_8_1_to_32_8_23(output)
        elif case_name == "ExpandOnBoardTest.test_expand_32_1_to_32_23":
            operation_expand_32_1_to_32_23(output)
        elif case_name == "ExpandOnBoardTest.test_expand_for_4_dim":
            operation_expand_for_4_dim(output)
        elif case_name == "ExpandOnBoardTest.test_expand_1_1_to_1_16384":
            operation_expand_1_1_to_1_16384(output)
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
        "ExpandOnBoardTest.test_expand_32_1_to_32_32",
    ]
    # 函数调用
    ret: bool = True
    for cs in case_name_list:
        output: Path = Path(g_src_root, "build/output/bin/golden", cs).resolve()
        output.mkdir(parents=True, exist_ok=True)
        ret = expand_operator_func1(case_name=cs, output=output)
    return ret


if __name__ == "__main__":
    exit(0 if main() else 1)
