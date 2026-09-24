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
"""row_max_single/row_sum_single 相关用例 Golden 生成逻辑.

本脚本有 2 种执行模式:
1. CI批跑时, 由 cmake/scripts/golden_ctrl.py 调用, 为避免日志过多, 此时 logging 级别为 logging.INFO;
2. 单独调试时, 本脚本单独被调用, 此时 logging 级别为 logging.DEBUG;
"""

import logging
import os
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

np.random.seed(0)


def reduce_max(x, axis=-1):
    x_max = x.max(axis=axis, keepdims=True)
    return x_max


def reduce_sum(x, axis=-1):
    x_sum = x.sum(axis=axis, keepdims=True)
    return x_sum


def golden_generator(dir_path, case_name, x_shape, dtype, reduce_axis=-1, reduce_type="Max"):
    x_shape_dir_path = Path(dir_path, '_'.join([str(i) for i in x_shape]))
    if not os.path.exists(x_shape_dir_path):
        os.makedirs(x_shape_dir_path)

    x_path = Path(x_shape_dir_path, "x.bin")
    out_path = Path(x_shape_dir_path, f"{reduce_type.lower()}_res.bin")

    if x_path.exists() and out_path.exists():
        logging.debug("Case(%s), Golden hit cache.", case_name)
    else:
        x = np.random.uniform(-1, 1, x_shape).astype(dtype)
        x.tofile(x_path)
        # logging.debug(x)
        # logging.debug("x shape: %s", x.shape)

        x = x.astype(np.float32)
        if reduce_type.lower() == "max":
            logging.debug("======================= max golden =======================")
            y = reduce_max(x, reduce_axis)
        elif reduce_type.lower() == "sum":
            logging.debug("======================= sum golden =======================")
            y = reduce_sum(x, reduce_axis)
        else:
            raise KeyError(f"Unknown Reduce Type {reduce_type}")

        y = y.astype(dtype)
        y.tofile(out_path)
        # logging.debug(y)
        # logging.debug("res shape: %s", y.shape)
    return True


@GoldenRegister.reg_golden_func(
    case_names=[
        # RowSumSingle/RowMaxSingle
        "RowMaxSumSingleOnBoardTest.test_operation_row_max_single",
        "RowMaxSumSingleOnBoardTest.test_operation_row_sum_single",
        "RowMaxSumSingleOnBoardTest.test_operation_row_max_single_3dim",
        "RowMaxSumSingleOnBoardTest.test_operation_row_sum_single_3dim_mla_rmsNorm",
        "RowMaxSumSingleOnBoardTest.test_operation_row_max_single_4dim_softmax",
        "RowMaxSumSingleOnBoardTest.test_operation_row_max_single_4dim_softmax_unalign",
        "RowMaxSumSingleOnBoardTest.test_operation_row_sum_single_4dim_softmax",
        "RowMaxSumSingleOnBoardTest.test_operation_row_sum_single_3dim_moe",
        "RowMaxSumSingleOnBoardTest.test_operation_row_sum_single_2dim_moe",
        "RowMaxSumSingleOnBoardTest.test_operation_row_sum_single_3dim_big_moe",
        "RowMaxSumSingleOnBoardTest.test_row_sum_single_4dim_axis0_unalign",
        "RowMaxSumSingleOnBoardTest.test_row_sum_single_4dim_axis1_unalign",
        "RowMaxSumSingleOnBoardTest.test_row_sum_single_4dim_axis2_unalign",
        "RowMaxSumSingleOnBoardTest.test_row_sum_single_unalign",
        "RowMaxSumSingleOnBoardTest.test_row_sum_single_unalign_4_93",
        "RowMaxSumSingleOnBoardTest.test_row_sum_single_unalign_4d",
        "RowMaxSumSingleOnBoardTest.test_row_max_single_unalign",
        "RowMaxSumSingleOnBoardTest.test_row_max_single_unalign_4_93",
        "RowMaxSumSingleOnBoardTest.test_row_max_single_unalign_4d",
    ]
)
def gen_row_max_sum_single_golden(case_name: str, output: Path) -> bool:
    if case_name == "RowMaxSumSingleOnBoardTest.test_operation_row_max_single":
        x_shape = [257, 128]
        dtype = np.float32
        reduce_type = "max"
        golden_generator(output, case_name, x_shape, dtype, -1, reduce_type)
    elif case_name == "RowMaxSumSingleOnBoardTest.test_operation_row_sum_single":
        x_shape = [257, 128]
        dtype = np.float32
        reduce_type = "sum"
        golden_generator(output, case_name, x_shape, dtype, -1, reduce_type)
    elif case_name == "RowMaxSumSingleOnBoardTest.test_operation_row_max_single_3dim":
        x_shape = [8, 4, 128]
        dtype = np.float32
        reduce_type = "max"
        golden_generator(output, case_name, x_shape, dtype, -1, reduce_type)
    elif case_name == "RowMaxSumSingleOnBoardTest.test_operation_row_sum_single_3dim_mla_rmsNorm":
        x_shape = [16, 1, 1536]
        dtype = np.float32
        reduce_type = "sum"
        golden_generator(output, case_name, x_shape, dtype, -1, reduce_type)
    elif case_name == "RowMaxSumSingleOnBoardTest.test_operation_row_max_single_4dim_softmax":
        x_shape = [2, 128, 1, 256]
        dtype = np.float32
        reduce_type = "max"
        golden_generator(output, case_name, x_shape, dtype, -1, reduce_type)
    elif case_name == "RowMaxSumSingleOnBoardTest.test_operation_row_max_single_4dim_softmax_unalign":
        x_shape = [1, 128, 1, 248]
        dtype = np.float32
        reduce_type = "max"
        golden_generator(output, case_name, x_shape, dtype, -1, reduce_type)
    elif case_name == "RowMaxSumSingleOnBoardTest.test_operation_row_sum_single_4dim_softmax":
        x_shape = [32, 128, 1, 256]
        dtype = np.float32
        reduce_type = "sum"
        golden_generator(output, case_name, x_shape, dtype, -1, reduce_type)
    elif case_name == "RowMaxSumSingleOnBoardTest.test_operation_row_sum_single_3dim_moe":
        x_shape = [6, 1, 8, 1024]
        dtype = np.float32
        reduce_type = "sum"
        golden_generator(output, case_name, x_shape, dtype, -2, reduce_type)
    elif case_name == "RowMaxSumSingleOnBoardTest.test_operation_row_sum_single_2dim_moe":
        x_shape = [8, 1, 1, 256]
        dtype = np.float32
        reduce_type = "sum"
        golden_generator(output, case_name, x_shape, dtype, 0, reduce_type)
    elif case_name == "RowMaxSumSingleOnBoardTest.test_operation_row_sum_single_3dim_big_moe":
        x_shape = [8, 1, 8, 7168]
        dtype = np.float32
        reduce_type = "sum"
        golden_generator(output, case_name, x_shape, dtype, -2, reduce_type)
    elif case_name == "RowMaxSumSingleOnBoardTest.test_row_sum_single_4dim_axis0_unalign":
        x_shape = [6, 2, 8, 255]
        dtype = np.float32
        reduce_type = "sum"
        golden_generator(output, case_name, x_shape, dtype, 0, reduce_type)
    elif case_name == "RowMaxSumSingleOnBoardTest.test_row_sum_single_4dim_axis1_unalign":
        x_shape = [4, 2, 8, 255]
        dtype = np.float32
        reduce_type = "sum"
        golden_generator(output, case_name, x_shape, dtype, 1, reduce_type)
    elif case_name == "RowMaxSumSingleOnBoardTest.test_row_sum_single_4dim_axis2_unalign":
        x_shape = [3, 2, 8, 255]
        dtype = np.float32
        reduce_type = "sum"
        golden_generator(output, case_name, x_shape, dtype, 2, reduce_type)
    elif case_name == "RowMaxSumSingleOnBoardTest.test_row_sum_single_unalign":
        x_shape = [4, 530]
        dtype = np.float32
        reduce_type = "sum"
        golden_generator(output, case_name, x_shape, dtype, -1, reduce_type)
    elif case_name == "RowMaxSumSingleOnBoardTest.test_row_sum_single_unalign_4_93":
        x_shape = [4, 93]
        dtype = np.float32
        reduce_type = "sum"
        golden_generator(output, case_name, x_shape, dtype, -1, reduce_type)
    elif case_name == "RowMaxSumSingleOnBoardTest.test_row_sum_single_unalign_4d":
        x_shape = [3, 3, 4, 530]
        dtype = np.float32
        reduce_type = "sum"
        golden_generator(output, case_name, x_shape, dtype, -1, reduce_type)
    elif case_name == "RowMaxSumSingleOnBoardTest.test_row_max_single_unalign":
        x_shape = [4, 93]
        dtype = np.float32
        reduce_type = "max"
        golden_generator(output, case_name, x_shape, dtype, -1, reduce_type)
    elif case_name == "RowMaxSumSingleOnBoardTest.test_row_max_single_unalign_4_93":
        x_shape = [4, 93]
        dtype = np.float32
        reduce_type = "max"
        golden_generator(output, case_name, x_shape, dtype, -1, reduce_type)
    elif case_name == "RowMaxSumSingleOnBoardTest.test_row_max_single_unalign_4d":
        x_shape = [3, 3, 4, 530]
        dtype = np.float32
        reduce_type = "max"
        golden_generator(output, case_name, x_shape, dtype, -1, reduce_type)
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
        "RowMaxSumSingleOnBoardTest.test_operation_row_max_single",
        "RowMaxSumSingleOnBoardTest.test_operation_row_sum_single",
        "RowMaxSumSingleOnBoardTest.test_operation_row_max_single_3dim",
        "RowMaxSumSingleOnBoardTest.test_operation_row_sum_single_3dim_mla_rmsNorm",
        "RowMaxSumSingleOnBoardTest.test_operation_row_max_single_4dim_softmax",
        "RowMaxSumSingleOnBoardTest.test_operation_row_max_single_4dim_softmax_unalign",
        "RowMaxSumSingleOnBoardTest.test_operation_row_sum_single_4dim_softmax",
        "RowMaxSumSingleOnBoardTest.test_operation_row_sum_single_3dim_moe",
        "RowMaxSumSingleOnBoardTest.test_row_sum_single_unalign",
        "RowMaxSumSingleOnBoardTest.test_row_sum_single_unalign_4d",
    ]
    # 函数调用
    ret: bool = True
    for cs in case_name_list:
        output: Path = Path(g_src_root, "build/output/bin/golden", cs).resolve()
        output.mkdir(parents=True, exist_ok=True)
        ret = gen_row_max_sum_single_golden(case_name=cs, output=output)
    return ret


if __name__ == "__main__":
    exit(0 if main() else 1)
