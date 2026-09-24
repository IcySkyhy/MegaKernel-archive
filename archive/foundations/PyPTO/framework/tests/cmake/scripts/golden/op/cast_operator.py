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
"""CastOperator 相关用例 Golden 生成逻辑.

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


def gen_rint(x, dst_dtype):
    res = np.rint(x).astype(dst_dtype)
    return res


def gen_none(x, dst_dtype):
    res = x.astype(dst_dtype)
    return res


def gen_trunc(x, dst_dtype):
    res = np.trunc(x).astype(dst_dtype)
    return res


@GoldenRegister.reg_golden_func(
    case_names=[
        # Cast/Floor/Round
        "CastOnBoard.test_cast_fp32toint32rint_1_4608",
        "CastOnBoard.test_cast_int32tofp16none_1_4608",
        "CastOnBoard.test_cast_fp16toint8trunc_1_4608",
        "CastOnBoard.test_cast_fp16tofp32_unalign",
        "CastOnBoard.test_cast_fp32toint32rint_1_16384",
    ]
)
def cast_operator_func(case_name: str, output: Path) -> bool:
    if case_name == "CastOnBoard.test_cast_fp32toint32rint_1_4608":
        shape = [1, 4608]
        src_dtype = np.float32
        dst_dtype = np.int32
        x_path = Path(output, 'fp32toint32rint_x.bin')
        y_path = Path(output, 'fp32toint32rint_golden.bin')
        complete = x_path.exists() and y_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-128, 127, shape).astype(src_dtype)
            x.tofile(x_path)
            y = gen_rint(x, dst_dtype)
            y.tofile(y_path)
    elif case_name == "CastOnBoard.test_cast_int32tofp16none_1_4608":
        shape = [1, 4608]
        src_dtype = np.int32
        dst_dtype = np.float16
        x_path = Path(output, 'int32tofp16none_x.bin')
        y_path = Path(output, 'int32tofp16none_golden.bin')
        complete = x_path.exists() and y_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.randint(-128, 128, shape).astype(src_dtype)
            x.tofile(x_path)
            y = gen_none(x, dst_dtype)
            y.tofile(y_path)
    elif case_name == "CastOnBoard.test_cast_fp16toint8trunc_1_4608":
        shape = [1, 4608]
        src_dtype = np.float16
        dst_dtype = np.int8
        x_path = Path(output, 'fp16toint8trunc_x.bin')
        y_path = Path(output, 'fp16toint8trunc_golden.bin')
        complete = x_path.exists() and y_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-128, 127, shape).astype(src_dtype)
            x.tofile(x_path)
            y = gen_trunc(x, dst_dtype)
            y.tofile(y_path)
    elif case_name == "CastOnBoard.test_cast_fp16tofp32_unalign":
        shape = [4, 130]
        src_dtype = np.float16
        dst_dtype = np.float32
        x1_path = Path(output, 'fp16tofp32_unalign_x1.bin')
        y_path = Path(output, 'fp16tofp32_unalign_golden.bin')
        complete = x1_path.exists() and y_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x1 = np.random.uniform(-10, 10, shape).astype(src_dtype)
            x1.tofile(x1_path)
            y = gen_none(x1, dst_dtype)
            y.tofile(y_path)
    elif case_name == "CastOnBoard.test_cast_fp32toint32rint_1_16384":
        shape = [1, 16384]
        src_dtype = np.float32
        dst_dtype = np.int32
        x_path = Path(output, 'fp32toint32rint_x.bin')
        y_path = Path(output, 'fp32toint32rint_golden.bin')
        complete = x_path.exists() and y_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-128, 127, shape).astype(src_dtype)
            x.tofile(x_path)
            y = gen_rint(x, dst_dtype)
            y.tofile(y_path)
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
        "CastOnBoard.test_cast_int32tofp16none_1_4608",
    ]
    # 函数调用
    ret: bool = True
    for cs in case_name_list:
        output: Path = Path(g_src_root, "build/output/bin/golden", cs).resolve()
        output.mkdir(parents=True, exist_ok=True)
        ret = cast_operator_func(case_name=cs, output=output)
    return ret


if __name__ == "__main__":
    exit(0 if main() else 1)
