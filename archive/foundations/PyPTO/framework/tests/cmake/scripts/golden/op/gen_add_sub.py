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
"""

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
        "OnBoardTestAstApi.test_add_sub_all2all_torchapi_multi_function",
    ]
)
def add_sub_main(case_name: str, output: Path) -> bool:
    row = 64
    col = 64
    shape_i = [row, col]
    _shape_o = [shape_i[0], shape_i[1]]

    dtype = np.float32

    x = np.random.uniform(-1, 1, shape_i).astype(dtype)
    x.tofile(Path(output, 'x.bin'))
    y = np.random.uniform(-1, 1, shape_i).astype(dtype)
    y.tofile(Path(output, 'y.bin'))

    a = x + y
    a.tofile(Path(output, 'res.bin'))

    sub = x - y
    sub.tofile(Path(output, 'res_sub.bin'))
    logging.debug("gen add sub golden success!!!")
    logging.debug("row is %s, col is %s", row, col)
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
        ret = add_sub_main(case_name=cs, output=output)
    return ret


if __name__ == "__main__":
    exit(0 if main() else 1)
