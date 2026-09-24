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
"""reshape 相关用例 Golden 生成逻辑."""

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


def gen_uniform_data(data_shape, min_value, max_value, dtype):
    if min_value == 0 and max_value == 0:
        return np.zeros(data_shape, dtype=dtype)
    if dtype == np.bool_:
        return np.random.choice([True, False], size=data_shape)
    return np.random.uniform(low=min_value, high=max_value, size=data_shape).astype(dtype)


def gen_only_reshape2(output: Path):
    b = 2
    sq = 32
    d = 16
    b_sq = b * sq

    min_value = -1.0
    max_value = 1.0
    dtype = np.float32

    q_shape = (b, sq, d)
    q = gen_uniform_data(q_shape, min_value, max_value, dtype)

    q_reshape = q.reshape((b_sq, d))

    out = np.exp(q_reshape)

    q_path = Path(output, 'q.bin')
    out_path = Path(output, 'out.bin')

    q.tofile(q_path)
    out.tofile(out_path)

    logging.info(f"Generated input data saved to: {q_path}")
    logging.info(f"Generated output data saved to: {out_path}")


@GoldenRegister.reg_golden_func(
    case_names=[
        "DynamicReshapeTest.test_only_reshape2",
    ]
)
def reshape_operator_func1(case_name: str, output: Path) -> bool:
    if case_name == "DynamicReshapeTest.test_only_reshape2":
        gen_only_reshape2(output)
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
        "DynamicReshapeTest.test_only_reshape2",
    ]
    # 函数调用
    ret: bool = True
    for cs in case_name_list:
        output: Path = Path(g_src_root, "build/output/bin/golden", cs).resolve()
        output.mkdir(parents=True, exist_ok=True)
        ret = reshape_operator_func1(case_name=cs, output=output)
    return ret


if __name__ == "__main__":
    exit(0 if main() else 1)
