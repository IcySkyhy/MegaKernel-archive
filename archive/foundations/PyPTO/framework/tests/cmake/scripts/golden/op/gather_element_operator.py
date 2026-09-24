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
import os
from pathlib import Path
import sys
from typing import List

import torch

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


def gen_gather_element_data(s0, s1, d0, d1, axis, dtype, indices_dtype, output_dir: Path):
    shape_params = [s0, s1]
    _shape_indices = [d0, d1]
    axis = axis
    _shape_res = [d0, d1]

    params_path = Path(output_dir, 'params.bin')
    indices_path = Path(output_dir, 'indices.bin')
    res_path = Path(output_dir, 'res_golden.bin')

    target_dir = f'./{s0}_{s1}_{d0}_{d1}_{axis}'
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        logging.debug("dir %s created", target_dir)

    params = torch.randn(s0, s1, dtype=dtype)
    params_np = params.numpy()
    params_np.tofile(params_path)
    indices = torch.randint(0, shape_params[axis], (d0, d1), dtype=indices_dtype)
    indices_np = indices.numpy()
    indices_np.tofile(indices_path)

    indices_int64 = indices.int().to(torch.int64)

    res = params.gather(axis, indices_int64)
    res_np = res.numpy()
    res_np.tofile(res_path)


@GoldenRegister.reg_golden_func(
    case_names=[
        # GatherElement
        "GatherElementOnBoardTest.test_gather_element_float_16_64_8_32_1",
        "GatherElementOnBoardTest.test_gather_element_float_16_70_8_40_1",
        "GatherElementOnBoardTest.test_gather_element_float_16_64_7_32_1",
        "GatherElementOnBoardTest.test_gather_element_float_16_64_7_32_0",
    ]
)
def gen_gather_element_op_date(case_name: str, output: Path) -> bool:
    dtype = torch.float32
    indices_dtype = torch.int32

    params_path = Path(output, 'params.bin')
    indices_path = Path(output, 'indices.bin')
    res_path = Path(output, 'res_golden.bin')
    complete = params_path.exists() and indices_path.exists() and res_path.exists()

    if complete:
        logging.debug("Case(%s), Golden complete.", case_name)
        return True
    else:
        if case_name == "GatherElementOnBoardTest.test_gather_element_float_16_64_8_32_1":
            s0, s1, d0, d1 = 16, 64, 8, 32
            axis = 1
            gen_gather_element_data(s0, s1, d0, d1, axis, dtype, indices_dtype, output)
        elif case_name == "GatherElementOnBoardTest.test_gather_element_float_16_70_8_40_1":
            s0, s1, d0, d1 = 16, 70, 8, 40
            axis = 1
            gen_gather_element_data(s0, s1, d0, d1, axis, dtype, indices_dtype, output)
        elif case_name == "GatherElementOnBoardTest.test_gather_element_float_16_64_7_32_1":
            s0, s1, d0, d1 = 16, 64, 7, 32
            axis = 1
            gen_gather_element_data(s0, s1, d0, d1, axis, dtype, indices_dtype, output)
        elif case_name == "GatherElementOnBoardTest.test_gather_element_float_16_64_7_32_0":
            s0, s1, d0, d1 = 16, 64, 7, 32
            axis = 0
            gen_gather_element_data(s0, s1, d0, d1, axis, dtype, indices_dtype, output)
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
        "GatherElementOnBoardTest.test_gather_element_float_16_64_8_32_1",
    ]
    # 函数调用
    ret: bool = True
    for cs in case_name_list:
        output: Path = Path(g_src_root, "build/output/bin/golden", cs).resolve()
        output.mkdir(parents=True, exist_ok=True)
        ret = gen_gather_element_op_date(case_name=cs, output=output)
    return ret


if __name__ == "__main__":
    exit(0 if main() else 1)
