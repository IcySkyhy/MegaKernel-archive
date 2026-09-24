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
"""topk 相关用例 Golden 生成逻辑.

本脚本有 2 种执行模式:
1. CI批跑时, 由 cmake/scripts/golden_ctrl.py 调用, 为避免日志过多, 此时 logging 级别为 logging.INFO;
2. 单独调试时, 本脚本单独被调用, 此时 logging 级别为 logging.DEBUG;
"""

import logging
from pathlib import Path
import sys
from typing import List

import numpy as np
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


@GoldenRegister.reg_golden_func(
    case_names=[
        # topk
        "TopkOnBoardTest.test_operation_tensor_128_32_32_topk",
        "TopkOnBoardTest.test_operation_tensor_128_32_16_topk",
        "TopkOnBoardTest.test_operation_tensor_4_32_8_topk",
        "TopkOnBoardTest.test_operation_tensor_2_16_8_topk",
        "TopkOnBoardTest.test_operation_tensor_2_8_4_topk",
        "TopkOnBoardTest.test_operation_tensor_1_8_4_topk",
        "TopkOnBoardTest.test_operation_tensor_2_288_15_topk",
        "TopkOnBoardTest.test_operation_tensor_2_288_15_topk_reverse",
    ]
)
def gen_topk_golden(case_name: str, output: Path) -> bool:
    seed = 50
    torch.manual_seed(seed)
    if case_name == "TopkOnBoardTest.test_operation_tensor_128_32_32_topk":
        x_path = Path(output, 'x.bin')
        val_path = Path(output, 'val.bin')
        idx_path = Path(output, 'idx.bin')
        shape0 = 128
        shape1 = 32
        k = 32
        complete = x_path.exists() and val_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            # x = torch.randn((128, 32))
            x = torch.arange(shape0 * shape1, dtype=torch.float32).reshape(shape0, shape1)
            val, idx = x.topk(k, dim=1, largest=True, sorted=True)
            x.numpy().tofile(x_path)
            val.numpy().tofile(val_path)
            idx.numpy().astype(np.int32).tofile(idx_path)
    elif case_name == "TopkOnBoardTest.test_operation_tensor_128_32_16_topk":
        x_path = Path(output, 'x.bin')
        val_path = Path(output, 'val.bin')
        idx_path = Path(output, 'idx.bin')
        shape0 = 128
        shape1 = 32
        k = 16
        complete = x_path.exists() and val_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            # x = torch.randn((shape0, shape1))
            x = torch.arange(shape0 * shape1, dtype=torch.float32).reshape(shape0, shape1)
            val, idx = x.topk(k, dim=1, largest=True, sorted=True)
            x.numpy().tofile(x_path)
            val.numpy().tofile(val_path)
            idx.numpy().astype(np.int32).tofile(idx_path)
    elif case_name == "TopkOnBoardTest.test_operation_tensor_4_32_8_topk":
        x_path = Path(output, 'x.bin')
        val_path = Path(output, 'val.bin')
        idx_path = Path(output, 'idx.bin')
        shape0 = 4
        shape1 = 32
        k = 8
        complete = x_path.exists() and val_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            # x = torch.randn((shape0, shape1))
            x = torch.arange(shape0 * shape1, dtype=torch.float32).reshape(shape0, shape1)
            val, idx = x.topk(k, dim=1, largest=True, sorted=True)
            x.numpy().tofile(x_path)
            val.numpy().tofile(val_path)
            idx.numpy().astype(np.int32).tofile(idx_path)
    elif case_name == "TopkOnBoardTest.test_operation_tensor_2_16_8_topk":
        x_path = Path(output, 'x.bin')
        val_path = Path(output, 'val.bin')
        idx_path = Path(output, 'idx.bin')
        shape0 = 2
        shape1 = 16
        k = 8
        complete = x_path.exists() and val_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            # x = torch.randn((shape0, shape1))
            x = torch.arange(shape0 * shape1, dtype=torch.float32).reshape(shape0, shape1)
            val, idx = x.topk(k, dim=1, largest=True, sorted=True)
            x.numpy().tofile(x_path)
            val.numpy().tofile(val_path)
            idx.numpy().astype(np.int32).tofile(idx_path)
    elif case_name == "TopkOnBoardTest.test_operation_tensor_2_8_4_topk":
        x_path = Path(output, 'x.bin')
        val_path = Path(output, 'val.bin')
        idx_path = Path(output, 'idx.bin')
        shape0 = 2
        shape1 = 8
        k = 4
        complete = x_path.exists() and val_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            # x = torch.randn((shape0, shape1))
            x = torch.arange(shape0 * shape1, dtype=torch.float32).reshape(shape0, shape1)
            val, idx = x.topk(k, dim=1, largest=True, sorted=True)
            x.numpy().tofile(x_path)
            val.numpy().tofile(val_path)
            idx.numpy().astype(np.int32).tofile(idx_path)
    elif case_name == "TopkOnBoardTest.test_operation_tensor_1_8_4_topk":
        x_path = Path(output, 'x.bin')
        val_path = Path(output, 'val.bin')
        idx_path = Path(output, 'idx.bin')
        shape0 = 1
        shape1 = 8
        k = 4
        complete = x_path.exists() and val_path.exists()
        x = torch.arange(shape0 * shape1, dtype=torch.float32).reshape(shape0, shape1)
        val, idx = x.topk(k, dim=1, largest=True, sorted=True)
        x.numpy().tofile(x_path)
        val.numpy().tofile(val_path)
        idx.numpy().astype(np.int32).tofile(idx_path)
    elif case_name == "TopkOnBoardTest.test_operation_tensor_2_288_15_topk":
        x_path = Path(output, 'x.bin')
        val_path = Path(output, 'val.bin')
        idx_path = Path(output, 'idx.bin')
        shape0 = 2
        shape1 = 288
        k = 15
        x = torch.arange(shape0 * shape1, dtype=torch.float32).reshape(shape0, shape1)
        val, idx = x.topk(k, dim=1, largest=True, sorted=True)
        x.numpy().tofile(x_path)
        val.numpy().tofile(val_path)
        idx.numpy().astype(np.int32).tofile(idx_path)
    elif case_name == "TopkOnBoardTest.test_operation_tensor_2_288_15_topk_reverse":
        x_path = Path(output, 'x.bin')
        val_path = Path(output, 'val.bin')
        idx_path = Path(output, 'idx.bin')
        shape0 = 2
        shape1 = 288
        k = 15
        x = torch.randn((shape0, shape1), dtype=torch.float32)
        val, idx = x.topk(k, dim=1, largest=False, sorted=True)
        x.numpy().tofile(x_path)
        val.numpy().tofile(val_path)
        idx.numpy().astype(np.int32).tofile(idx_path)
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
        "TopkOnBoardTest.test_operation_tensor_128_32_32_topk",
    ]
    # 函数调用
    ret: bool = True
    for cs in case_name_list:
        output: Path = Path(g_src_root, "build/output/bin/golden", cs).resolve()
        output.mkdir(parents=True, exist_ok=True)
        ret = gen_topk_golden(case_name=cs, output=output)
    return ret


if __name__ == "__main__":
    exit(0 if main() else 1)
