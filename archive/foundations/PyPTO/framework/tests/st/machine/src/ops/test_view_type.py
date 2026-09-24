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

project_root = os.path.dirname(os.path.abspath(__file__))  # 当前脚本目录
golden_parent = os.path.join(project_root, "../../../../")  # 假设 golden 在上级目录
sys.path.insert(0, golden_parent)


if __name__ == "__main__":
    """单独调试时配置"""
    # 日志级别
    logging.basicConfig(
        format="%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s",
        level=logging.DEBUG,
    )
    # 系统 import 路径
    g_src_root: Path = Path(Path(__file__).parent, "../../../../../").resolve()
    logging.debug("SrcRoot: %s", g_src_root)
    g_ctrl_path: Path = Path(g_src_root, "scripts")
    if str(g_ctrl_path) not in sys.path:
        sys.path.append(str(g_ctrl_path))
    from golden_register import (
        GoldenRegister,
    )  # 单独调试 import 失败, 需确认上文中 '系统 import 路径' 配置正确
else:
    from golden_register import GoldenRegister


def tensor_tofile(t: torch.Tensor, output: Path):
    input_file_bin = open(str(output), "wb")
    for each in t:
        if t.dtype == torch.bfloat16:
            input_file_bin.write(each.view(torch.int16).numpy().tobytes())
        elif t.dtype == torch.float16:
            input_file_bin.write(each.view(torch.int16).numpy().tobytes())
        elif t.dtype == torch.float32:
            input_file_bin.write(each.view(torch.int32).numpy().tobytes())
        elif t.dtype == torch.int32:
            input_file_bin.write(each.numpy().tobytes())
        elif t.dtype == torch.int8:
            input_file_bin.write(each.numpy().tobytes())
        else:
            raise ValueError(
                f"Unsupported dtype: {t.dtype}, please add in framework/tests/st/operator/src/test_view_type.py"
            )
    input_file_bin.close()


def view_type_entry(mkn, origin_dtype, dst_dtype, output_dir: Path):
    x_path = Path(output_dir, 'x.bin')
    result_path = Path(output_dir, 'result.bin')

    m, k, n = mkn
    x_shape = [m, k, n]

    x = torch.randint(-128, 127, x_shape, dtype=origin_dtype)
    result = x.view(dst_dtype)

    tensor_tofile(x, x_path)
    tensor_tofile(result, result_path)


def view_type_cast_entry(mkn, origin_dtype, dst_dtype, output_dir: Path, cast_dtype):
    x_path = Path(output_dir, 'x.bin')
    result_path = Path(output_dir, 'result.bin')

    m, k, n = mkn
    x_shape = [m, k, n]

    x = torch.randint(-128, 127, x_shape, dtype=origin_dtype)
    result = x.view(dst_dtype).to(cast_dtype)

    tensor_tofile(x, x_path)
    tensor_tofile(result, result_path)


def view_type_quant_test_entry(output_dir: Path):
    x_path = Path(output_dir, 'x.bin')
    result_path = Path(output_dir, 'result.bin')

    t, d_kv = 64, 512
    n_kv = 1

    def dynamic_quant(x):
        x_dtype = x.dtype
        x_fp32 = x.to(torch.float32)
        max_value = torch.amax(torch.abs(x_fp32), dim=-1, keepdim=True)
        scale_quant = 127.0 / max_value
        y_fp32 = x_fp32 * scale_quant
        y_fp32 = y_fp32.view(x.shape)
        y_int32 = torch.round(y_fp32).to(torch.int32)
        y_int8 = torch.trunc(y_int32.to(x_dtype)).to(torch.int8)
        scale_dequant = 1.0 / scale_quant
        return y_int8, scale_dequant

    k_nope = torch.ones((t, n_kv, d_kv), dtype=torch.bfloat16)
    x = k_nope
    tensor_tofile(x, x_path)

    k_nope = k_nope.reshape((t, n_kv, 4, d_kv // 4))
    k_nope_int8, k_scale = dynamic_quant(k_nope)

    k_nope_int8 = k_nope_int8.reshape((t, n_kv, d_kv))
    k_scale = k_scale.reshape((t, n_kv, 4))
    k_scale_view_int8 = k_scale.view(torch.int8)
    k_combined = torch.cat((k_nope_int8, k_scale_view_int8), dim=-1)

    result = k_combined
    tensor_tofile(result, result_path)


def view_type_dequant_test_entry(output_dir: Path):
    x_path = Path(output_dir, 'x.bin')
    result_path = Path(output_dir, 'result.bin')

    d_kv, d_r = 512, 64
    n_kv = 1
    selected_count = 2048
    cache_combined = torch.ones((selected_count, n_kv, d_kv + 2 * d_r + 4 * 4), dtype=torch.int8)
    cache_nope_int8 = cache_combined[:, :, :d_kv]
    cache_rope_int8 = cache_combined[:, :, d_kv:d_kv + 2 * d_r]
    cache_scale_int8 = cache_combined[:, :, d_kv + 2 * d_r:]

    cache_nope_fp16 = cache_nope_int8.to(torch.float16)
    cache_nope_fp32 = cache_nope_fp16.to(torch.float32)
    cache_rope_bf16 = cache_rope_int8.view(torch.bfloat16)
    cache_scale_fp32 = cache_scale_int8.view(torch.float32).reshape((selected_count, n_kv, 4, 1))

    cache_nope_fp32 = cache_nope_fp32.reshape((selected_count, n_kv, 4, 128))
    cache_nope = cache_nope_fp32 * cache_scale_fp32
    cache_nope_bf16 = cache_nope.to(torch.bfloat16)
    cache_nope_bf16_res = cache_nope_bf16.reshape((selected_count, n_kv, 512))
    cache = torch.cat((cache_nope_bf16_res, cache_rope_bf16), dim=-1)

    x = cache_combined
    result = cache
    tensor_tofile(x, x_path)
    tensor_tofile(result, result_path)


@GoldenRegister.reg_golden_func(
    case_names=[
        "ViewType.int8_2_float32",
        "ViewType.int8_2_bfloat16",
        "ViewType.int8_2_float16",
        "ViewType.float32_2_int8",
        "ViewType.bfloat16_2_int8",
        "ViewType.float16_2_int8",
        "ViewType.float16_2_float32",
        "ViewType.float32_2_float16",
        "ViewType.bfloat16_2_float32",
        "ViewType.float32_2_bfloat16",
        "ViewType.int8_2_bfloat16_cast_fp32",
        "ViewType.int8_2_float16_cast_fp32",
        "ViewType.quant_test_bf16_2_int8",
        "ViewType.dequant_test_bf16_2_int8",
    ]
)
def view_type_func(case_name: str, output: Path) -> bool:
    if case_name == "ViewType.int8_2_float32":
        view_type_entry((4, 32, 1024), torch.int8, torch.float32, output)
    elif case_name == "ViewType.int8_2_bfloat16":
        view_type_entry((4, 32, 1024), torch.int8, torch.bfloat16, output)
    elif case_name == "ViewType.int8_2_float16":
        view_type_entry((4, 32, 1024), torch.int8, torch.float16, output)
    elif case_name == "ViewType.float32_2_int8":
        view_type_entry((4, 32, 1024), torch.float32, torch.int8, output)
    elif case_name == "ViewType.bfloat16_2_int8":
        view_type_entry((4, 32, 1024), torch.bfloat16, torch.int8, output)
    elif case_name == "ViewType.float16_2_int8":
        view_type_entry((4, 32, 1024), torch.float16, torch.int8, output)
    elif case_name == "ViewType.float16_2_float32":
        view_type_entry((4, 32, 1024), torch.float16, torch.float32, output)
    elif case_name == "ViewType.float32_2_float16":
        view_type_entry((4, 32, 1024), torch.float32, torch.float16, output)
    elif case_name == "ViewType.bfloat16_2_float32":
        view_type_entry((4, 32, 1024), torch.bfloat16, torch.float32, output)
    elif case_name == "ViewType.float32_2_bfloat16":
        view_type_entry((4, 32, 1024), torch.float32, torch.bfloat16, output)
    elif case_name == "ViewType.int8_2_bfloat16_cast_fp32":
        view_type_cast_entry((4, 32, 1024), torch.int8, torch.bfloat16, output, torch.float32)
    elif case_name == "ViewType.int8_2_float16_cast_fp32":
        view_type_cast_entry((4, 32, 1024), torch.int8, torch.float16, output, torch.float32)
    elif case_name == "ViewType.quant_test_bf16_2_int8":
        view_type_quant_test_entry(output)
    elif case_name == "ViewType.dequant_test_bf16_2_int8":
        view_type_dequant_test_entry(output)
    else:
        logging.error("Can't get func to gen golden, Case(%s)", case_name)
        return False
    return True


def main() -> bool:
    """
    单独调试 入口函数
    """
    case_name_list: List[str] = []

    for cs in case_name_list:
        output = Path(g_src_root, "build/tests/st/golden", cs).resolve()
        output.mkdir(parents=True, exist_ok=True)
        ret = view_type_func(case_name=cs, output=output)

    return ret


if __name__ == "__main__":
    exit(0 if main() else 1)
