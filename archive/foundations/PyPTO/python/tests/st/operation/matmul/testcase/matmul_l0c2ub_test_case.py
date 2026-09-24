#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
pypto MatmulL0C2UB ST测试用例配置
用于System Test自动化测试框架
算子公式：C = A @ B + C（逐元素加）
"""

from dataclasses import dataclass

import torch

import pypto


@dataclass
class MatmulL0C2UBConfig:
    """MatmulL0C2UB 配置参数"""

    shape: tuple[int, int, int]
    tile_shape: tuple[list, list, list]
    vec_tile_shape: tuple[int, int]
    view_shape: tuple[int, int]
    a_dtype: pypto.DataType
    b_dtype: pypto.DataType
    c_dtype: pypto.DataType
    out_dtype: pypto.DataType
    a_trans: bool = False
    b_trans: bool = False
    extend_params: dict = None

    DTYPE_CONFIG = {
        "DT_FP16": {"pto": pypto.DT_FP16, "torch": torch.float16, "atol": 1e-3, "rtol": 1e-3},
        "DT_FP32": {"pto": pypto.DT_FP32, "torch": torch.float32, "atol": 1e-3, "rtol": 1e-3},
        "DT_BF16": {"pto": pypto.DT_BF16, "torch": torch.bfloat16, "atol": 1e-2, "rtol": 1e-2},
        "DT_INT8": {"pto": pypto.DT_INT8, "torch": torch.int8, "atol": 0, "rtol": 0},
        "DT_INT32": {"pto": pypto.DT_INT32, "torch": torch.int32, "atol": 0, "rtol": 0},
    }

    @classmethod
    def from_test_case(cls, case: dict) -> "MatmulL0C2UBConfig":
        """从测试用例字典构造配置"""
        return cls(
            shape=(case["m"], case["k"], case["n"]),
            tile_shape=tuple(case["tileshape"]),
            vec_tile_shape=tuple(case["vec_tileshape"]),
            view_shape=tuple(case["viewshape"]),
            a_dtype=cls.DTYPE_CONFIG[case["a_dtype"]]["pto"],
            b_dtype=cls.DTYPE_CONFIG[case["b_dtype"]]["pto"],
            c_dtype=cls.DTYPE_CONFIG[case["c_dtype"]]["pto"],
            out_dtype=cls.DTYPE_CONFIG[case["c_dtype"]]["pto"],
            a_trans=case.get("a_trans", False),
            b_trans=case.get("b_trans", False),
            extend_params=case.get("extend_params", {}),
        )

    @classmethod
    def get_torch_dtype(cls, dtype_str: str) -> torch.dtype:
        """获取 PyTorch dtype"""
        return cls.DTYPE_CONFIG[dtype_str]["torch"]

    @classmethod
    def get_tolerance(cls, dtype_str: str) -> tuple[float, float]:
        """获取精度容差"""
        info = cls.DTYPE_CONFIG[dtype_str]
        return info["atol"], info["rtol"]


L0C2UB_TESTS = [
    {
        "id": "L01",
        "name": "fp16_TInsert",
        "desc": "FP16 输入 FP16 输出, L0C_COPY_UB小搬大",
        "m": 129,
        "k": 256,
        "n": 513,
        "a_dtype": "DT_FP16",
        "b_dtype": "DT_FP16",
        "c_dtype": "DT_FP16",
        "a_format": "ND",
        "b_format": "ND",
        "a_trans": False,
        "b_trans": False,
        "viewshape": [128, 128],
        "tileshape": [[64, 128], [128, 128], [64, 128]],
        "vec_tileshape": [128, 128],
        "extend_params": {},
        "products": ["950"],
    },
    {
        "id": "L02",
        "name": "bf16_TExtract",
        "desc": "BF16 输入 BF16 输出, L0C_COPY_UB大搬小",
        "m": 3,
        "k": 128,
        "n": 129,
        "a_dtype": "DT_BF16",
        "b_dtype": "DT_BF16",
        "c_dtype": "DT_BF16",
        "a_format": "ND",
        "b_format": "ND",
        "a_trans": False,
        "b_trans": False,
        "viewshape": [128, 128],
        "tileshape": [[128, 128], [128, 128], [128, 128]],
        "vec_tileshape": [64, 64],
        "extend_params": {},
        "products": ["950"],
    },
    {
        "id": "L03",
        "name": "fp32_toGM",
        "desc": "不满足L0C_COPY_UB约束用例",
        "m": 127,
        "k": 256,
        "n": 511,
        "a_dtype": "DT_FP32",
        "b_dtype": "DT_FP32",
        "c_dtype": "DT_FP32",
        "a_format": "ND",
        "b_format": "ND",
        "a_trans": False,
        "b_trans": False,
        "viewshape": [128, 128],
        "tileshape": [[96, 192], [128, 128], [64, 128]],
        "vec_tileshape": [128, 32],
        "extend_params": {},
        "products": ["950"],
    },
]
