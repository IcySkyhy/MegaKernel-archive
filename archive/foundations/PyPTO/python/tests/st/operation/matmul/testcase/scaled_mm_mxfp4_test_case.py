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
pypto.scaled_mm FP4输入 ST测试用例配置
用于System Test自动化测试框架
"""

from dataclasses import dataclass
from typing import Tuple

import torch

import pypto


@dataclass
class MXFP4Config:
    a_shape: Tuple[int, int, int]
    b_shape: Tuple[int, int, int]
    m_tile_shape: Tuple[int, int]
    k_tile_shape: Tuple[int, int]
    n_tile_shape: Tuple[int, int]
    view_shape: Tuple[int, int]
    in_dtype: pypto.DataType
    out_dtype: pypto.DataType
    a_trans: bool = False
    b_trans: bool = False
    scale_a_trans: bool = False
    scale_b_trans: bool = False
    a_format: str = "ND"
    b_format: str = "ND"
    c_format: str = "ND"
    has_bias: bool = True

    DTYPE_CONFIG = {
        "DT_FP4_E2M1": {"pto": pypto.DataType.DT_FP4_E2M1, "torch": torch.uint8},
        "DT_FP4_E1M2": {"pto": pypto.DataType.DT_FP4_E1M2, "torch": torch.uint8},
        "DT_FP4_E2M1X2": {"pto": pypto.DataType.DT_FP4_E2M1X2, "torch": torch.uint8},
        "DT_FP16": {"pto": pypto.DataType.DT_FP16, "torch": torch.float16},
        "DT_FP32": {"pto": pypto.DataType.DT_FP32, "torch": torch.float32},
        "DT_BF16": {"pto": pypto.DataType.DT_BF16, "torch": torch.bfloat16},
    }

    TOLERANCE_CONFIG = {
        "DT_FP16": {"atol": 1e-3, "rtol": 1e-3},
        "DT_FP32": {"atol": 1e-3, "rtol": 1e-3},
        "DT_BF16": {"atol": 1e-2, "rtol": 1e-2},
    }

    @classmethod
    def from_test_case(cls, case: dict) -> "MXFP4Config":
        return cls(
            a_shape=tuple(case["a_shape"]),
            b_shape=tuple(case["b_shape"]),
            m_tile_shape=tuple(case["m_tile_shape"]),
            k_tile_shape=tuple(case["k_tile_shape"]),
            n_tile_shape=tuple(case["n_tile_shape"]),
            view_shape=tuple(case["view_shape"]),
            in_dtype=cls.DTYPE_CONFIG[case["in_dtype"]]["pto"],
            out_dtype=cls.DTYPE_CONFIG[case["out_dtype"]]["pto"],
            a_trans=case.get("a_trans", False),
            b_trans=case.get("b_trans", False),
            scale_a_trans=case.get("scale_a_trans", False),
            scale_b_trans=case.get("scale_b_trans", False),
            a_format=case.get("a_format", "ND"),
            b_format=case.get("b_format", "ND"),
            c_format=case.get("c_format", "ND"),
            has_bias=case.get("has_bias", True),
        )

    @classmethod
    def get_tolerance(cls, dtype_str: str) -> Tuple[float, float]:
        info = cls.TOLERANCE_CONFIG[dtype_str]
        return info["atol"], info["rtol"]

    @classmethod
    def pto_to_torch(cls, pto_dtype: pypto.DataType) -> torch.dtype:
        for info in cls.DTYPE_CONFIG.values():
            if info["pto"] == pto_dtype:
                return info["torch"]
        raise ValueError(f"Unsupported pypto.DataType: {pto_dtype}")


MXFP4_TESTS = [
    {
        "id": "FP4_01",
        "name": "fp4_e2m1_scaled_mm_with_bias",
        "desc": "FP4 E2M1 scaled_mm 带bias, B转置, scale_A转置",
        "a_shape": [129, 254],
        "b_shape": [254, 511],
        "m_tile_shape": [64, 64],
        "k_tile_shape": [64, 256],
        "n_tile_shape": [256, 256],
        "view_shape": [51, 498],
        "in_dtype": "DT_FP4_E2M1",
        "out_dtype": "DT_FP16",
        "a_trans": False,
        "b_trans": True,
        "scale_a_trans": True,
        "scale_b_trans": False,
        "a_format": "ND",
        "b_format": "ND",
        "c_format": "ND",
        "has_bias": False,
        "products": ["950"],
    },
    {
        "id": "FP4_01",
        "name": "fp4_e2m1_scaled_mm_with_bias",
        "desc": "FP4 E2M1 scaled_mm 带bias, A,B转置, scale_A,scale_B转置",
        "a_shape": [154, 194],
        "b_shape": [194, 361],
        "m_tile_shape": [64, 128],
        "k_tile_shape": [64, 128],
        "n_tile_shape": [64, 256],
        "view_shape": [64, 64],
        "in_dtype": "DT_FP4_E2M1",
        "out_dtype": "DT_FP16",
        "a_trans": True,
        "b_trans": True,
        "scale_a_trans": True,
        "scale_b_trans": True,
        "a_format": "ND",
        "b_format": "ND",
        "c_format": "ND",
        "has_bias": True,
        "products": ["950"],
    },
]
