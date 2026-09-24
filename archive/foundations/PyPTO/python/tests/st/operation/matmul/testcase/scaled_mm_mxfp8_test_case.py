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
pypto.scaled_mm ST测试用例配置
用于System Test自动化测试框架
"""

from dataclasses import dataclass
from typing import Tuple

import torch

import pypto


@dataclass
class ScaledMMConfig:
    ori_shape: Tuple[int, int, int]
    output_shape: Tuple[int, int]
    m_tile_shape: Tuple[int, int]
    k_tile_shape: Tuple[int, int]
    n_tile_shape: Tuple[int, int]
    view_shape: Tuple[int, int]
    in_dtype: pypto.DataType
    out_dtype: pypto.DataType
    scale_a_trans: bool = False
    scale_b_trans: bool = False
    a_trans: bool = False
    b_trans: bool = False
    a_format: str = "ND"
    b_format: str = "ND"
    c_format: str = "ND"
    has_bias: bool = False
    enable_ksplit: bool = False

    DTYPE_CONFIG = {
        "DT_FP8E4M3": {"pto": pypto.DataType.DT_FP8E4M3, "torch": torch.float8_e4m3fn},
        "DT_FP8E5M2": {"pto": pypto.DataType.DT_FP8E5M2, "torch": torch.float8_e5m2},
        "DT_FP16": {"pto": pypto.DataType.DT_FP16, "torch": torch.float16},
        "DT_FP32": {"pto": pypto.DataType.DT_FP32, "torch": torch.float32},
        "DT_BF16": {"pto": pypto.DataType.DT_BF16, "torch": torch.bfloat16},
    }

    TOLERANCE_CONFIG = {
        "DT_FP16": {"atol": 1e-3, "rtol": 1e-3},
        "DT_FP32": {"atol": 1e-3, "rtol": 1e-3},
        "DT_BF16": {"atol": 1e-2, "rtol": 1e-2},
        "DT_FP8E4M3": {"atol": 1e-3, "rtol": 1e-3},
        "DT_FP8E5M2": {"atol": 1e-3, "rtol": 1e-3},
    }

    @classmethod
    def from_test_case(cls, case: dict) -> "ScaledMMConfig":
        return cls(
            ori_shape=tuple(case["ori_shape"]),
            output_shape=tuple(case["output_shape"]),
            m_tile_shape=tuple(case["m_tile_shape"]),
            k_tile_shape=tuple(case["k_tile_shape"]),
            n_tile_shape=tuple(case["n_tile_shape"]),
            view_shape=tuple(case["view_shape"]),
            in_dtype=cls.DTYPE_CONFIG[case["in_dtype"]]["pto"],
            out_dtype=cls.DTYPE_CONFIG[case["out_dtype"]]["pto"],
            scale_a_trans=case.get("scale_a_trans", False),
            scale_b_trans=case.get("scale_b_trans", False),
            a_trans=case.get("a_trans", False),
            b_trans=case.get("b_trans", False),
            a_format=case.get("a_format", "ND"),
            b_format=case.get("b_format", "ND"),
            c_format=case.get("c_format", "ND"),
            has_bias=case.get("has_bias", False),
            enable_ksplit=case.get("enable_ksplit", False),
        )

    @classmethod
    def get_tolerance(cls, dtype_str: str) -> Tuple[float, float]:
        info = cls.TOLERANCE_CONFIG[dtype_str]
        return info["atol"], info["rtol"]

    @classmethod
    def get_torch_dtype(cls, dtype_str: str) -> torch.dtype:
        return cls.DTYPE_CONFIG[dtype_str]["torch"]

    @classmethod
    def pto_to_torch(cls, pto_dtype: pypto.DataType) -> torch.dtype:
        for info in cls.DTYPE_CONFIG.values():
            if info["pto"] == pto_dtype:
                return info["torch"]
        raise ValueError(f"Unsupported pypto.DataType: {pto_dtype}")


SCALED_MM_TESTS = [
    {
        "id": "SCALEDMM_BASIC",
        "name": "scaled_mm_basic_no_bias",
        "desc": "2D scaled_mm 无bias场景",
        "ori_shape": [243, 193, 169],
        "output_shape": [243, 169],
        "m_tile_shape": [64, 128],
        "k_tile_shape": [128, 256],
        "n_tile_shape": [64, 256],
        "view_shape": [154, 69],
        "in_dtype": "DT_FP8E4M3",
        "out_dtype": "DT_FP32",
        "scale_a_trans": True,
        "scale_b_trans": False,
        "a_trans": False,
        "b_trans": True,
        "a_format": "ND",
        "b_format": "ND",
        "c_format": "ND",
        "has_bias": False,
        "enable_ksplit": True,
        "products": ["950"],
    },
    {
        "id": "SCALEDMM_BIAS",
        "name": "scaled_mm_with_bias",
        "desc": "2D scaled_mm 带bias场景",
        "ori_shape": [383, 192, 129],
        "output_shape": [384, 144],
        "m_tile_shape": [64, 64],
        "k_tile_shape": [64, 256],
        "n_tile_shape": [256, 256],
        "view_shape": [192, 32],
        "in_dtype": "DT_FP8E4M3",
        "out_dtype": "DT_BF16",
        "scale_a_trans": True,
        "scale_b_trans": False,
        "a_trans": False,
        "b_trans": True,
        "a_format": "NZ",
        "b_format": "NZ",
        "c_format": "ND",
        "has_bias": True,
        "enable_ksplit": False,
        "products": ["950"],
    },
]
