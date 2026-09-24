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
pypto.scaled_mm 3D/4D ST测试用例配置
用于System Test自动化测试框架
"""

from dataclasses import dataclass
from typing import Tuple, Union

import torch

import pypto


@dataclass
class BaseScaledConfig:
    a_shape: Tuple[int, ...]
    b_shape: Tuple[int, ...]
    out_shape: Tuple[int, ...]
    m_tile_shape: Tuple[int, int]
    k_tile_shape: Tuple[int, int]
    n_tile_shape: Tuple[int, int]
    view_shape: Union[Tuple[int, int, int], Tuple[int, int, int, int]]
    in_dtype: pypto.DataType
    out_dtype: pypto.DataType
    a_trans: bool = False
    b_trans: bool = False
    scale_a_trans: bool = False
    scale_b_trans: bool = False
    a_format: str = "ND"
    b_format: str = "ND"
    c_format: str = "ND"
    has_bias: bool = False
    bias_batch: int = 0
    bias_shape_type: str = "1_n"
    enable_ksplit: bool = False
    scale: float = 0.0
    has_scale_tensor: bool = False
    relu_type: pypto.ReLuType = pypto.ReLuType.NO_RELU

    DTYPE_CONFIG = {
        "DT_FP8E4M3": {"pto": pypto.DataType.DT_FP8E4M3, "torch": torch.float8_e4m3fn},
        "DT_FP8E5M2": {"pto": pypto.DataType.DT_FP8E5M2, "torch": torch.float8_e5m2},
        "DT_FP16": {"pto": pypto.DataType.DT_FP16, "torch": torch.float16},
        "DT_FP32": {"pto": pypto.DataType.DT_FP32, "torch": torch.float32},
        "DT_INT8": {"pto": pypto.DataType.DT_INT8, "torch": torch.int8},
    }

    TOLERANCE_CONFIG = {
        "DT_FP16": {"atol": 1e-3, "rtol": 1e-3},
        "DT_FP32": {"atol": 1e-3, "rtol": 1e-3},
        "DT_FP8E4M3": {"atol": 1e-3, "rtol": 1e-3},
        "DT_FP8E5M2": {"atol": 1e-3, "rtol": 1e-3},
        "DT_INT8": {"atol": 1, "rtol": 0},
    }

    @classmethod
    def from_test_case(cls, case: dict):
        return cls(
            a_shape=tuple(case["a_shape"]),
            b_shape=tuple(case["b_shape"]),
            out_shape=tuple(case["out_shape"]),
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
            has_bias=case.get("has_bias", False),
            bias_batch=case.get("bias_batch", 0),
            bias_shape_type=case.get("bias_shape_type", "1_n"),
            enable_ksplit=case.get("enable_ksplit", False),
            scale=case.get("scale", 0.0),
            has_scale_tensor=case.get("has_scale_tensor", False),
            relu_type=case.get("relu_type", pypto.ReLuType.NO_RELU),
        )

    @classmethod
    def get_torch_dtype(cls, dtype_str: str) -> torch.dtype:
        return cls.DTYPE_CONFIG[dtype_str]["torch"]

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


@dataclass
class ScaledBmmConfig(BaseScaledConfig):
    @property
    def batch_a(self) -> int:
        return self.a_shape[0]

    @property
    def batch_b(self) -> int:
        return self.b_shape[0]

    @property
    def batch(self) -> int:
        return max(self.batch_a, self.batch_b)

    def get_logical_dims_3d(self):
        m = self.a_shape[2] if self.a_trans else self.a_shape[1]
        k = self.a_shape[1] if self.a_trans else self.a_shape[2]
        n = self.b_shape[1] if self.b_trans else self.b_shape[2]
        return self.batch, m, k, n

    def get_logical_dims_4d(self):
        b0 = max(self.a_shape[0], self.b_shape[0])
        b1 = max(self.a_shape[1], self.b_shape[1])
        m = self.a_shape[3] if self.a_trans else self.a_shape[2]
        k = self.a_shape[2] if self.a_trans else self.a_shape[3]
        n = self.b_shape[2] if self.b_trans else self.b_shape[3]
        return b0, b1, m, k, n


# ================= 测试用例 =================
SCALED_BMM_TESTS = [
    # 3D bias (b,1,n) + pertensor scale
    {
        "id": "BMM3D_BIAS_B1N_UNALIGN_QUANT_PERTENSOR",
        "name": "scaled_bmm_3d_bias_b1n_quant_pertensor",
        "a_shape": [4, 255, 192],
        "b_shape": [1, 224, 192],
        "out_shape": [4, 256, 224],
        "m_tile_shape": [64, 64],
        "k_tile_shape": [128, 256],
        "n_tile_shape": [256, 256],
        "view_shape": [3, 192, 32],
        "in_dtype": "DT_FP8E4M3",
        "out_dtype": "DT_INT8",
        "b_trans": True,
        "a_trans": False,
        "scale_a_trans": True,
        "scale_b_trans": True,
        "a_format": "NZ",
        "c_format": "ND",
        "b_format": "NZ",
        "has_bias": True,
        "scale": 1.5,
        "bias_batch": 3,
        "bias_shape_type": "b_1_n",
        "enable_ksplit": False,
        "products": ["950"],
    },
    # 3D bias (1,n)
    {
        "id": "BMM3D_BIAS_1N_UNALIGN",
        "name": "scaled_bmm_3d_bias_1n",
        "a_shape": [5, 135, 192],
        "b_shape": [1, 192, 351],
        "out_shape": [5, 135, 351],
        "m_tile_shape": [64, 128],
        "k_tile_shape": [64, 256],
        "n_tile_shape": [256, 256],
        "view_shape": [2, 192, 215],
        "in_dtype": "DT_FP8E4M3",
        "out_dtype": "DT_FP32",
        "scale_a_trans": True,
        "scale_b_trans": False,
        "a_trans": False,
        "b_trans": False,
        "a_format": "ND",
        "b_format": "ND",
        "c_format": "ND",
        "has_bias": True,
        "bias_batch": 2,
        "enable_ksplit": False,
        "products": ["950"],
    },
    # 4D bias (1,n)
    {
        "id": "BMM4D_BIAS_UNALIGN_QUANT_PERCHENNEL",
        "name": "scaled_bmm_4d_bias_quant_perchennel",
        "a_shape": [2, 3, 383, 320],
        "b_shape": [1, 3, 178, 320],
        "out_shape": [2, 3, 384, 178],
        "m_tile_shape": [128, 256],
        "k_tile_shape": [64, 256],
        "n_tile_shape": [64, 192],
        "view_shape": [3, 2, 192, 32],
        "in_dtype": "DT_FP8E4M3",
        "out_dtype": "DT_INT8",
        "a_trans": False,
        "b_trans": True,
        "scale_b_trans": True,
        "scale_a_trans": True,
        "a_format": "NZ",
        "b_format": "ND",
        "c_format": "ND",
        "has_bias": True,
        "bias_batch": 2,
        "has_scale_tensor": True,
        "relu_type": pypto.ReLuType.RELU,
        "enable_ksplit": False,
        "products": ["950"],
    },
]
