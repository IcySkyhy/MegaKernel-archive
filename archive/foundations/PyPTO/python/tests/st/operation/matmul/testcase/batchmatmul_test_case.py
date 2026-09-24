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
pypto.batchmatmul ST测试用例配置
用于System Test自动化测试框架
"""

from dataclasses import dataclass

import torch

import pypto


@dataclass
class BatchMatmulConfig:
    a_shape: list
    b_shape: list
    out_shape: list
    tile_shape: tuple[list, list, list]
    view_shape: list
    out_dtype: pypto.DataType
    in_dtype: pypto.DataType
    a_trans: bool = False
    b_trans: bool = False
    is_acc: bool = False
    dim: int = 3
    b_format: str = "ND"
    a_format: str = "ND"

    DTYPE_CONFIG = {
        "DT_FP16": {"pto": pypto.DT_FP16, "torch": torch.float16, "atol": 1e-3, "rtol": 1e-3},
        "DT_FP32": {"pto": pypto.DT_FP32, "torch": torch.float32, "atol": 1e-3, "rtol": 1e-3},
        "DT_BF16": {"pto": pypto.DT_BF16, "torch": torch.bfloat16, "atol": 1e-2, "rtol": 1e-2},
        "DT_INT8": {"pto": pypto.DT_INT8, "torch": torch.int8, "atol": 0, "rtol": 0},
        "DT_INT32": {"pto": pypto.DT_INT32, "torch": torch.int32, "atol": 0, "rtol": 0},
    }

    @classmethod
    def from_test_case(cls, case: dict) -> "BatchMatmulConfig":
        return cls(
            a_shape=case["a_shape"],
            b_shape=case["b_shape"],
            out_shape=case["out_shape"],
            tile_shape=tuple(case["tileshape"]),
            view_shape=case["viewshape"],
            out_dtype=cls.DTYPE_CONFIG[case["c_dtype"]]["pto"],
            in_dtype=cls.DTYPE_CONFIG[case["a_dtype"]]["pto"],
            a_trans=case["a_trans"],
            b_trans=case["b_trans"],
            is_acc=case["is_acc"],
            dim=case["dim"],
            a_format=case.get("a_format", "ND"),
            b_format=case.get("b_format", "ND"),
        )

    @classmethod
    def get_torch_dtype(cls, dtype_str: str) -> torch.dtype:
        return cls.DTYPE_CONFIG[dtype_str]["torch"]

    @classmethod
    def get_tolerance(cls, dtype_str: str) -> tuple[float, float]:
        info = cls.DTYPE_CONFIG[dtype_str]
        return info["atol"], info["rtol"]

    def get_logical_dims_3d(self):
        """获取3D矩阵乘的逻辑维度 (b, m, k, n)"""
        b = max(self.a_shape[0], self.b_shape[0])
        m = self.a_shape[2] if self.a_trans else self.a_shape[1]
        k = self.a_shape[1] if self.a_trans else self.a_shape[2]
        n = self.b_shape[1] if self.b_trans else self.b_shape[2]
        return b, m, k, n

    def get_logical_dims_4d(self):
        """获取4D矩阵乘的逻辑维度 (b0, b1, m, k, n)"""
        b0 = max(self.a_shape[0], self.b_shape[0])
        b1 = max(self.a_shape[1], self.b_shape[1])
        m = self.a_shape[3] if self.a_trans else self.a_shape[2]
        k = self.a_shape[2] if self.a_trans else self.a_shape[3]
        n = self.b_shape[2] if self.b_trans else self.b_shape[3]
        return b0, b1, m, k, n


BASIC_3D_TESTS = [
    {
        "id": "3DACCTEST1",
        "name": "fp16_3d_nz_out_fp32_gcc",
        "desc": "FP16 3D NZ输入FP32输出",
        "a_shape": [5, 215, 251],
        "b_shape": [5, 251, 451],
        "out_shape": [5, 215, 451],
        "dim": 3,
        "a_dtype": "DT_FP16",
        "b_dtype": "DT_FP16",
        "c_dtype": "DT_FP32",
        "a_format": "ND",
        "b_format": "ND",
        "a_trans": False,
        "b_trans": False,
        "is_acc": True,
        "viewshape": [4, 106, 374],
        "tileshape": [[32, 288], [64, 128], [96, 192]],
        "products": ["950", "910"],
    },
    {
        "id": "3DACCTEST2",
        "name": "int8_3d_nd_out_fp16_trans_a",
        "desc": "INT8 3D NZ ND 输入INT8输出+A转置",
        "a_shape": [4, 256, 129],
        "b_shape": [4, 256, 399],
        "out_shape": [4, 160, 399],
        "dim": 3,
        "a_dtype": "DT_INT8",
        "b_dtype": "DT_INT8",
        "c_dtype": "DT_INT32",
        "a_format": "NZ",
        "b_format": "ND",
        "a_trans": True,
        "b_trans": False,
        "is_acc": True,
        "viewshape": [3, 128, 396],
        "tileshape": [[64, 64], [32, 160], [32, 224]],
        "products": ["950"],
    },
]


BASIC_4D_TESTS = [
    {
        "id": "4DACCTEST1",
        "name": "fp16_4d_nz_out_fp32",
        "desc": "FP16 4D NZ输入FP32输出",
        "a_shape": [5, 4, 127, 192],
        "b_shape": [5, 4, 192, 289],
        "out_shape": [5, 4, 128, 304],
        "dim": 4,
        "a_dtype": "DT_FP16",
        "b_dtype": "DT_FP16",
        "c_dtype": "DT_FP32",
        "a_format": "NZ",
        "b_format": "NZ",
        "a_trans": False,
        "b_trans": False,
        "is_acc": True,
        "viewshape": [3, 3, 64, 256],
        "tileshape": [[64, 64], [64, 128], [128, 128]],
        "products": ["950"],
    },
    {
        "id": "4DACCTEST2",
        "name": "int8_4d_nz_out_fp16_trans_a",
        "desc": "INT8 4D ND输入INT8输出+A转置",
        "a_shape": [4, 3, 312, 165],
        "b_shape": [4, 3, 312, 145],
        "out_shape": [4, 3, 165, 145],
        "dim": 4,
        "a_dtype": "DT_INT8",
        "b_dtype": "DT_INT8",
        "c_dtype": "DT_INT32",
        "a_format": "ND",
        "b_format": "ND",
        "a_trans": True,
        "b_trans": False,
        "is_acc": True,
        "viewshape": [3, 2, 64, 141],
        "tileshape": [[64, 192], [128, 256], [128, 128]],
        "products": ["950", "910"],
    },
]
