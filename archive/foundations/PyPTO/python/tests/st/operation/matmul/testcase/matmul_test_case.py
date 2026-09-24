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
pypto.matmul ST测试用例配置
用于System Test自动化测试框架
"""

from dataclasses import dataclass

import torch

import pypto


@dataclass
class MatmulConfig:
    shape: tuple[int, int, int]
    out_shape: tuple[int, int]
    tile_shape: tuple[list, list, list]
    view_shape: tuple[int, int]
    out_dtype: pypto.DataType
    a_trans: bool = False
    b_trans: bool = False
    a_format: str = "ND"
    b_format: str = "ND"

    DTYPE_CONFIG = {
        "DT_FP16": {"pto": pypto.DT_FP16, "torch": torch.float16, "atol": 1e-3, "rtol": 1e-3},
        "DT_FP32": {"pto": pypto.DT_FP32, "torch": torch.float32, "atol": 1e-3, "rtol": 1e-3},
        "DT_BF16": {"pto": pypto.DT_BF16, "torch": torch.bfloat16, "atol": 1e-2, "rtol": 1e-2},
        "DT_INT8": {"pto": pypto.DT_INT8, "torch": torch.int8, "atol": 0, "rtol": 0},
        "DT_INT32": {"pto": pypto.DT_INT32, "torch": torch.int32, "atol": 0, "rtol": 0},
    }

    @classmethod
    def from_test_case(cls, case: dict) -> "MatmulConfig":
        return cls(
            shape=(case["m"], case["k"], case["n"]),
            out_shape=tuple(case["out_shape"]),
            tile_shape=tuple(case["tileshape"]),
            view_shape=tuple(case["viewshape"]),
            out_dtype=cls.DTYPE_CONFIG[case["c_dtype"]]["pto"],
            a_trans=case["a_trans"],
            b_trans=case["b_trans"],
            a_format=case["a_format"],
            b_format=case["b_format"],
        )

    @classmethod
    def get_torch_dtype(cls, dtype_str: str) -> torch.dtype:
        return cls.DTYPE_CONFIG[dtype_str]["torch"]

    @classmethod
    def get_tolerance(cls, dtype_str: str) -> tuple[float, float]:
        info = cls.DTYPE_CONFIG[dtype_str]
        return info["atol"], info["rtol"]


BASIC_TESTS = [
    {
        "id": "B01",
        "name": "fp16_2d_nd_out_fp32_trans_a",
        "desc": "FP16输入FP32输出+A转置",
        "m": 129,
        "k": 257,
        "n": 513,
        "out_shape": [129, 513],
        "a_dtype": "DT_FP16",
        "b_dtype": "DT_FP16",
        "c_dtype": "DT_FP32",
        "a_format": "ND",
        "b_format": "ND",
        "a_trans": True,
        "b_trans": False,
        "viewshape": [128, 256],
        "tileshape": [[128, 128], [128, 128], [256, 256]],
        "extend_params": {},
        "products": ["950", "910"],
    },
    {
        "id": "B02",
        "name": "bf16_2d_nd_out_fp32_trans_b",
        "desc": "BF16输入FP32输出+B转置",
        "m": 129,
        "k": 255,
        "n": 511,
        "out_shape": [129, 511],
        "a_dtype": "DT_BF16",
        "b_dtype": "DT_BF16",
        "c_dtype": "DT_FP32",
        "a_format": "ND",
        "b_format": "ND",
        "a_trans": False,
        "b_trans": True,
        "viewshape": [64, 256],
        "tileshape": [[64, 64], [64, 128], [128, 128]],
        "extend_params": {},
        "products": ["950", "910"],
    },
    {
        "id": "B03",
        "name": "fp32_2d_nd_out_fp32_trans_both",
        "desc": "FP32输入FP32输出+双转置",
        "m": 127,
        "k": 255,
        "n": 513,
        "out_shape": [127, 513],
        "a_dtype": "DT_FP32",
        "b_dtype": "DT_FP32",
        "c_dtype": "DT_FP32",
        "a_format": "ND",
        "b_format": "ND",
        "a_trans": True,
        "b_trans": True,
        "viewshape": [128, 256],
        "tileshape": [[128, 128], [64, 128], [256, 256]],
        "extend_params": {},
        "products": ["950", "910"],
    },
]

NZ_FORMAT_TESTS = [
    {
        "id": "NZ01",
        "name": "fp16_2d_nz",
        "desc": "FP16 NZ格式",
        "m": 127,
        "k": 256,
        "n": 513,
        "out_shape": [128, 528],
        "a_dtype": "DT_FP16",
        "b_dtype": "DT_FP16",
        "c_dtype": "DT_FP16",
        "a_format": "NZ",
        "b_format": "NZ",
        "a_trans": False,
        "b_trans": False,
        "viewshape": [64, 256],
        "tileshape": [[64, 64], [64, 128], [128, 128]],
        "extend_params": {},
        "alignment": {"inner_axis": 32, "outer_axis": 16},
        "products": ["950"],
    },
]

EXTRA_PARAM_TESTS = [
    {
        "id": "E01",
        "name": "fp16_bias_relu",
        "desc": "FP16带FP16 Bias+ReLU",
        "m": 127,
        "k": 257,
        "n": 511,
        "a_dtype": "DT_FP16",
        "b_dtype": "DT_FP16",
        "c_dtype": "DT_FP16",
        "a_format": "ND",
        "b_format": "ND",
        "a_trans": False,
        "b_trans": False,
        "viewshape": [64, 256],
        "tileshape": [[64, 64], [64, 128], [128, 128]],
        "extend_params": {"bias_tensor": {"dtype": "DT_FP16", "shape": [1, 511]}, "relu_type": "RELU"},
        "products": ["950", "910"],
    },
    {
        "id": "E02",
        "name": "fp16_bias_fp32_trans_a",
        "desc": "FP16带FP32 Bias+A转置",
        "m": 127,
        "k": 257,
        "n": 513,
        "a_dtype": "DT_FP16",
        "b_dtype": "DT_FP16",
        "c_dtype": "DT_FP32",
        "a_format": "ND",
        "b_format": "ND",
        "a_trans": True,
        "b_trans": False,
        "viewshape": [128, 256],
        "tileshape": [[128, 128], [128, 128], [256, 256]],
        "extend_params": {"bias_tensor": {"dtype": "DT_FP32", "shape": [1, 513]}},
        "products": ["950", "910"],
    },
    {
        "id": "E03",
        "name": "int8_scale_bias_relu_trans_b",
        "desc": "INT8 PerTensor+Bias+ReLU+B转置",
        "m": 129,
        "k": 255,
        "n": 513,
        "a_dtype": "DT_INT8",
        "b_dtype": "DT_INT8",
        "c_dtype": "DT_FP16",
        "a_format": "ND",
        "b_format": "ND",
        "a_trans": False,
        "b_trans": True,
        "viewshape": [64, 256],
        "tileshape": [[64, 64], [64, 128], [128, 128]],
        "extend_params": {"scale": 0.125, "bias_tensor": {"dtype": "DT_INT32", "shape": [1, 513]}, "relu_type": "RELU"},
        "products": ["950", "910"],
    },
    {
        "id": "E04",
        "name": "int8_scale_tensor_trans_a",
        "desc": "INT8 PerChannel量化+A转置",
        "m": 129,
        "k": 257,
        "n": 511,
        "a_dtype": "DT_INT8",
        "b_dtype": "DT_INT8",
        "c_dtype": "DT_FP16",
        "a_format": "ND",
        "b_format": "ND",
        "a_trans": True,
        "b_trans": False,
        "viewshape": [128, 256],
        "tileshape": [[128, 128], [128, 128], [256, 256]],
        "extend_params": {"scale_tensor": {"dtype": "DT_UINT64", "shape": [1, 511]}},
        "products": ["950", "910"],
    },
    {
        "id": "E05",
        "name": "bf16_bias_fp32",
        "desc": "BF16带FP32 Bias",
        "m": 129,
        "k": 255,
        "n": 513,
        "a_dtype": "DT_BF16",
        "b_dtype": "DT_BF16",
        "c_dtype": "DT_FP32",
        "a_format": "ND",
        "b_format": "ND",
        "a_trans": False,
        "b_trans": False,
        "viewshape": [64, 256],
        "tileshape": [[64, 64], [64, 128], [128, 128]],
        "extend_params": {"bias_tensor": {"dtype": "DT_FP32", "shape": [1, 513]}},
        "products": ["950", "910"],
    },
    {
        "id": "E06",
        "name": "fp32_tf32_rint",
        "desc": "FP32使能TF32(RINT)",
        "m": 127,
        "k": 257,
        "n": 511,
        "a_dtype": "DT_FP32",
        "b_dtype": "DT_FP32",
        "c_dtype": "DT_FP32",
        "a_format": "ND",
        "b_format": "ND",
        "a_trans": False,
        "b_trans": False,
        "viewshape": [64, 256],
        "tileshape": [[64, 64], [64, 128], [128, 128]],
        "extend_params": {"trans_mode": "CAST_RINT"},
        "products": ["950"],
    },
]

FP8_TESTS = [
    {
        "id": "P01",
        "name": "fp8e5m2_basic",
        "desc": "FP8E5M2基础场景",
        "m": 129,
        "k": 255,
        "n": 513,
        "a_dtype": "DT_FP8E5M2",
        "b_dtype": "DT_FP8E5M2",
        "c_dtype": "DT_FP16",
        "a_format": "ND",
        "b_format": "ND",
        "a_trans": False,
        "b_trans": False,
        "viewshape": [64, 256],
        "tileshape": [[64, 64], [64, 128], [128, 128]],
        "extend_params": {},
        "products": ["950"],
    },
    {
        "id": "P02",
        "name": "fp8e4m3_out_bf16_trans_a",
        "desc": "FP8E4M3输出BF16+A转置",
        "m": 127,
        "k": 255,
        "n": 513,
        "a_dtype": "DT_FP8E4M3",
        "b_dtype": "DT_FP8E4M3",
        "c_dtype": "DT_BF16",
        "a_format": "ND",
        "b_format": "ND",
        "a_trans": True,
        "b_trans": False,
        "viewshape": [128, 256],
        "tileshape": [[128, 128], [128, 128], [256, 256]],
        "extend_params": {},
        "products": ["950"],
    },
    {
        "id": "P03",
        "name": "hf8_out_fp32_trans_b",
        "desc": "HF8输出FP32+B转置",
        "m": 127,
        "k": 257,
        "n": 511,
        "a_dtype": "DT_HF8",
        "b_dtype": "DT_HF8",
        "c_dtype": "DT_FP32",
        "a_format": "ND",
        "b_format": "ND",
        "a_trans": False,
        "b_trans": True,
        "viewshape": [64, 128],
        "tileshape": [[64, 64], [64, 128], [128, 128]],
        "extend_params": {},
        "products": ["950"],
    },
    {
        "id": "P04",
        "name": "fp8_bias_trans_both",
        "desc": "FP8带FP32 Bias+双转置",
        "m": 129,
        "k": 257,
        "n": 511,
        "a_dtype": "DT_FP8E5M2",
        "b_dtype": "DT_FP8E5M2",
        "c_dtype": "DT_FP16",
        "a_format": "ND",
        "b_format": "ND",
        "a_trans": True,
        "b_trans": True,
        "viewshape": [128, 128],
        "tileshape": [[128, 128], [128, 256], [256, 256]],
        "extend_params": {"bias_tensor": {"dtype": "DT_FP32", "shape": [1, 511]}},
        "products": ["950"],
    },
]

SPLIT_K_TESTS = [
    {
        "id": "SK01",
        "name": "int8_split_k_out_int32",
        "desc": "INT8 SplitK输出INT32",
        "m": 129,
        "k": 257,
        "n": 513,
        "a_dtype": "DT_INT8",
        "b_dtype": "DT_INT8",
        "c_dtype": "DT_INT32",
        "a_format": "ND",
        "b_format": "ND",
        "a_trans": False,
        "b_trans": False,
        "viewshape": [64, 256],
        "tileshape": [[64, 64], [64, 128], [128, 128]],
        "extend_params": {},
        "products": ["950", "910"],
    },
    {
        "id": "SK02",
        "name": "fp16_split_k_out_fp32",
        "desc": "FP16 SplitK输出FP32",
        "m": 127,
        "k": 255,
        "n": 511,
        "a_dtype": "DT_FP16",
        "b_dtype": "DT_FP16",
        "c_dtype": "DT_FP32",
        "a_format": "ND",
        "b_format": "ND",
        "a_trans": False,
        "b_trans": False,
        "viewshape": [64, 256],
        "tileshape": [[64, 64], [64, 128], [128, 128]],
        "extend_params": {},
        "products": ["950", "910"],
    },
]
