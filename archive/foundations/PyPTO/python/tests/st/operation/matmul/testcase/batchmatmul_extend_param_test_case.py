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
pypto.matmul fixpipe bias copy test case configuration
For System Test automated testing framework
"""

from dataclasses import dataclass

import torch

import pypto


@dataclass
class BiasFixpipeMatmulConfig:
    input_shape_a: list
    input_shape_b: list
    output_shape: list
    tile_shape: tuple[list, list, list]
    view_shape: list
    a_trans: bool
    b_trans: bool
    a_dtype: str
    b_dtype: str
    c_dtype: str
    bias_shape: list
    bias_dtype: str
    relu_mode: pypto.ReLuType
    mode: str
    scale: float
    products: list

    DTYPE_CONFIG = {
        "DT_FP16": {"pto": pypto.DT_FP16, "torch": torch.float16, "atol": 1e-3, "rtol": 1e-3},
        "DT_FP32": {"pto": pypto.DT_FP32, "torch": torch.float32, "atol": 1e-3, "rtol": 1e-3},
        "DT_BF16": {"pto": pypto.DT_BF16, "torch": torch.bfloat16, "atol": 1e-2, "rtol": 1e-2},
        "DT_INT8": {"pto": pypto.DT_INT8, "torch": torch.int8, "atol": 0, "rtol": 0},
        "DT_INT32": {"pto": pypto.DT_INT32, "torch": torch.int32, "atol": 0, "rtol": 0},
    }

    @classmethod
    def from_test_case(cls, case: dict) -> "BiasFixpipeMatmulConfig":
        return cls(
            input_shape_a=case["input_shape"][0],
            input_shape_b=case["input_shape"][1],
            output_shape=case["output_shape"],
            tile_shape=tuple(case["tile_shape"]),
            view_shape=case["view_shape"],
            a_trans=case["a_trans"],
            b_trans=case["b_trans"],
            a_dtype=case["a_dtype"],
            b_dtype=case["b_dtype"],
            c_dtype=case["c_dtype"],
            bias_shape=case.get("bias_shape", []),
            bias_dtype=case.get("bias_dtype", ""),
            relu_mode=case["relu_mode"],
            mode=case.get("mode", "bias"),
            scale=case.get("scale", 1.0),
            products=case.get("products", ["950", "910"]),
        )

    @classmethod
    def get_torch_dtype(cls, dtype_str: str) -> torch.dtype:
        return cls.DTYPE_CONFIG[dtype_str]["torch"]

    @classmethod
    def get_pto_dtype(cls, dtype_str: str) -> pypto.DataType:
        return cls.DTYPE_CONFIG[dtype_str]["pto"]

    @classmethod
    def get_tolerance(cls, dtype_str: str) -> tuple[float, float]:
        info = cls.DTYPE_CONFIG[dtype_str]
        return info["atol"], info["rtol"]

    def get_m(self) -> int:
        return self.output_shape[-2]

    def get_n(self) -> int:
        return self.output_shape[-1]

    def get_k(self) -> int:
        return self.input_shape_a[-2] if self.a_trans else self.input_shape_a[-1]

    def get_batch_dims(self) -> int:
        return len(self.output_shape) - 2

    def get_batch_sizes(self) -> list:
        return self.output_shape[:-2]

    def get_a_torch_dtype(self) -> torch.dtype:
        return self.get_torch_dtype(self.a_dtype)

    def get_b_torch_dtype(self) -> torch.dtype:
        return self.get_torch_dtype(self.b_dtype)

    def get_c_torch_dtype(self) -> torch.dtype:
        return self.get_torch_dtype(self.c_dtype)

    def get_c_pto_dtype(self) -> pypto.DataType:
        return self.get_pto_dtype(self.c_dtype)

    def get_tile_m(self) -> int:
        return self.view_shape[-2]

    def get_tile_n(self) -> int:
        return self.view_shape[-1]

    def get_tile_batch(self) -> list:
        return self.view_shape[:-2]


BIAS_FIXPIPE_TESTS = [
    {
        "id": "BF01",
        "name": "3d_bias_fp16_relu",
        "desc": "3D BatchMatmul fp16 with Bias and ReLU",
        "a_trans": False,
        "b_trans": False,
        "input_shape": [[3, 127, 255], [3, 255, 512]],
        "output_shape": [3, 127, 512],
        "a_dtype": "DT_FP16",
        "b_dtype": "DT_FP16",
        "c_dtype": "DT_FP32",
        "view_shape": [2, 127, 512],
        "tile_shape": [[128, 128], [128, 128], [128, 128]],
        "bias_shape": [3, 1, 512],
        "relu_mode": pypto.ReLuType.RELU,
        "bias_dtype": "DT_FP32",
        "mode": "bias",
        "products": ["950", "910"],
    },
    {
        "id": "BF02",
        "name": "3d_pertensor_int8_scale",
        "desc": "3D BatchMatmul int8 with PerTensor scale",
        "a_trans": False,
        "b_trans": True,
        "input_shape": [[3, 160, 448], [1, 255, 448]],
        "output_shape": [3, 160, 255],
        "a_dtype": "DT_INT8",
        "b_dtype": "DT_INT8",
        "c_dtype": "DT_FP16",
        "view_shape": [2, 160, 255],
        "tile_shape": [[128, 128], [128, 256], [128, 128]],
        "scale": 2.0,
        "relu_mode": pypto.ReLuType.NO_RELU,
        "mode": "pertensor",
        "products": ["950", "910"],
    },
    {
        "id": "BF03",
        "name": "4d_perchannel_int8",
        "desc": "4D BatchMatmul int8 with PerChannel scale",
        "a_trans": False,
        "b_trans": False,
        "input_shape": [[3, 1, 127, 255], [1, 3, 255, 512]],
        "output_shape": [3, 3, 127, 512],
        "a_dtype": "DT_INT8",
        "b_dtype": "DT_INT8",
        "c_dtype": "DT_FP16",
        "view_shape": [2, 3, 127, 512],
        "tile_shape": [[128, 128], [256, 256], [32, 32]],
        "relu_mode": pypto.ReLuType.NO_RELU,
        "mode": "perchannel",
        "products": ["950", "910"],
    },
    {
        "id": "BF04",
        "name": "3d_bias_bf16_relu",
        "desc": "3D BatchMatmul bf16 with bias and relu",
        "a_trans": False,
        "b_trans": False,
        "input_shape": [[3, 3, 640], [1, 640, 1536]],
        "output_shape": [3, 3, 1536],
        "a_dtype": "DT_BF16",
        "b_dtype": "DT_BF16",
        "c_dtype": "DT_FP32",
        "view_shape": [3, 1, 512],
        "tile_shape": [[256, 256], [64, 128], [128, 128]],
        "bias_shape": [1, 1536],
        "relu_mode": pypto.ReLuType.RELU,
        "bias_dtype": "DT_FP32",
        "mode": "bias",
        "products": ["950", "910"],
    },
]
