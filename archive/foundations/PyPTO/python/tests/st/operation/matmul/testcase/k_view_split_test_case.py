#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
K-axis split test case configuration.
Tests the k validshape fix for GM accumulation scenarios
Tests verify correct behavior when k-axis is manually split into multiple blocks with tail blocks
where viewshape > validshape.

可配置参数说明:
- m, k, n: 矩阵形状 (M x K) @ (K x N) = (M x N)
- k_tile: K轴分块大小，用于将K维度切分为多个块
- a_dtype, b_dtype: 输入数据类型 (DT_FP16, DT_BF16, DT_FP32等)
- c_dtype: 输出数据类型 (DT_FP16, DT_BF16, DT_FP32, DT_INT32等)
- a_trans, b_trans: 是否转置A/B矩阵
- cube_tile_shape: Cube单元分块形状 [[m_m, m_n], [k_m, k_n], [n_m, n_n]]
- vec_tile_shape: Vector单元分块形状，根据维度不同而不同
- batch_size: 批大小 (3D BMM使用)
- batch_dims: 批维度数量 (2=2D, 3=3D, 4=4D)
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

import pypto


@dataclass
class KSplitConfig:
    """K-axis split test configuration.

    Unified configuration class supporting both regular matmul and ScaledMM operations.
    """

    m: int
    k: int
    n: int
    k_view: int
    a_dtype: str = "DT_FP16"
    b_dtype: str = "DT_FP16"
    c_dtype: str = "DT_FP32"
    a_trans: bool = False
    b_trans: bool = False
    batch_size: int = 0  # 0 for 2D, >0 for batch
    batch_dims: int = 2  # 2 for 2D, 3 for 3D, 4 for 4D
    cube_tile_shape: Optional[list] = None  # [[m_m, m_n], [k_m, k_n], [n_m, n_n]]
    vec_tile_shape: Optional[list] = None  # For 2D: [m, n]; 3D: [b, m, n]; 4D: [b0, b1, m, n]
    extend_params: Optional[dict] = None  # 扩展参数: bias, relu, scale等
    is_acc: bool = False  # 是否累加场景 (k_split使用atomic_add累加)
    # ScaledMM specific fields (optional)
    m_tile_shape: Optional[Tuple[int, int]] = None
    k_tile_shape: Optional[Tuple[int, int]] = None
    n_tile_shape: Optional[Tuple[int, int]] = None
    scale_a_trans: bool = False
    scale_b_trans: bool = False
    a_format: str = "ND"
    b_format: str = "ND"
    c_format: str = "ND"
    has_bias: bool = False

    def __post_init__(self):
        if self.cube_tile_shape is None:
            # Default cube tile shape
            self.cube_tile_shape = [[64, 64], [32, self.k_view], [32, 64]]
        if self.vec_tile_shape is None:
            # Default vec tile shape based on dimensions
            if self.batch_dims == 2:
                self.vec_tile_shape = [self.m, self.n]
            elif self.batch_dims == 3:
                self.vec_tile_shape = [self.batch_size, self.m, self.n]
            elif self.batch_dims == 4:
                self.vec_tile_shape = [2, 2, self.m, self.n]
        if self.extend_params is None:
            self.extend_params = {}

    DTYPE_CONFIG = {
        "DT_FP16": {"pto": pypto.DT_FP16, "torch": torch.float16, "atol": 1e-3, "rtol": 1e-3},
        "DT_FP32": {"pto": pypto.DT_FP32, "torch": torch.float32, "atol": 1e-3, "rtol": 1e-3},
        "DT_BF16": {"pto": pypto.DT_BF16, "torch": torch.bfloat16, "atol": 1e-2, "rtol": 1e-2},
        "DT_INT8": {"pto": pypto.DT_INT8, "torch": torch.int8, "atol": 0, "rtol": 0},
        "DT_INT32": {"pto": pypto.DT_INT32, "torch": torch.int32, "atol": 0, "rtol": 0},
        "DT_FP8E4M3": {"pto": pypto.DT_FP8E4M3, "torch": torch.float8_e4m3fn, "atol": 1e-3, "rtol": 1e-3},
        "DT_FP8E5M2": {"pto": pypto.DT_FP8E5M2, "torch": torch.float8_e5m2, "atol": 1e-3, "rtol": 1e-3},
    }

    @classmethod
    def from_test_case(cls, case: dict) -> "KSplitConfig":
        """Create a KSplitConfig from a test case dictionary."""
        return cls(
            m=case["m"],
            k=case["k"],
            n=case["n"],
            k_view=case["k_view"],
            m_tile_shape=tuple(case["m_tile_shape"]) if "m_tile_shape" in case else None,
            k_tile_shape=tuple(case["k_tile_shape"]) if "k_tile_shape" in case else None,
            n_tile_shape=tuple(case["n_tile_shape"]) if "n_tile_shape" in case else None,
            a_dtype=case.get("a_dtype", "DT_FP16"),
            b_dtype=case.get("b_dtype", "DT_FP16"),
            c_dtype=case.get("c_dtype", "DT_FP32"),
            scale_a_trans=case.get("scale_a_trans", False),
            scale_b_trans=case.get("scale_b_trans", False),
            a_trans=case.get("a_trans", False),
            b_trans=case.get("b_trans", False),
            a_format=case.get("a_format", "ND"),
            b_format=case.get("b_format", "ND"),
            c_format=case.get("c_format", "ND"),
            has_bias=case.get("has_bias", False),
            is_acc=case.get("is_acc", False),
            batch_size=case.get("batch_size", 0),
            batch_dims=case.get("batch_dims", 2),
            cube_tile_shape=case.get("cube_tile_shape"),
            vec_tile_shape=case.get("vec_tile_shape"),
        )

    @classmethod
    def get_torch_dtype(cls, dtype_str: str) -> torch.dtype:
        return cls.DTYPE_CONFIG[dtype_str]["torch"]

    @classmethod
    def get_tolerance(cls, dtype_str: str) -> tuple[float, float]:
        info = cls.DTYPE_CONFIG[dtype_str]
        return info["atol"], info["rtol"]

    @classmethod
    def get_pto_dtype(cls, dtype_str: str) -> pypto.DataType:
        return cls.DTYPE_CONFIG[dtype_str]["pto"]


K_BLOCK_SIZE_64 = 64
SHAPE_DIM_2 = 2


def _process_scale_tensors(scale_a_cpu, scale_b_cpu, config: KSplitConfig):
    """Process scale tensors for golden computation."""
    m, k, n = config.m, config.k, config.n
    scale_k_32 = (k + K_BLOCK_SIZE_64 - 1) // K_BLOCK_SIZE_64 * SHAPE_DIM_2

    if config.scale_a_trans:
        scale_a_tmp = torch.transpose(scale_a_cpu, -2, -1).reshape(scale_k_32, m).T
    else:
        scale_a_tmp = scale_a_cpu.view(m, scale_k_32)

    if config.scale_b_trans:
        scale_b_tmp = scale_b_cpu.view(n, scale_k_32).T
    else:
        scale_b_tmp = torch.transpose(scale_b_cpu, -2, -1).reshape(scale_k_32, n)

    scale_a_tmp = scale_a_tmp.to(torch.float32).repeat_interleave(32, dim=1)
    scale_b_tmp = scale_b_tmp.to(torch.float32).repeat_interleave(32, dim=0)

    return scale_a_tmp, scale_b_tmp


# 2D Matmul k-split test cases
K_SPLIT_2D_TESTS = [
    {
        "id": "KS2D_01",
        "name": "fp16_2d_k_split_basic",
        "desc": "FP16 2D basic k-split with tail (k=200, tail=8)",
        "m": 64,
        "k": 192,
        "n": 64,
        "k_view": 320,
        "a_dtype": "DT_FP16",
        "b_dtype": "DT_FP16",
        "c_dtype": "DT_FP32",
        "a_trans": False,
        "b_trans": False,
        "is_acc": True,
        "cube_tile_shape": [[64, 64], [64, 128], [64, 64]],
        "vec_tile_shape": [64, 64],
        "batch_dims": 2,
        "products": ["950", "910"],
    },
]

# 3D BMM k-split test cases
K_SPLIT_3D_TESTS = [
    {
        "id": "KS3D_01",
        "name": "fp16_3d_bmm_k_split_basic",
        "desc": "FP16 3D BMM basic k-split with tail (k=200, tail=8)",
        "batch_size": 2,
        "m": 64,
        "k": 192,
        "n": 64,
        "k_view": 320,
        "a_dtype": "DT_FP16",
        "b_dtype": "DT_FP16",
        "c_dtype": "DT_FP32",
        "a_trans": False,
        "b_trans": False,
        "is_acc": True,
        "cube_tile_shape": [[64, 64], [64, 128], [64, 64]],
        "vec_tile_shape": [2, 64, 64],
        "batch_dims": 3,
        "products": ["950", "910"],
    },
]

# ScaledMM k-split test cases
SCALED_MM_K_SPLIT_TESTS = [
    {
        "id": "SKS_01",
        "name": "scaled_mm_k_split_basic",
        "desc": "ScaledMM k-split basic with tail (k=200, tail=8)",
        "m": 128,
        "k": 192,
        "n": 64,
        "k_view": 320,
        "m_tile_shape": [64, 64],
        "k_tile_shape": [64, 128],
        "n_tile_shape": [64, 64],
        "a_dtype": "DT_FP8E4M3",
        "b_dtype": "DT_FP8E4M3",
        "c_dtype": "DT_FP32",
        "scale_a_trans": True,
        "scale_b_trans": False,
        "a_trans": False,
        "b_trans": True,
        "a_format": "ND",
        "b_format": "ND",
        "c_format": "ND",
        "has_bias": False,
        "is_acc": True,
        "batch_dims": 2,
        "products": ["950"],
    },
]
