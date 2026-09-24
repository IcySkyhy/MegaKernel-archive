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
pypto.gather_in_l1 ST测试用例配置
用于System Test自动化测试框架
"""

from dataclasses import dataclass

import torch

import pypto


@dataclass
class GatherInL1Config:
    topk: int
    num_logical_blocks: int
    num_buffer_tokens: int
    token_dim: int
    block_size: int
    is_gather_b_matrix: bool
    is_gather_trans: bool
    out_dtype: pypto.DataType

    DTYPE_CONFIG = {
        "DT_FP16": {"pto": pypto.DT_FP16, "torch": torch.float16, "atol": 1e-3, "rtol": 1e-3},
        "DT_FP32": {"pto": pypto.DT_FP32, "torch": torch.float32, "atol": 1e-3, "rtol": 1e-3},
        "DT_BF16": {"pto": pypto.DT_BF16, "torch": torch.bfloat16, "atol": 1e-2, "rtol": 1e-2},
        "DT_INT8": {"pto": pypto.DT_INT8, "torch": torch.int8, "atol": 0, "rtol": 0},
        "DT_INT32": {"pto": pypto.DT_INT32, "torch": torch.int32, "atol": 0, "rtol": 0},
    }

    @classmethod
    def from_test_case(cls, case: dict) -> "GatherInL1Config":
        return cls(
            topk=case["topk"],
            num_logical_blocks=case["num_logical_blocks"],
            num_buffer_tokens=case["num_buffer_tokens"],
            token_dim=case["token_dim"],
            block_size=case["block_size"],
            is_gather_b_matrix=case["is_b_matrix"],
            is_gather_trans=case["is_trans"],
            out_dtype=cls.DTYPE_CONFIG[case["out_dtype"]]["pto"],
        )

    @classmethod
    def get_torch_dtype(cls, dtype_str: str) -> torch.dtype:
        return cls.DTYPE_CONFIG[dtype_str]["torch"]

    @classmethod
    def get_tolerance(cls, dtype_str: str) -> tuple[float, float]:
        info = cls.DTYPE_CONFIG[dtype_str]
        return info["atol"], info["rtol"]


GATHER_IN_L1_TESTS = [
    {
        "id": "C01",
        "name": "gather_in_l1_32_tokens",
        "desc": "物理token容量32个",
        "topk": 8,
        "num_logical_blocks": 3,
        "num_buffer_tokens": 33,
        "token_dim": 30,
        "block_size": 10,
        "is_b_matrix": False,
        "is_trans": False,
        "in_dtype": "DT_FP16",
        "out_dtype": "DT_FP16",
        "products": ["950", "910"],
    },
    {
        "id": "C02",
        "name": "gather_in_l1_82816_tokens",
        "desc": "物理token容量82816个",
        "topk": 2048,
        "num_logical_blocks": 512,
        "num_buffer_tokens": 82816,
        "token_dim": 64,
        "block_size": 128,
        "is_b_matrix": True,
        "is_trans": True,
        "in_dtype": "DT_INT8",
        "out_dtype": "DT_INT32",
        "products": ["950", "910"],
    },
]
